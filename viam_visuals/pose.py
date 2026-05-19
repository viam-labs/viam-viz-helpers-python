"""Pose — position + orientation vector + theta.

The Viam world-state-store wire format encodes each entity's pose as
``(x, y, z)`` in millimeters plus an orientation specified by an
orientation vector ``(ox, oy, oz)`` and a rotation ``theta`` (in
degrees) around that vector.

The :class:`Pose` dataclass is the typed surface for this. Identity
is ``OZ=1``, everything else zero — the entity's local +Z aligns
with world +Z. Use :meth:`Pose.at` for the common case of setting
position with the orientation defaulting to identity, or construct
fields explicitly when the orientation matters.

Position is in millimeters, in keeping with the rest of the Viam
convention (the renderer treats file coordinates as meters and
multiplies by 1000; this surface stays in mm so callers don't have
to remember the unit boundary).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Tuple, Union


__all__ = ["Pose", "PoseLike"]


@dataclass
class Pose:
    """Position + orientation in the parent frame.

    The orientation vector ``(ox, oy, oz)`` defaults to ``(0, 0, 1)``
    (identity — local +Z aligned with world +Z). ``theta`` is rotation
    around the orientation vector in degrees.
    """

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    ox: float = 0.0
    oy: float = 0.0
    oz: float = 1.0
    theta: float = 0.0

    @classmethod
    def identity(cls) -> "Pose":
        """The identity pose: origin, OZ=1, theta=0."""
        return cls()

    @classmethod
    def at(
        cls,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        ox: float = 0.0,
        oy: float = 0.0,
        oz: float = 1.0,
        theta: float = 0.0,
    ) -> "Pose":
        """Build a pose with positional defaults — handy when only
        the position matters and orientation should stay identity.

        Example::

            Pose.at(x=500, y=-200, z=100)
        """
        return cls(x=x, y=y, z=z, ox=ox, oy=oy, oz=oz, theta=theta)

    def to_dict(self) -> Mapping[str, float]:
        """Serialize to the wire-format dict the service consumes."""
        return {
            "x": self.x, "y": self.y, "z": self.z,
            "ox": self.ox, "oy": self.oy, "oz": self.oz,
            "theta": self.theta,
        }


# Accepted by every Visual constructor: a Pose, a partial dict (missing
# keys filled with identity defaults), or None (→ identity).
PoseLike = Union[None, Pose, Mapping[str, float]]


def lerp_pose(a: PoseLike, b: PoseLike, t: float) -> Pose:
    """Interpolate between two poses.

    Position (``x``, ``y``, ``z``) is linearly interpolated. The
    orientation is interpolated by *true* quaternion SLERP — the
    orientation vector + theta is converted to a quaternion, SLERPed
    in SO(3), then converted back. This avoids the discontinuities
    of the naive lerp-and-normalize on ``(ox, oy, oz)`` when the
    interpolation path passes through the OV singularity at
    ``|oz| = 1`` (where the renderer's roll reference is unstable).

    ``t`` should be in ``[0, 1]``; values outside that range
    extrapolate (no clamping).

    Useful for motion-plan playback: feed in two adjacent waypoint
    poses from a planner output, interpolate the runner's pose at
    each tick, and the orientation visibly rotates between
    waypoints with no axis flips.

    Example — runner walking a CBiRRT plan with smooth orientation::

        plan = await motion.plan(...)           # CBiRRT planner output
        waypoints = [Pose.at(...), ...]         # after forward-kinematics
        # ...
        def tick(self, scene, t):
            seg = int((t / 12) * (len(waypoints) - 1))
            local = (t / 12 * (len(waypoints) - 1)) - seg
            self.runner.pose = lerp_pose(
                waypoints[seg], waypoints[seg + 1], local,
            )
            return scene.update(self.runner)
    """
    pa = a if isinstance(a, Pose) else _from_like(a)
    pb = b if isinstance(b, Pose) else _from_like(b)
    x = pa.x + (pb.x - pa.x) * t
    y = pa.y + (pb.y - pa.y) * t
    z = pa.z + (pb.z - pa.z) * t
    qa = _ov_to_quat(pa.ox, pa.oy, pa.oz, pa.theta)
    qb = _ov_to_quat(pb.ox, pb.oy, pb.oz, pb.theta)
    qi = _slerp_quat(qa, qb, t)
    ox, oy, oz, theta = _quat_to_ov(*qi)
    return Pose(x=x, y=y, z=z, ox=ox, oy=oy, oz=oz, theta=theta)


def _from_like(p: PoseLike) -> Pose:
    if p is None:
        return Pose.identity()
    if isinstance(p, Pose):
        return p
    if isinstance(p, Mapping):
        return Pose(
            x=float(p.get("x", 0.0)),
            y=float(p.get("y", 0.0)),
            z=float(p.get("z", 0.0)),
            ox=float(p.get("ox", 0.0)),
            oy=float(p.get("oy", 0.0)),
            oz=float(p.get("oz", 1.0)),
            theta=float(p.get("theta", 0.0)),
        )
    raise TypeError(f"unsupported PoseLike: {type(p).__name__}")


_Quat = Tuple[float, float, float, float]


def _ov_to_quat(ox: float, oy: float, oz: float, theta_deg: float) -> _Quat:
    """Convert a Viam orientation vector + theta (degrees) to a unit
    quaternion (w, x, y, z).

    Uses the ZYZ Euler decomposition: ``R = R_z(phi) R_y(delta) R_z(theta)``
    where ``phi = atan2(oy, ox)`` and ``delta = acos(oz)``. At the
    singularity ``|oz| ≈ 1`` (local Z aligned with world ±Z), phi
    becomes undefined; we collapse it into theta.
    """
    theta = math.radians(theta_deg)
    # Match RDK's pole threshold (orientationVectorPoleRadius = 1e-4).
    if 1.0 - abs(oz) <= _POLE_RADIUS:
        # Gimbal-lock-like: local Z aligned with world ±Z. RDK sets
        # lon = 0 (folds phi into theta) and uses ZYZ.
        half_t = theta / 2
        if oz >= 0:
            # R = R_z(0) R_y(0) R_z(theta) = q_z(theta)
            return (math.cos(half_t), 0.0, 0.0, math.sin(half_t))
        # R = R_z(0) R_y(pi) R_z(theta) = q_y(pi) * q_z(theta)
        # = (0, 0, 1, 0) * (cos(t/2), 0, 0, sin(t/2)) = (0, sin(t/2), cos(t/2), 0)
        return (0.0, math.sin(half_t), math.cos(half_t), 0.0)

    phi = math.atan2(oy, ox)
    delta = math.acos(max(-1.0, min(1.0, oz)))
    cp, sp = math.cos(phi / 2), math.sin(phi / 2)
    cd, sd = math.cos(delta / 2), math.sin(delta / 2)
    ct, st = math.cos(theta / 2), math.sin(theta / 2)
    # q_y(delta) * q_z(theta) = (cd*ct, sd*st, sd*ct, cd*st)
    a, b, c, d = cd * ct, sd * st, sd * ct, cd * st
    # q_z(phi) * (a, b, c, d)
    w = cp * a - sp * d
    x = cp * b - sp * c
    y = cp * c + sp * b
    z = cp * d + sp * a
    return (w, x, y, z)


_POLE_RADIUS = 1e-4  # matches RDK's orientationVectorPoleRadius / defaultAngleEpsilon


def _quat_to_ov(w: float, x: float, y: float, z: float) -> Tuple[float, float, float, float]:
    """Inverse of :func:`_ov_to_quat`. Returns ``(ox, oy, oz, theta_deg)``.

    Ported from RDK's ``spatialmath.QuatToOV`` so the OV → R map used
    by the renderer round-trips exactly. The previous ZYZ-Euler
    extraction split the total rotation between ``phi`` and ``theta``
    using a 1e-12 pole threshold; the renderer applied a 1e-4 pole
    threshold and dropped the ``phi`` contribution, producing a
    visible ~45° flip per tick as the interpolation approached
    ``|oz| = 1``.
    """
    # Rotated +Z (newZ) and rotated -X (newX), per RDK convention.
    nz = (
        2.0 * (x * z + w * y),
        2.0 * (y * z - w * x),
        1.0 - 2.0 * (x * x + y * y),
    )
    nx = (
        -(1.0 - 2.0 * (y * y + z * z)),
        -(2.0 * (x * y + w * z)),
        -(2.0 * (x * z - w * y)),
    )
    ox, oy, oz = nz

    if 1.0 - abs(oz) > _POLE_RADIUS:
        # Non-pole: theta is the angle between the plane (newZ, newX, origin)
        # and the plane (newZ, world+Z, origin), measured around newZ.
        n1 = (
            nz[1] * nx[2] - nz[2] * nx[1],
            nz[2] * nx[0] - nz[0] * nx[2],
            nz[0] * nx[1] - nz[1] * nx[0],
        )
        n2 = (nz[1], -nz[0], 0.0)
        n1_dot_n2 = n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2]
        n1_len = math.sqrt(n1[0] ** 2 + n1[1] ** 2 + n1[2] ** 2)
        n2_len = math.sqrt(n2[0] ** 2 + n2[1] ** 2 + n2[2] ** 2)
        denom = n1_len * n2_len
        if denom == 0.0:
            return (ox, oy, oz, 0.0)
        cos_theta = max(-1.0, min(1.0, n1_dot_n2 / denom))
        theta = math.acos(cos_theta)
        if theta > _POLE_RADIUS:
            # Sign disambiguation: rotate newZ by -theta around (ox, oy, oz)
            # and check whether the result is coplanar with the (newZ, world+Z)
            # plane. Coplanar → theta is negative.
            half_t = -theta / 2.0
            sin_h = math.sin(half_t)
            q2 = (math.cos(half_t), ox * sin_h, oy * sin_h, oz * sin_h)
            q2w, q2x, q2y, q2z = q2
            tz = (
                2.0 * (q2x * q2z + q2w * q2y),
                2.0 * (q2y * q2z - q2w * q2x),
                1.0 - 2.0 * (q2x * q2x + q2y * q2y),
            )
            n3 = (
                nz[1] * tz[2] - nz[2] * tz[1],
                nz[2] * tz[0] - nz[0] * tz[2],
                nz[0] * tz[1] - nz[1] * tz[0],
            )
            n3_len = math.sqrt(n3[0] ** 2 + n3[1] ** 2 + n3[2] ** 2)
            if n3_len == 0.0:
                pass  # leave theta positive
            else:
                cos_test = (n1[0] * n3[0] + n1[1] * n3[1] + n1[2] * n3[2]) / (n1_len * n3_len)
                if 1.0 - cos_test < _POLE_RADIUS * _POLE_RADIUS:
                    theta = -theta
        else:
            theta = 0.0
    else:
        # Pole: extract from the rotated -X direction.
        if oz >= 0:
            theta = -math.atan2(nx[1], -nx[0])
        else:
            theta = -math.atan2(nx[1], nx[0])

    if theta == 0.0:
        theta = 0.0  # collapse -0.0 to +0.0
    return (ox, oy, oz, math.degrees(theta))


def _slerp_quat(qa: _Quat, qb: _Quat, t: float) -> _Quat:
    """Spherical linear interpolation between two unit quaternions.

    Picks the shorter great-circle arc (negates ``qb`` if the dot
    product is negative). Falls back to a normalized linear interpolation
    for nearly-parallel quaternions where the SLERP denominator goes
    to zero.
    """
    w1, x1, y1, z1 = qa
    w2, x2, y2, z2 = qb
    dot = w1 * w2 + x1 * x2 + y1 * y2 + z1 * z2
    if dot < 0.0:
        w2, x2, y2, z2 = -w2, -x2, -y2, -z2
        dot = -dot
    if dot > 0.9995:
        w = w1 + (w2 - w1) * t
        x = x1 + (x2 - x1) * t
        y = y1 + (y2 - y1) * t
        z = z1 + (z2 - z1) * t
    else:
        theta_0 = math.acos(max(-1.0, min(1.0, dot)))
        sin_theta_0 = math.sin(theta_0)
        theta = theta_0 * t
        sin_theta = math.sin(theta)
        s1 = math.cos(theta) - dot * sin_theta / sin_theta_0
        s2 = sin_theta / sin_theta_0
        w = s1 * w1 + s2 * w2
        x = s1 * x1 + s2 * x2
        y = s1 * y1 + s2 * y2
        z = s1 * z1 + s2 * z2
    n = math.sqrt(w * w + x * x + y * y + z * z)
    return (w / n, x / n, y / n, z / n)


def normalize_pose(p: PoseLike) -> Mapping[str, float]:
    """Coerce a PoseLike into the full dict the wire format expects.

    None → identity. Mapping → fill missing keys from identity. Pose
    → to_dict(). Anything else raises ``TypeError``.
    """
    if p is None:
        return Pose.identity().to_dict()
    if isinstance(p, Pose):
        return p.to_dict()
    if isinstance(p, Mapping):
        out = dict(Pose.identity().to_dict())
        out.update({k: float(v) for k, v in p.items()})
        return out
    raise TypeError(f"pose must be None | Pose | dict; got {type(p).__name__}")
