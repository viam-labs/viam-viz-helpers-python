"""Tests for :func:`viam_visuals.lerp_pose` — quaternion SLERP on the
orientation vector + theta encoding.

The orientation vector representation has a singularity at ``|oz| = 1``
(local Z aligned with world ±Z) where ``phi = atan2(oy, ox)`` becomes
undefined. The naive lerp-and-normalize on ``(ox, oy, oz)`` produces
visible flips as the interpolation approaches this region; true
quaternion SLERP does not.
"""

import math

import pytest

from viam_visuals import Pose, lerp_pose
from viam_visuals.pose import _ov_to_quat, _quat_to_ov


def _rotate_vec_by_quat(q, v):
    """Apply rotation quaternion (w, x, y, z) to a 3-vector."""
    w, x, y, z = q
    vx, vy, vz = v
    # q * (0, v) * q^-1
    # Standard quaternion rotation formula
    rx = (1 - 2*(y*y + z*z))*vx + 2*(x*y - w*z)*vy + 2*(x*z + w*y)*vz
    ry = 2*(x*y + w*z)*vx + (1 - 2*(x*x + z*z))*vy + 2*(y*z - w*x)*vz
    rz = 2*(x*z - w*y)*vx + 2*(y*z + w*x)*vy + (1 - 2*(x*x + y*y))*vz
    return (rx, ry, rz)


def _rotations_equivalent(q1, q2, tol=1e-6):
    """Two quaternions represent the same rotation if rotating any
    test vector produces the same result. Test with three linearly-
    independent probes."""
    for v in [(1, 0, 0), (0, 1, 0), (1, 2, 3)]:
        v1 = _rotate_vec_by_quat(q1, v)
        v2 = _rotate_vec_by_quat(q2, v)
        for a, b in zip(v1, v2, strict=True):
            if abs(a - b) > tol:
                return False
    return True


# ---- round-trip OV ↔ quat ---------------------------------------------------

@pytest.mark.parametrize("ox,oy,oz,theta", [
    (0, 0, 1, 0),       # identity
    (0, 0, 1, 90),      # roll around world Z
    (1, 0, 0, 0),       # tipped: local Z → world +X
    (0, 1, 0, 0),       # tipped: local Z → world +Y
    (0, 0, -1, 0),      # flipped: local Z → world -Z
    (1/math.sqrt(2), 0, 1/math.sqrt(2), 45),  # generic
    (0.5, 0.5, 1/math.sqrt(2), 30),           # generic
])
def test_ov_to_quat_round_trip_preserves_rotation(ox, oy, oz, theta):
    """OV → quat → OV should preserve the underlying rotation,
    even if the (ox,oy,oz,θ) tuple itself changes near singularities."""
    q1 = _ov_to_quat(ox, oy, oz, theta)
    ox2, oy2, oz2, theta2 = _quat_to_ov(*q1)
    q2 = _ov_to_quat(ox2, oy2, oz2, theta2)
    assert _rotations_equivalent(q1, q2), (
        f"round-trip changed rotation: q1={q1} q2={q2} "
        f"OV in=({ox},{oy},{oz},{theta}) out=({ox2},{oy2},{oz2},{theta2})"
    )


# ---- SLERP endpoints --------------------------------------------------------

def test_lerp_pose_at_t_zero_returns_first_pose():
    a = Pose.at(x=10, y=20, z=30, ox=1, oy=0, oz=0, theta=0)
    b = Pose.at(x=40, y=50, z=60, ox=0, oy=0, oz=1, theta=90)
    r = lerp_pose(a, b, 0.0)
    assert abs(r.x - a.x) < 1e-6
    assert abs(r.y - a.y) < 1e-6
    assert abs(r.z - a.z) < 1e-6
    qa = _ov_to_quat(a.ox, a.oy, a.oz, a.theta)
    qr = _ov_to_quat(r.ox, r.oy, r.oz, r.theta)
    assert _rotations_equivalent(qa, qr, tol=1e-5)


def test_lerp_pose_at_t_one_returns_second_pose():
    a = Pose.at(x=10, y=20, z=30, ox=1, oy=0, oz=0, theta=0)
    b = Pose.at(x=40, y=50, z=60, ox=0, oy=0, oz=1, theta=90)
    r = lerp_pose(a, b, 1.0)
    assert abs(r.x - b.x) < 1e-6
    assert abs(r.y - b.y) < 1e-6
    assert abs(r.z - b.z) < 1e-6
    qb = _ov_to_quat(b.ox, b.oy, b.oz, b.theta)
    qr = _ov_to_quat(r.ox, r.oy, r.oz, r.theta)
    assert _rotations_equivalent(qb, qr, tol=1e-5)


def test_lerp_pose_position_is_linear():
    a = Pose.at(x=0, y=0, z=0)
    b = Pose.at(x=100, y=200, z=300)
    r = lerp_pose(a, b, 0.5)
    assert abs(r.x - 50) < 1e-6
    assert abs(r.y - 100) < 1e-6
    assert abs(r.z - 150) < 1e-6


# ---- Continuity through the OV singularity (the bug fix) ------------------

def test_lerp_pose_smooth_through_oz_singularity():
    """The original bug: lerping wp1 (OX=1, θ=0) → wp2 (OZ=1, θ=90)
    passes through the OV singularity at |oz| = 1. Naive lerp produces
    visible flips; SLERP does not.

    Test: the angular distance between consecutive interpolated samples
    should be bounded — no sample should rotate by more than the
    average rotation per step plus a small tolerance."""
    a = Pose.at(ox=1, oy=0, oz=0, theta=0)
    b = Pose.at(ox=0, oy=0, oz=1, theta=90)
    n = 100
    quats = []
    for i in range(n + 1):
        t = i / n
        r = lerp_pose(a, b, t)
        quats.append(_ov_to_quat(r.ox, r.oy, r.oz, r.theta))

    # Compute angular distances between consecutive samples.
    angles = []
    for i in range(len(quats) - 1):
        q1, q2 = quats[i], quats[i + 1]
        dot = abs(q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3])
        dot = max(-1.0, min(1.0, dot))
        angles.append(2 * math.acos(dot))

    avg = sum(angles) / len(angles)
    max_a = max(angles)
    # With true SLERP each step has identical angular distance — the
    # tolerance is purely for floating-point noise.
    assert max_a < avg * 1.05, (
        f"largest step {math.degrees(max_a):.4f}° vs average "
        f"{math.degrees(avg):.4f}° — interpolation is not constant-speed "
        f"SLERP (flips through singularity)")


def test_lerp_pose_midpoint_of_extreme_rotation_is_halfway_in_so3():
    """At t=0.5 between OX=1,θ=0 and OZ=1,θ=90, the result should be
    the halfway rotation in SO(3) — not the discontinuous lerp result."""
    a = Pose.at(ox=1, oy=0, oz=0, theta=0)
    b = Pose.at(ox=0, oy=0, oz=1, theta=90)
    r = lerp_pose(a, b, 0.5)
    qa = _ov_to_quat(a.ox, a.oy, a.oz, a.theta)
    qb = _ov_to_quat(b.ox, b.oy, b.oz, b.theta)
    qr = _ov_to_quat(r.ox, r.oy, r.oz, r.theta)
    # Halfway in SO(3): angle(qa, qr) should equal angle(qr, qb).
    def ang(q1, q2):
        dot = abs(q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3])
        return 2 * math.acos(max(-1.0, min(1.0, dot)))
    a_to_mid = ang(qa, qr)
    mid_to_b = ang(qr, qb)
    assert abs(a_to_mid - mid_to_b) < 1e-4, (
        f"midpoint not halfway in SO(3): a→mid={math.degrees(a_to_mid):.3f}° "
        f"mid→b={math.degrees(mid_to_b):.3f}°")


def test_lerp_pose_takes_shorter_arc():
    """SLERP should take the shorter great-circle arc between
    rotations. Construct two rotations whose direct quaternion lerp
    would go the long way unless the dot-product sign correction kicks
    in."""
    a = Pose.at(ox=0, oy=0, oz=1, theta=0)
    b = Pose.at(ox=0, oy=0, oz=1, theta=270)  # Equivalent to -90
    # Equivalent rotations: 270° is the same as -90° via the shorter
    # arc. SLERP should travel ~90° not ~270°.
    r = lerp_pose(a, b, 0.5)
    # At t=0.5 the midpoint should be 45° in the SHORT direction, i.e.
    # equivalent to theta = -45° = 315° (mod 360).
    # Verify by computing the angular distance from a to r — should be
    # ~45° not ~135°.
    qa = _ov_to_quat(a.ox, a.oy, a.oz, a.theta)
    qr = _ov_to_quat(r.ox, r.oy, r.oz, r.theta)
    dot = abs(qa[0]*qr[0] + qa[1]*qr[1] + qa[2]*qr[2] + qa[3]*qr[3])
    angle = 2 * math.degrees(math.acos(max(-1.0, min(1.0, dot))))
    assert abs(angle - 45) < 1.0, (
        f"expected ~45° (short arc), got {angle:.3f}° — "
        f"SLERP not taking the shorter path")
