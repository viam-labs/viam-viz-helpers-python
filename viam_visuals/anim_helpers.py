"""Imperative pose-composing helpers for animation.

These functions take a "base" pose (the rest state of the entity)
plus a time ``t`` (elapsed seconds), and return a new :class:`Pose`
with the appropriate animation math applied. They're pure — the
input ``base`` is not mutated.

Pair with :meth:`SceneServiceBase.scene_tick`::

    def __init__(self, name):
        super().__init__(name)
        self.box = viz.Box("demo", pose=viz.Pose.at(z=100),
                           dims_mm=(120, 120, 120), color=(255, 100, 0))
        self._base_pose = self.box.pose

    def scene_tick(self, scene, t):
        self.box.pose = viz.spin_pose(self._base_pose, period_s=3, t=t)
        return scene.update(self.box)

The helpers below cover the simple "absolute t → pose" animation
modes. For animations that mutate non-pose fields (radius pulse,
color cycle, etc.), mutate the field directly in your scene_tick;
no helper is needed. See ``standalone-playground``'s
``src/animation.py`` for the full set of modes (still on the
legacy ``compute_tick`` path).
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Literal

from .pose import Pose, PoseLike, normalize_pose


__all__ = [
    "spin_pose",
    "orbit_pose",
    "oscillate_pose",
    "swing_pose",
    "pulse_range",
    "trajectory_pose",
]


_Axis = Literal["x", "y", "z"]


def _as_pose(p: PoseLike) -> Pose:
    """Coerce a PoseLike into a Pose for ``replace`` semantics."""
    if isinstance(p, Pose):
        return p
    d = normalize_pose(p)
    return Pose(
        x=float(d["x"]), y=float(d["y"]), z=float(d["z"]),
        ox=float(d["ox"]), oy=float(d["oy"]), oz=float(d["oz"]),
        theta=float(d["theta"]),
    )


def spin_pose(base: PoseLike, period_s: float, t: float) -> Pose:
    """Return ``base`` with ``theta`` set to a continuous rotation
    of ``360° / period_s`` per second.

    Theta is computed as ``(360° × t / period_s) mod 360°`` — an
    absolute value (base.theta is ignored), so callers don't need
    to track accumulated angle.
    """
    b = _as_pose(base)
    return replace(b, theta=(360.0 * t / period_s) % 360.0)


def orbit_pose(
    base: PoseLike,
    period_s: float,
    radius_mm: float,
    t: float,
    axis: _Axis = "z",
) -> Pose:
    """Return ``base`` with position orbiting in a circle of
    ``radius_mm`` around ``base``'s position, in the plane
    perpendicular to ``axis``.

    Defaults to ``axis="z"`` (orbit in the world XY plane — the
    common "boat circling a buoy" case). Use ``"y"`` for an orbit
    in XZ (vertical loop) or ``"x"`` for YZ.
    """
    b = _as_pose(base)
    phase = 2 * math.pi * t / period_s
    c, s = math.cos(phase), math.sin(phase)
    if axis == "z":
        return replace(b, x=b.x + radius_mm * c, y=b.y + radius_mm * s)
    if axis == "y":
        return replace(b, x=b.x + radius_mm * c, z=b.z + radius_mm * s)
    if axis == "x":
        return replace(b, y=b.y + radius_mm * c, z=b.z + radius_mm * s)
    raise ValueError(f"axis must be 'x', 'y', or 'z'; got {axis!r}")


def oscillate_pose(
    base: PoseLike,
    period_s: float,
    amplitude_mm: float,
    t: float,
    axis: _Axis = "y",
) -> Pose:
    """Return ``base`` with ``axis`` offset by ``amplitude_mm × sin(2π t / period_s)``.

    Defaults to ``axis="y"`` (the common bob-up-bob-down via Y; flip
    to ``"z"`` if your "up" is world-Z).
    """
    b = _as_pose(base)
    delta = amplitude_mm * math.sin(2 * math.pi * t / period_s)
    if axis == "x":
        return replace(b, x=b.x + delta)
    if axis == "y":
        return replace(b, y=b.y + delta)
    if axis == "z":
        return replace(b, z=b.z + delta)
    raise ValueError(f"axis must be 'x', 'y', or 'z'; got {axis!r}")


def swing_pose(
    base: PoseLike,
    period_s: float,
    amplitude_deg: float,
    t: float,
) -> Pose:
    """Return ``base`` with theta swinging sinusoidally around
    ``base.theta``: ``theta = base.theta + amplitude_deg × sin(2π t / period_s)``.

    Unlike :func:`spin_pose`, swing is relative to the base's theta
    — useful for pendulum-like motion where the "rest" theta
    matters.
    """
    b = _as_pose(base)
    return replace(
        b,
        theta=b.theta + amplitude_deg * math.sin(2 * math.pi * t / period_s),
    )


def pulse_range(lo: float, hi: float, period_s: float, t: float) -> float:
    """Return a sinusoidal value swinging between ``lo`` and ``hi``
    with the given period.

    Use for any scalar field that should breathe between two
    extremes — sphere radius, box dim, opacity, etc.

    Example — box that pulses between 80 mm and 160 mm at period 2 s::

        scale = viz.pulse_range(80, 160, period_s=2, t=t)
        self.box.dims_mm = (scale, scale, scale)
        return scene.update(self.box)

    Equivalent to ``base + amplitude × sin(2π t / period_s)`` where
    ``base = (lo + hi) / 2`` and ``amplitude = (hi - lo) / 2``.
    """
    base = (lo + hi) / 2.0
    amplitude = (hi - lo) / 2.0
    return base + amplitude * math.sin(2 * math.pi * t / period_s)


def trajectory_pose(
    waypoints: list,
    duration_s: float,
    t: float,
    loop: bool = True,
) -> Pose:
    """Return an interpolated pose along a multi-waypoint trajectory.

    Walks ``waypoints`` over ``duration_s`` seconds, lerping between
    adjacent pairs with :func:`lerp_pose` (quaternion SLERP on
    orientation). With ``loop=True``, the trajectory restarts (snap
    back) once ``t`` exceeds ``duration_s``; with ``loop=False``,
    clamps to the final waypoint.

    The waypoint list should match the shape of a planner output
    (CBiRRT / RRT* / motion-service): each element is a PoseLike
    (Pose, Pose-shaped dict, or None for identity).

    Example — runner walking a 5-waypoint plan over 12 seconds::

        plan = [Pose.at(...), Pose.at(...), ...]
        # In scene_tick:
        self.runner.pose = viz.trajectory_pose(plan, 12.0, t)
        return scene.update(self.runner)
    """
    # Local import to avoid a top-level cycle (lerp_pose lives in
    # pose.py which this module already imports from).
    from .pose import lerp_pose

    n = len(waypoints)
    if n < 2:
        raise ValueError(f"trajectory_pose needs ≥ 2 waypoints; got {n}")
    n_segs = n - 1
    if loop:
        progress = (t / duration_s * n_segs) % n_segs
    else:
        progress = max(0.0, min(float(n_segs), t / duration_s * n_segs))
    seg_idx = int(progress)
    if seg_idx >= n_segs:
        seg_idx = n_segs - 1
    local = progress - seg_idx
    return lerp_pose(waypoints[seg_idx], waypoints[seg_idx + 1], local)
