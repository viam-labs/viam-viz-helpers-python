"""Composite shapes — multi-Visual constructs.

Composites are the "second tier" of the library's typed surface.
Each one packages a common multi-item pattern (a coordinate-frame
triad, a polyline drawn as a capsule chain, a wireframe bounding
box) behind a single typed object so callers don't have to rebuild
the same hand-wired item lists every time.

Each composite is a dataclass; call :meth:`Composite.to_visuals` to
get the underlying ``Visual`` list, or iterate the composite directly
— it implements ``__iter__``. Visual instances inside a composite
follow consistent label conventions (e.g.,
``CoordinateFrame("tcp").to_visuals()`` produces ``"tcp"`` for the
anchor and ``"tcp_axis_x"`` / ``"_y"`` / ``"_z"`` for the axes).

Mutation pattern
----------------

Composites can be passed to :meth:`viam_visuals.Scene.add` and
:meth:`viam_visuals.Scene.update`. The Scene expands the composite,
diffs each constituent, and emits one event per changed Visual.
After mutating a composite field (``line.width_mm = 12``,
``plan.waypoints = [...]``), call ``scene.update(composite)`` to
re-expand and emit the deltas.

The scene tracks constituents by label — internal sub-Visual labels
follow a deterministic pattern per composite class (documented on
each). Two composites with the same label prefix WILL collide; use
the optional ``namespace`` on the driver to scope them.

Single-item "composites" — like a point-to-point arrow — live on
the primitive classes themselves as alternate constructors (e.g.,
:meth:`viam_visuals.Arrow.from_to`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterator, List, Optional, Sequence, Tuple, Union

from .animations import AnimationLike
from .color import ColorLike
from .pose import Pose, PoseLike, normalize_pose
from .shapes import Arrow, Box, Capsule, Sphere, Visual


__all__ = [
    "Composite",
    "CoordinateFrame",
    "Line",
    "BoundingBox",
    "TrajectoryPlan",
]


@dataclass
class Composite:
    """Base class for multi-Visual composites. Subclasses override
    :meth:`to_visuals` to return the expanded list."""

    def to_visuals(self) -> List[Visual]:  # pragma: no cover - abstract
        raise NotImplementedError

    def __iter__(self) -> Iterator[Visual]:
        return iter(self.to_visuals())


@dataclass
class CoordinateFrame(Composite):
    """A coordinate-frame triad: an anchor sphere plus three colored
    axis arrows parented to the anchor.

    Use this whenever you'd otherwise hand-build a parent-anchor
    sphere plus three axis-aligned arrows — the
    parent-frame-composition pattern that powers the playground's
    ``reference_frame_demo`` preset.

    The three axes are :class:`Arrow` primitives (cylindrical shaft
    + conical tip) so direction is visually obvious from the tip.
    Each arrow's tail sits at the frame origin, its tip pointing
    along the corresponding axis. ``axis_length_mm`` and
    ``axis_radius_mm`` parameterize the arrow size; per-axis colors
    can each be customized independently.

    Animations attach to the anchor; the axes inherit motion through
    the parent-frame chain. To probe whether the renderer composes
    through chained parents, set an animation here and watch the
    triad sweep coherently.

    The anchor's ``show_axes_helper`` defaults to ``True`` — the
    renderer's built-in RGB axes triad renders alongside the
    composite's explicit arrows, which is occasionally redundant but
    useful when you've tinted the arrow colors away from R/G/B.

    Internal labels: anchor uses ``label``; axes use
    ``f"{label}_axis_x"`` / ``"_y"`` / ``"_z"`` so they're unique
    even when multiple frames coexist in one scene.
    """

    label: str
    pose: PoseLike = None
    size_mm: float = 100.0
    parent_frame: Optional[str] = None
    animation: Any = None  # AnimationLike
    show_axes_helper: bool = True
    anchor_radius_mm: float = 12.0
    axis_radius_mm: float = 8.0
    axis_length_mm: Optional[float] = None  # defaults to size_mm
    anchor_color: ColorLike = (120, 120, 120)  # gray
    anchor_opacity: Optional[float] = 0.6
    axis_color_x: ColorLike = (230, 25, 75)
    axis_color_y: ColorLike = (60, 180, 75)
    axis_color_z: ColorLike = (0, 130, 200)
    axis_opacity: Optional[float] = 1.0

    def to_visuals(self) -> List[Visual]:
        axis_len = float(
            self.axis_length_mm if self.axis_length_mm is not None
            else self.size_mm
        )
        out: List[Visual] = [
            Sphere(
                self.label,
                pose=self.pose,
                parent_frame=self.parent_frame,
                radius_mm=self.anchor_radius_mm,
                color=self.anchor_color,
                opacity=self.anchor_opacity,
                show_axes_helper=self.show_axes_helper,
                animation=self.animation,
            ),
            # Arrow's tail is at its origin; local +Z is the shaft
            # direction. Orient each axis arrow so its local +Z
            # aligns with the corresponding world axis.
            Arrow(
                f"{self.label}_axis_x",
                parent_frame=self.label,
                pose=Pose.at(ox=1, oy=0, oz=0),
                radius_mm=self.axis_radius_mm,
                length_mm=axis_len,
                color=self.axis_color_x,
                opacity=self.axis_opacity,
            ),
            Arrow(
                f"{self.label}_axis_y",
                parent_frame=self.label,
                pose=Pose.at(ox=0, oy=1, oz=0),
                radius_mm=self.axis_radius_mm,
                length_mm=axis_len,
                color=self.axis_color_y,
                opacity=self.axis_opacity,
            ),
            Arrow(
                f"{self.label}_axis_z",
                parent_frame=self.label,
                # Default identity orientation (+Z up).
                pose=None,
                radius_mm=self.axis_radius_mm,
                length_mm=axis_len,
                color=self.axis_color_z,
                opacity=self.axis_opacity,
            ),
        ]
        return out


@dataclass
class Line(Composite):
    """A polyline drawn as a chain of capsule segments. The wire
    format has no first-class line primitive (see
    ``LESSONS.md::viewer-has-a-second-wire-format-we-cant-emit``);
    this composite synthesizes one from capsules whose local +Z is
    aligned to each segment's direction.

    ``points`` is a sequence of poses or pose-like dicts; only the
    position fields are read. ``width_mm`` is the visual line
    thickness (each capsule's diameter). Successive points further
    apart than ~1 µm get a segment between them; coincident points
    are skipped silently.

    Internal labels: segments use ``f"{label_prefix}_seg_NN"``.
    """

    label_prefix: str
    points: Sequence[Any] = field(default_factory=list)  # Pose or pose-like
    width_mm: float = 4.0
    parent_frame: Optional[str] = None
    color: ColorLike = None
    opacity: Optional[float] = 1.0

    def __post_init__(self) -> None:
        if len(self.points) < 2:
            raise ValueError(
                f"Line needs at least 2 points; got {len(self.points)}"
            )

    def to_visuals(self) -> List[Visual]:
        out: List[Visual] = []
        # Normalize each point to a full pose dict (so callers can pass
        # Pose, pose dict, or any sub-dict).
        pts = [normalize_pose(p) for p in self.points]
        seg_idx = 0
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            dx = b["x"] - a["x"]
            dy = b["y"] - a["y"]
            dz = b["z"] - a["z"]
            seg_len = math.sqrt(dx * dx + dy * dy + dz * dz)
            if seg_len < 1e-6:
                continue
            out.append(Capsule(
                f"{self.label_prefix}_seg_{seg_idx:02d}",
                parent_frame=self.parent_frame,
                pose=Pose.at(
                    x=(a["x"] + b["x"]) / 2.0,
                    y=(a["y"] + b["y"]) / 2.0,
                    z=(a["z"] + b["z"]) / 2.0,
                    ox=dx / seg_len, oy=dy / seg_len, oz=dz / seg_len,
                ),
                radius_mm=self.width_mm / 2.0,
                length_mm=seg_len,
                color=self.color,
                opacity=self.opacity,
            ))
            seg_idx += 1
        return out


@dataclass
class BoundingBox(Composite):
    """Axis-aligned bounding box.

    With ``wireframe=False`` (default), produces a single solid
    :class:`viam_visuals.Box`. With ``wireframe=True``, produces 12
    capsule edges tracing the box outline — useful for object-
    detection overlays where you want the bounds without occluding
    what's inside.

    ``dims_mm`` is ``(x, y, z)`` in millimeters. ``edge_radius_mm``
    sets the wireframe edge thickness.

    Internal labels (wireframe mode): ``f"{label}_edge_NN"`` for the
    12 edges, indexed in (x, y, z) order.
    """

    label: str
    dims_mm: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    pose: PoseLike = None
    parent_frame: Optional[str] = None
    wireframe: bool = False
    color: ColorLike = None
    opacity: Optional[float] = 1.0
    edge_radius_mm: float = 2.0

    def __post_init__(self) -> None:
        if len(self.dims_mm) != 3 or any(d <= 0 for d in self.dims_mm):
            raise ValueError(
                f"BoundingBox.dims_mm must be 3 positive components; "
                f"got {self.dims_mm!r}"
            )

    def to_visuals(self) -> List[Visual]:
        if not self.wireframe:
            return [Box(
                self.label, pose=self.pose, parent_frame=self.parent_frame,
                dims_mm=self.dims_mm, color=self.color, opacity=self.opacity,
            )]

        dx, dy, dz = (float(d) for d in self.dims_mm)
        hx, hy, hz = dx / 2.0, dy / 2.0, dz / 2.0
        out: List[Visual] = []
        i = 0

        # 4 X-edges (length dx, oriented along ±X). Position varies
        # in (y, z) at the four corners of the YZ face.
        for sy in (-1, 1):
            for sz in (-1, 1):
                out.append(Capsule(
                    f"{self.label}_edge_{i:02d}",
                    parent_frame=self.parent_frame,
                    pose=Pose.at(y=sy * hy, z=sz * hz, ox=1, oy=0, oz=0),
                    radius_mm=self.edge_radius_mm, length_mm=dx,
                    color=self.color, opacity=self.opacity,
                ))
                i += 1
        # 4 Y-edges.
        for sx in (-1, 1):
            for sz in (-1, 1):
                out.append(Capsule(
                    f"{self.label}_edge_{i:02d}",
                    parent_frame=self.parent_frame,
                    pose=Pose.at(x=sx * hx, z=sz * hz, ox=0, oy=1, oz=0),
                    radius_mm=self.edge_radius_mm, length_mm=dy,
                    color=self.color, opacity=self.opacity,
                ))
                i += 1
        # 4 Z-edges.
        for sx in (-1, 1):
            for sy in (-1, 1):
                out.append(Capsule(
                    f"{self.label}_edge_{i:02d}",
                    parent_frame=self.parent_frame,
                    pose=Pose.at(x=sx * hx, y=sy * hy),
                    radius_mm=self.edge_radius_mm, length_mm=dz,
                    color=self.color, opacity=self.opacity,
                ))
                i += 1
        return out


@dataclass
class TrajectoryPlan(Composite):
    """Visualization for a motion plan — a list of poses with
    orientation. Expands to a polyline connecting the waypoints
    plus a :class:`CoordinateFrame` triad at each waypoint so the
    orientation at each step is visible.

    Designed to match the shape of motion-planner output
    (CBiRRT, RRT*, motion-service plans). After forward-kinematics
    on a planner's joint-position output, each step is a Cartesian
    pose; pass that list in as ``waypoints`` and the composite
    renders the plan.

    Example::

        from viam_visuals import Scene, TrajectoryPlan, Pose

        plan = [
            Pose.at(x=-400, y=-300, z=100),                       # wp 0
            Pose.at(x=-200, y=-150, z=200, ox=1, oy=0, oz=0),     # wp 1 — tipped
            Pose.at(x=0,    y=0,    z=300, theta=45),             # wp 2 — rolled
            Pose.at(x=200,  y=150,  z=200, ox=0, oy=1, oz=0),     # wp 3
            Pose.at(x=400,  y=300,  z=100),                       # wp 4
        ]
        scene.add(TrajectoryPlan(label_prefix="plan", waypoints=plan))

    Internal labels: line segments use ``f"{label_prefix}_path_seg_NN"``;
    waypoint frames use ``f"{label_prefix}_wp_N"`` for the anchor and
    ``f"{label_prefix}_wp_N_axis_x"`` etc. for the axis arrows.

    The pair :class:`viam_visuals.lerp_pose` interpolates between
    two adjacent waypoints — call it from a driver tick to animate
    a "runner" sphere that walks the plan with smoothly-rotating
    orientation.
    """

    label_prefix: str
    waypoints: Sequence[Any] = field(default_factory=list)  # PoseLike each
    parent_frame: Optional[str] = None

    # Path line styling.
    line_color: ColorLike = (100, 180, 220)
    line_width_mm: float = 6.0
    line_opacity: Optional[float] = 0.6

    # Per-waypoint CoordinateFrame styling.
    show_frames: bool = True
    frame_size_mm: float = 80.0
    frame_anchor_radius_mm: float = 6.0
    frame_axis_radius_mm: float = 4.0
    frame_anchor_color: ColorLike = (120, 120, 120)
    frame_anchor_opacity: Optional[float] = 0.5
    frame_axis_opacity: Optional[float] = 1.0
    frame_show_axes_helper: bool = False  # avoid renderer-side noise

    def __post_init__(self) -> None:
        if len(self.waypoints) < 2:
            raise ValueError(
                f"TrajectoryPlan needs at least 2 waypoints; "
                f"got {len(self.waypoints)}"
            )

    def to_visuals(self) -> List[Visual]:
        out: List[Visual] = []
        # Path line connecting the waypoint positions.
        out.extend(Line(
            label_prefix=f"{self.label_prefix}_path",
            points=list(self.waypoints),
            width_mm=self.line_width_mm,
            color=self.line_color,
            opacity=self.line_opacity,
            parent_frame=self.parent_frame,
        ).to_visuals())
        # CoordinateFrame triad at each waypoint.
        if self.show_frames:
            for i, wp in enumerate(self.waypoints):
                out.extend(CoordinateFrame(
                    label=f"{self.label_prefix}_wp_{i}",
                    pose=wp,
                    parent_frame=self.parent_frame,
                    size_mm=self.frame_size_mm,
                    axis_length_mm=self.frame_size_mm,
                    axis_radius_mm=self.frame_axis_radius_mm,
                    anchor_radius_mm=self.frame_anchor_radius_mm,
                    anchor_color=self.frame_anchor_color,
                    anchor_opacity=self.frame_anchor_opacity,
                    axis_opacity=self.frame_axis_opacity,
                    show_axes_helper=self.frame_show_axes_helper,
                ).to_visuals())
        return out
