"""Visual shapes — Box, Sphere, Capsule, Point, Arrow, Mesh, PointCloud.

Each class is a typed scene-item constructor: its fields cover only
the parameters that apply to its shape, and bad inputs error at
construction rather than at wire-encode time. :meth:`Visual.to_dict`
serializes the instance into the dict format the Viam
world-state-store service consumes.

The library is currently scoped to the five native ``commonpb.Geometry``
primitives (Box, Sphere, Capsule, Mesh, PointCloud) plus two sugar
types (Point, Arrow) shared by every author. Composite shapes
(coordinate frames, arrows from point-to-point, text plaques) are
planned for a future ``viam_visuals.composites`` submodule but not
in this release.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Tuple

from .animations import AnimationLike, normalize_animation
from .color import ColorLike, normalize_color
from .pose import Pose, PoseLike, normalize_pose


__all__ = [
    "Visual",
    "Box",
    "Sphere",
    "Capsule",
    "Point",
    "Frame",
    "Arrow",
    "Mesh",
    "PointCloud",
    "to_dicts",
]


@dataclass
class Visual:
    """Base class for scene items.

    Subclasses set ``_TYPE`` (the wire-format type string) and override
    :meth:`_shape_fields` to contribute their geometry-specific keys.
    The common fields here cover identity, placement, appearance, and
    animation — shared by every concrete shape.
    """

    label: str
    pose: PoseLike = None
    parent_frame: Optional[str] = None
    color: ColorLike = None
    opacity: Optional[float] = None
    show_axes_helper: bool = False
    invisible: bool = False
    # AnimationLike (Union of None / Animation / Mapping). Typed as
    # Any here so the dataclass default-value machinery doesn't fight
    # us; normalize_animation() coerces at to_dict() time. The new
    # SceneServiceBase.tick(scene, t) hook doesn't use this field at
    # all — recommended pattern is to leave it None and mutate the
    # Visual directly each tick.
    animation: Any = None

    _TYPE: str = field(default="", repr=False, init=False)

    def _shape_fields(self) -> Mapping[str, Any]:
        """Geometry-specific fields contributed by the subclass.

        Subclasses override and return e.g. ``{"dims_mm": {...}}`` for
        Box or ``{"radius_mm": 90.0}`` for Sphere.
        """
        return {}

    def to_dict(self) -> MutableMapping[str, Any]:
        """Serialize to the wire-format dict.

        The output is the same shape the Viam world-state-store service
        consumes (and the same shape DoCommand ``snapshot`` returns).
        """
        if not self._TYPE:
            raise ValueError(f"{type(self).__name__} forgot to set _TYPE")
        if not self.label:
            raise ValueError(f"{type(self).__name__} requires a non-empty label")

        out: MutableMapping[str, Any] = {
            "type": self._TYPE,
            "label": self.label,
            "pose": normalize_pose(self.pose),
        }
        out.update(self._shape_fields())

        color = normalize_color(self.color)
        if color is not None:
            out["color"] = color
        if self.opacity is not None:
            if not 0.0 <= float(self.opacity) <= 1.0:
                raise ValueError(
                    f"opacity must be in [0, 1]; got {self.opacity!r}"
                )
            out["opacity"] = float(self.opacity)
        if self.parent_frame:
            out["parent_frame"] = self.parent_frame
        if self.show_axes_helper:
            out["show_axes_helper"] = True
        if self.invisible:
            out["invisible"] = True

        anim = normalize_animation(self.animation)
        out["animation"] = anim if anim is not None else {"mode": "none"}
        return out


@dataclass
class Box(Visual):
    """Solid axis-aligned box. ``dims_mm`` is ``(x, y, z)`` in mm."""

    dims_mm: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    _TYPE: str = field(default="box", repr=False, init=False)

    def __post_init__(self) -> None:
        if len(self.dims_mm) != 3:
            raise ValueError(f"Box.dims_mm needs 3 components; got {self.dims_mm!r}")
        if any(d <= 0 for d in self.dims_mm):
            raise ValueError(f"Box.dims_mm must all be > 0; got {self.dims_mm!r}")

    def _shape_fields(self) -> Mapping[str, Any]:
        return {"dims_mm": {"x": float(self.dims_mm[0]),
                            "y": float(self.dims_mm[1]),
                            "z": float(self.dims_mm[2])}}


@dataclass
class Sphere(Visual):
    """Solid sphere of the given radius in mm."""

    radius_mm: float = 0.0
    _TYPE: str = field(default="sphere", repr=False, init=False)

    def __post_init__(self) -> None:
        if float(self.radius_mm) <= 0:
            raise ValueError(f"Sphere.radius_mm must be > 0; got {self.radius_mm!r}")

    def _shape_fields(self) -> Mapping[str, Any]:
        return {"radius_mm": float(self.radius_mm)}


@dataclass
class Capsule(Visual):
    """Solid capsule — cylinder with hemispherical end caps.

    ``radius_mm`` is the cylinder radius; ``length_mm`` is the total
    length (capsule extends from -length/2 to +length/2 along its
    local Z).
    """

    radius_mm: float = 0.0
    length_mm: float = 0.0
    _TYPE: str = field(default="capsule", repr=False, init=False)

    def __post_init__(self) -> None:
        if float(self.radius_mm) <= 0:
            raise ValueError(f"Capsule.radius_mm must be > 0; got {self.radius_mm!r}")
        if float(self.length_mm) <= 0:
            raise ValueError(f"Capsule.length_mm must be > 0; got {self.length_mm!r}")

    def _shape_fields(self) -> Mapping[str, Any]:
        return {"radius_mm": float(self.radius_mm),
                "length_mm": float(self.length_mm)}


@dataclass
class Point(Visual):
    """Marker point — a sphere with a small but visible radius.

    The proto Geometry oneof has no Point variant, and the viewer
    skips zero-radius geometries. The service implementation
    (:func:`viam_visuals.build_basic_geometry`) substitutes a sphere
    with a fixed 8 mm radius so the marker has a visible footprint.

    For a sphere with custom radius, use :class:`Sphere` directly.
    """

    _TYPE: str = field(default="point", repr=False, init=False)


@dataclass
class Frame(Visual):
    """A pure transform anchor — a reference frame other Visuals can
    parent to without rendering anything visible itself.

    Use it to declare hierarchy: place a Frame at the position you
    want to be the "joint" or "pivot", then give other Visuals
    ``parent_frame=<frame.label>``. Moving the Frame transports the
    children with it; the renderer composes the parent transform
    automatically.

    Internally a tiny sphere with ``invisible=True``. Set
    ``show_axes_helper=True`` (the default for Frame) to debug
    orientation; set ``invisible=False`` to render the anchor
    sphere itself.

    Example::

        pivot = viz.Frame("pivot", pose=viz.Pose.at(x=500, z=300))
        child = viz.Box(
            "child", pose=viz.Pose.at(x=80),  # 80mm right of pivot
            parent_frame="pivot",
        )
        # Rotate the pivot; the child rotates with it:
        pivot.pose = viz.Pose.at(x=500, z=300, theta=45)
        return scene.update(pivot)  # child follows automatically
    """

    invisible: bool = True
    show_axes_helper: bool = True

    _TYPE: str = field(default="sphere", repr=False, init=False)

    def _shape_fields(self) -> Mapping[str, Any]:
        # Tiny sphere — invisible by default so the anchor doesn't
        # show, but the axes helper still renders if enabled.
        return {"radius_mm": 1.0}


@dataclass
class Arrow(Visual):
    """Procedural arrow mesh — cylindrical shaft + conical tip along
    the entity's local +Z. ``length_mm`` is the total tip-to-tail
    length; ``radius_mm`` is the shaft radius.

    See :meth:`Arrow.from_to` for the common "draw an arrow from
    pose A to pose B" pattern.
    """

    length_mm: float = 0.0
    radius_mm: float = 0.0
    _TYPE: str = field(default="arrow", repr=False, init=False)

    def __post_init__(self) -> None:
        if float(self.length_mm) <= 0:
            raise ValueError(f"Arrow.length_mm must be > 0; got {self.length_mm!r}")
        if float(self.radius_mm) <= 0:
            raise ValueError(f"Arrow.radius_mm must be > 0; got {self.radius_mm!r}")

    def _shape_fields(self) -> Mapping[str, Any]:
        return {"length_mm": float(self.length_mm),
                "radius_mm": float(self.radius_mm)}

    @classmethod
    def from_to(
        cls,
        label: str,
        start: "Pose",
        end: "Pose",
        radius_mm: float,
        color: Any = None,
        opacity: Optional[float] = None,
        parent_frame: Optional[str] = None,
        animation: Any = None,
    ) -> "Arrow":
        """Build an Arrow pointing from ``start`` to ``end``.

        The arrow's pose origin sits at ``start``; its orientation
        vector is the unit vector ``(end - start) / length``; its
        ``length_mm`` equals ``|end - start|``. Useful for "draw a
        force vector at this hand pose" / "show motion plan from A
        to B" without computing the orientation yourself — the
        direction vector is normalized for you and the resulting
        ``ox/oy/oz`` are guaranteed to be a unit vector regardless
        of input magnitudes.

        Raises ``ValueError`` if ``|end - start| < 1e-6`` (zero-length
        arrow would have an undefined direction).
        """
        import math as _math
        dx = end.x - start.x
        dy = end.y - start.y
        dz = end.z - start.z
        length_mm = _math.sqrt(dx * dx + dy * dy + dz * dz)
        if length_mm < 1e-6:
            raise ValueError(
                f"Arrow.from_to needs distinct points; |end-start|={length_mm}"
            )
        return cls(
            label=label,
            length_mm=length_mm,
            radius_mm=radius_mm,
            pose=Pose.at(
                x=start.x, y=start.y, z=start.z,
                ox=dx / length_mm, oy=dy / length_mm, oz=dz / length_mm,
            ),
            color=color,
            opacity=opacity,
            parent_frame=parent_frame,
            animation=animation,
        )


@dataclass
class Mesh(Visual):
    """Mesh loaded from a PLY (or STL — auto-converted to PLY at load
    time unless ``raw_stl=True``).

    ``mesh_path`` is resolved by the service implementation; the
    library doesn't open files.

    .. warning::
        ``raw_stl=True`` is an experimental knob that deliberately
        bypasses the library's STL→PLY conversion. It exists to
        reproduce a renderer-side bug (the viewer silently drops
        ``content_type="stl"`` despite the proto accepting it).
        Production callers should leave it ``False``; future
        versions may remove the field entirely.
    """

    mesh_path: str = ""
    raw_stl: bool = False
    _TYPE: str = field(default="mesh", repr=False, init=False)

    def __post_init__(self) -> None:
        if not self.mesh_path:
            raise ValueError("Mesh requires mesh_path")

    def _shape_fields(self) -> Mapping[str, Any]:
        out: MutableMapping[str, Any] = {"mesh_path": self.mesh_path}
        if self.raw_stl:
            out["raw_stl"] = True
        return out


@dataclass
class PointCloud(Visual):
    """Point cloud loaded from a PCD asset.

    PCD bytes must match RDK's ``pointcloud.ToPCD`` format byte-for-byte
    (binary, ``VERSION .7`` literal, no leading ``#`` comments). The
    viewer's parser is strict-order; ASCII / binary_compressed / extra
    comments silently fail.

    .. warning::
        Setting ``chunked=True`` with a positive ``chunk_size`` opts
        into experimental chunked delivery: the service ships chunk 0
        inline in the initial transform with a ``metadata.chunks``
        sub-struct, and exposes the rest via the ``get_entity_chunk``
        DoCommand. **The viewer's behavior on this contract is
        unverified** — whether it actually fetches subsequent chunks
        is open. Leave ``chunked=False`` unless you're specifically
        probing the chunked-delivery path.
    """

    pointcloud_path: str = ""
    chunked: bool = False
    chunk_size: Optional[int] = None
    _TYPE: str = field(default="pointcloud", repr=False, init=False)

    def __post_init__(self) -> None:
        if not self.pointcloud_path:
            raise ValueError("PointCloud requires pointcloud_path")
        if self.chunk_size is not None and int(self.chunk_size) <= 0:
            raise ValueError(
                f"PointCloud.chunk_size must be a positive integer; "
                f"got {self.chunk_size!r}"
            )

    def _shape_fields(self) -> Mapping[str, Any]:
        out: MutableMapping[str, Any] = {"pointcloud_path": self.pointcloud_path}
        if self.chunked:
            out["chunked"] = True
        if self.chunk_size is not None:
            out["chunk_size"] = int(self.chunk_size)
        return out


def to_dicts(*visuals: Visual) -> Sequence[Mapping[str, Any]]:
    """Materialize a sequence of :class:`Visual` instances into the
    wire-format dicts the service consumes. Convenience for callers
    that build visuals positionally."""
    return [v.to_dict() for v in visuals]
