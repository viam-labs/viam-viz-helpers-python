"""``build_basic_geometry`` — easy-mode geometry builder for services
that publish only the standard non-asset primitive types.

The ``viam_visuals.SceneServiceBase`` subclass has to implement a
``build_geometry`` hook that returns a ``commonpb.Geometry`` for each
item. For services that use only box / sphere / capsule / point /
arrow, this dispatcher is the one-line implementation:

.. code-block:: python

    class MyService(SceneServiceBase, EasyResource):
        ...
        def build_geometry(self, item, override):
            return viam_visuals.build_basic_geometry(item, override)

Mesh and pointcloud are excluded because they require I/O (a
``read_asset`` call) and a content-type / PLY-conversion decision
the helper can't make in isolation. Services that publish those need
their own dispatcher (see ``src/service.py`` in the example module
for the full version).
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from viam.proto.common import (
    Capsule,
    Geometry,
    Mesh,
    RectangularPrism,
    Sphere,
    Vector3,
)

from ._internal.constants import POINT_MARKER_RADIUS_MM
from ._internal.mesh import arrow_ply_bytes


RENDERER_MESH_CONTENT_TYPE = "ply"


__all__ = ["build_basic_geometry"]


def build_basic_geometry(
    item: Mapping[str, Any],
    override: Optional[Mapping[str, Any]] = None,
) -> Geometry:
    """Build a ``commonpb.Geometry`` from a wire-format item dict.

    Handles ``box``, ``sphere``, ``capsule``, ``point``, and ``arrow``
    primitive types. Raises ``ValueError`` for any other type.

    ``override`` is the optional per-tick geometry-overrides dict
    (only relevant to services with animation). If present, its
    ``dims_mm`` / ``radius_mm`` / ``length_mm`` keys take precedence
    over the item's base values. Pass ``None`` for static scenes.
    """
    override = override or {}
    t = item.get("type")
    label = item.get("label", "")
    if t == "box":
        d = override.get("dims_mm") or item.get("dims_mm") or {"x": 100, "y": 100, "z": 100}
        return Geometry(
            label=label,
            box=RectangularPrism(dims_mm=Vector3(
                x=float(d["x"]), y=float(d["y"]), z=float(d["z"]),
            )),
        )
    if t == "sphere":
        r = float(override.get("radius_mm") or item.get("radius_mm") or 50.0)
        return Geometry(label=label, sphere=Sphere(radius_mm=r))
    if t == "capsule":
        r = float(override.get("radius_mm") or item.get("radius_mm") or 50.0)
        L = float(override.get("length_mm") or item.get("length_mm") or 200.0)
        return Geometry(label=label, capsule=Capsule(radius_mm=r, length_mm=L))
    if t == "point":
        return Geometry(label=label, sphere=Sphere(radius_mm=POINT_MARKER_RADIUS_MM))
    if t == "arrow":
        L = float(override.get("length_mm") or item.get("length_mm") or 200.0)
        r = float(override.get("radius_mm") or item.get("radius_mm") or 10.0)
        ply = arrow_ply_bytes(length_mm=L, shaft_radius_mm=r)
        return Geometry(
            label=label,
            mesh=Mesh(content_type=RENDERER_MESH_CONTENT_TYPE, mesh=ply),
        )
    raise ValueError(
        f"build_basic_geometry doesn't handle item type {t!r} "
        f"(use a custom build_geometry hook for mesh / pointcloud)"
    )
