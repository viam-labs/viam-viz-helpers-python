"""Transform.metadata struct builder.

The Viam 3D viewer reads metadata in the schema defined by
``viamrobotics/visualization`` (``draw/transform.go::MetadataToStruct``,
``protos/draw/v1/metadata.proto``). All five keys — ``colors``,
``color_format``, ``opacities``, ``show_axes_helper``, ``invisible``
— must be present for the viewer to accept the entity; omitting any
of them caused entities to render as invisible in early versions of
this prototype (see ``LESSONS.md``).

The RDK's worldstatestore fake at ``services/worldstatestore/fake/``
emits an OBSOLETE ``{color: {r, g, b}, opacity: 0.5}`` shape that
the viewer silently ignores. Don't follow that pattern.
"""

from __future__ import annotations

import base64
from typing import Any, List, Mapping, Optional, Tuple

from google.protobuf.struct_pb2 import Struct
from viam.utils import dict_to_struct


__all__ = ["build_metadata", "clamp_u8"]


def clamp_u8(v: Any) -> int:
    """Clamp any number into the 0..255 range as a uint8."""
    iv = int(v)
    if iv < 0:
        return 0
    if iv > 255:
        return 255
    return iv


def build_metadata(
    color: Optional[Mapping[str, Any]] = None,
    opacity: Optional[float] = None,
    show_axes_helper: bool = False,
    invisible: bool = False,
    vertex_colors: Optional[List[Tuple[int, int, int]]] = None,
    chunks: Optional[Mapping[str, Any]] = None,
) -> Struct:
    """Build a ``Transform.metadata`` Struct in the viewer's schema.

    Keys the viewer consumes:

      * ``colors`` (string): base64 of packed RGB bytes. 3 bytes per
        color; one color for single-component primitives, N for
        multi-component (point clouds, polylines). Per
        ``draw/buffer_packer.go::packColors``.
      * ``color_format`` (number): ``1`` for ``COLOR_FORMAT_RGB`` —
        the only format defined today.
      * ``opacities`` (string): base64 of packed alpha bytes. One
        byte per color, or one uniform byte.
      * ``show_axes_helper`` (bool): renders an RGB XYZ triad at the
        entity's origin.
      * ``invisible`` (bool): hides the entity by default.

    Always emits all five keys. With empty ``colors`` and opacity
    255, the renderer falls back to embedded per-point RGB on point
    clouds and a viewer-default fill on solids.
    """
    fields: dict = {}
    if vertex_colors:
        # Per-vertex colors take precedence over the uniform `color`.
        # Pack N RGB triples; library's MetadataToStruct expects this
        # exact format.
        packed = bytearray()
        for c in vertex_colors:
            packed.append(clamp_u8(c[0]))
            packed.append(clamp_u8(c[1]))
            packed.append(clamp_u8(c[2]))
        fields["colors"] = base64.b64encode(bytes(packed)).decode("ascii")
    elif color is not None:
        rgb_bytes = bytes([
            clamp_u8(color.get("r", 0)),
            clamp_u8(color.get("g", 0)),
            clamp_u8(color.get("b", 0)),
        ])
        fields["colors"] = base64.b64encode(rgb_bytes).decode("ascii")
    else:
        # Empty colors → viewer falls back to default fill for solids
        # (or embedded per-point RGB for PCDs).
        fields["colors"] = ""
    fields["color_format"] = 1.0  # COLOR_FORMAT_RGB
    alpha = 255 if opacity is None else clamp_u8(round(float(opacity) * 255))
    fields["opacities"] = base64.b64encode(bytes([alpha])).decode("ascii")
    fields["show_axes_helper"] = bool(show_axes_helper)
    fields["invisible"] = bool(invisible)
    if chunks:
        # `chunks` declares chunked delivery for large entities.
        # Schema from the visualization library's e2e fixture and is
        # EXPERIMENTAL — the inner shape is best-effort. If the
        # viewer ignores these, the initial Transform still carries
        # a valid first-chunk PCD that renders standalone.
        fields["chunks"] = dict(chunks)
    return dict_to_struct(fields)
