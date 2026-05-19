"""Mesh asset I/O — PLY writers + readers + STL converter.

The viewer renders mesh assets only when ``content_type="ply"``. STL
input is converted to PLY at load time; PLY passes through. This
module owns the binary STL parser, the ASCII PLY writer (with
optional per-vertex colors), and the procedural arrow generator
used by the ``Arrow`` visual.

All vertex coordinates are emitted in METERS (the RDK PLY reader
multiplies by 1000 to convert to the renderer's mm convention).
Caller-side coordinates passed into these helpers stay in
millimeters; the conversion happens at the byte boundary.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import List, Optional, Tuple

from .constants import SUPPORTED_MESH_CONTENT_TYPES


__all__ = [
    "extract_ply_vertex_colors",
    "ply_ascii_bytes",
    "arrow_ply_bytes",
    "stl_to_ply",
    "load_mesh_bytes_as_ply",
    "infer_mesh_content_type",
]


def extract_ply_vertex_colors(
    ply_bytes: bytes,
) -> Optional[List[Tuple[int, int, int]]]:
    """Parse an ASCII PLY and return per-vertex ``(R, G, B)`` tuples
    if the file carries ``property uchar red/green/blue`` alongside
    position. Returns ``None`` if the PLY doesn't have vertex colors
    or can't be parsed.

    The Viam viewer reads vertex colors from
    ``Transform.metadata.colors``, not from PLY-embedded properties,
    so a transcoding step is required to surface them. This function
    is the read half; ``build_metadata(..., vertex_colors=...)`` is
    the write half.

    ASCII PLY only; binary PLY parsing is out of scope.
    """
    try:
        text = ply_bytes.decode("ascii", errors="strict")
    except UnicodeDecodeError:
        return None
    if not text.startswith("ply\n"):
        return None
    if "format ascii" not in text.split("end_header", 1)[0]:
        return None  # binary PLY — out of scope

    lines = text.split("\n")
    vertex_count = 0
    vertex_properties: List[str] = []
    parsing_vertex_element = False
    header_end_line = None
    for i, line in enumerate(lines):
        if line == "end_header":
            header_end_line = i + 1
            break
        if line.startswith("element vertex "):
            vertex_count = int(line.split()[-1])
            parsing_vertex_element = True
        elif line.startswith("element "):
            parsing_vertex_element = False
        elif parsing_vertex_element and line.startswith("property "):
            vertex_properties.append(line.split()[-1])
    if header_end_line is None or vertex_count == 0:
        return None

    try:
        r_idx = vertex_properties.index("red")
        g_idx = vertex_properties.index("green")
        b_idx = vertex_properties.index("blue")
    except ValueError:
        return None

    colors: List[Tuple[int, int, int]] = []
    for i in range(vertex_count):
        line_idx = header_end_line + i
        if line_idx >= len(lines):
            return None
        parts = lines[line_idx].split()
        if len(parts) < len(vertex_properties):
            return None
        try:
            colors.append((
                int(parts[r_idx]),
                int(parts[g_idx]),
                int(parts[b_idx]),
            ))
        except (ValueError, IndexError):
            return None
    return colors if colors else None


def ply_ascii_bytes(
    verts_mm: List[Tuple[float, float, float]],
    faces: List[Tuple[int, ...]],
    vertex_colors: Optional[List[Tuple[int, int, int]]] = None,
) -> bytes:
    """Build an ASCII PLY byte buffer from vertices and faces.

    ``verts_mm`` are in millimeters; the output PLY divides each by
    1000 so the file is in meters (the RDK PLY reader's convention).
    Faces are zero-indexed vertex tuples.

    If ``vertex_colors`` is provided (same length as ``verts_mm``),
    per-vertex ``property uchar red/green/blue`` are emitted in the
    header and each vertex line carries its color. Whether the
    viewer honors PLY-embedded colors is currently unclear — the
    parallel transcode path through ``build_metadata`` is the
    reliable channel.
    """
    has_colors = vertex_colors is not None
    if has_colors and len(vertex_colors) != len(verts_mm):
        raise ValueError(
            f"vertex_colors length {len(vertex_colors)} != vertex count {len(verts_mm)}"
        )
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(verts_mm)}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if has_colors:
        header.extend([
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ])
    header.extend([
        f"element face {len(faces)}",
        "property list uchar int vertex_indices",
        "end_header",
    ])
    lines = list(header)
    if has_colors:
        for (x, y, z), (r, g, b) in zip(verts_mm, vertex_colors):
            lines.append(
                f"{x / 1000.0:.6f} {y / 1000.0:.6f} {z / 1000.0:.6f} "
                f"{int(r) & 0xFF} {int(g) & 0xFF} {int(b) & 0xFF}"
            )
    else:
        for (x, y, z) in verts_mm:
            lines.append(
                f"{x / 1000.0:.6f} {y / 1000.0:.6f} {z / 1000.0:.6f}"
            )
    for face in faces:
        lines.append(f"{len(face)} " + " ".join(str(i) for i in face))
    return ("\n".join(lines) + "\n").encode("ascii")


def arrow_ply_bytes(
    length_mm: float,
    shaft_radius_mm: float,
    tip_radius_mm: Optional[float] = None,
    tip_length_mm: Optional[float] = None,
    sides: int = 12,
) -> bytes:
    """Procedural arrow mesh along local +Z, returned as ASCII PLY bytes.

    ``tip_radius_mm`` defaults to 2× the shaft radius and
    ``tip_length_mm`` defaults to 28% of total length — proportions
    chosen so the arrow head reads clearly without overwhelming the
    shaft.
    """
    if tip_radius_mm is None:
        tip_radius_mm = 2.0 * shaft_radius_mm
    if tip_length_mm is None:
        tip_length_mm = max(0.05 * length_mm, 0.28 * length_mm)
    shaft_length_mm = max(0.0, length_mm - tip_length_mm)

    verts: List[Tuple[float, float, float]] = []
    # v0: shaft bottom center (for the cap fan).
    verts.append((0.0, 0.0, 0.0))
    # v[1..sides]: shaft bottom ring.
    for i in range(sides):
        theta = 2 * math.pi * i / sides
        verts.append((
            shaft_radius_mm * math.cos(theta),
            shaft_radius_mm * math.sin(theta),
            0.0,
        ))
    # Shaft top ring.
    for i in range(sides):
        theta = 2 * math.pi * i / sides
        verts.append((
            shaft_radius_mm * math.cos(theta),
            shaft_radius_mm * math.sin(theta),
            shaft_length_mm,
        ))
    # Cone base ring.
    for i in range(sides):
        theta = 2 * math.pi * i / sides
        verts.append((
            tip_radius_mm * math.cos(theta),
            tip_radius_mm * math.sin(theta),
            shaft_length_mm,
        ))
    # Apex.
    apex_idx = 1 + 3 * sides
    verts.append((0.0, 0.0, shaft_length_mm + tip_length_mm))

    bot_ring_start = 1
    top_ring_start = 1 + sides
    cone_ring_start = 1 + 2 * sides

    faces: List[Tuple[int, ...]] = []
    # Shaft bottom cap fan around v0.
    for i in range(sides):
        v_curr = bot_ring_start + i
        v_next = bot_ring_start + (i + 1) % sides
        faces.append((0, v_next, v_curr))
    # Shaft side quads → triangles.
    for i in range(sides):
        b = bot_ring_start + i
        bn = bot_ring_start + (i + 1) % sides
        t = top_ring_start + i
        tn = top_ring_start + (i + 1) % sides
        faces.append((b, bn, t))
        faces.append((bn, tn, t))
    # Washer between shaft top (narrow) and cone base (wide).
    for i in range(sides):
        inner = top_ring_start + i
        inner_next = top_ring_start + (i + 1) % sides
        outer = cone_ring_start + i
        outer_next = cone_ring_start + (i + 1) % sides
        faces.append((inner, outer, inner_next))
        faces.append((inner_next, outer, outer_next))
    # Cone side triangles.
    for i in range(sides):
        b = cone_ring_start + i
        bn = cone_ring_start + (i + 1) % sides
        faces.append((b, bn, apex_idx))

    return ply_ascii_bytes(verts, faces)


def stl_to_ply(stl_bytes: bytes) -> bytes:
    """Convert binary STL bytes to ASCII PLY bytes.

    The viewer only renders PLY (the RDK's mesh.go comment is
    explicit: "The visualizer expects all meshes to be in PLY
    format"). STL input is converted at load time.

    Output is ASCII PLY with per-triangle vertices (no dedup) — fine
    for small assets; use ``trimesh`` offline if you need a smaller
    PLY from a large STL.
    """
    if len(stl_bytes) < 84:
        raise ValueError("STL data too small (need >=84 bytes for header)")
    n_tris = struct.unpack("<I", stl_bytes[80:84])[0]
    expected_size = 84 + n_tris * 50
    if len(stl_bytes) < expected_size:
        raise ValueError(
            f"STL truncated: expected {expected_size} bytes for "
            f"{n_tris} triangles, got {len(stl_bytes)}"
        )
    verts: List[Tuple[float, float, float]] = []
    faces: List[Tuple[int, ...]] = []
    offset = 84
    for _ in range(n_tris):
        offset += 12  # skip per-tri normal
        face_idx = []
        for _v in range(3):
            x, y, z = struct.unpack("<fff", stl_bytes[offset:offset + 12])
            offset += 12
            face_idx.append(len(verts))
            verts.append((x, y, z))
        offset += 2  # skip attribute byte count
        faces.append(tuple(face_idx))
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(verts)}",
        "property float x",
        "property float y",
        "property float z",
        f"element face {len(faces)}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    for (x, y, z) in verts:
        lines.append(f"{x:.6f} {y:.6f} {z:.6f}")
    for (a, b, c) in faces:
        lines.append(f"3 {a} {b} {c}")
    return ("\n".join(lines) + "\n").encode("ascii")


def load_mesh_bytes_as_ply(asset_bytes: bytes, source_path: str) -> bytes:
    """Return PLY bytes regardless of input format. Dispatches on the
    source path's extension."""
    fmt = infer_mesh_content_type(source_path)
    if fmt == "stl":
        return stl_to_ply(asset_bytes)
    return asset_bytes  # already PLY


def infer_mesh_content_type(asset_path: str) -> str:
    """Map a file extension to the lowercase content_type the renderer
    expects. Raises ``ValueError`` for unsupported extensions."""
    ext = Path(asset_path).suffix.lstrip(".").lower()
    if ext not in SUPPORTED_MESH_CONTENT_TYPES:
        raise ValueError(
            f"mesh content type {ext!r} is not supported; "
            f"only {SUPPORTED_MESH_CONTENT_TYPES} are accepted by the viewer"
        )
    return ext
