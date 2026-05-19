"""PCD asset parsing + chunked-delivery helpers.

The Viam 3D viewer renders PCD point clouds in the binary format
``pointcloud.ToPCD`` in the RDK produces. The header is strict: a
leading ``# comment`` line or ``VERSION 0.7`` instead of
``VERSION .7`` causes the viewer to silently drop the geometry (the
RDK's reader is more lax but the viewer is not).

This module parses + slices PCD bytes for chunked delivery; it
doesn't write new PCDs from scratch.
"""

from __future__ import annotations

__all__ = ["parse_pcd_binary", "build_pcd_chunk"]


def parse_pcd_binary(pcd_bytes: bytes) -> tuple[bytes, bytes, int, int]:
    """Split a PCDBinary blob into ``(header_bytes, body_bytes,
    stride, total_points)``. Used by chunked-delivery: callers split
    ``body_bytes`` on stride boundaries to emit individual chunks.

    Expects the binary PCD format ``pointcloud.ToPCD`` produces::

      VERSION .7
      FIELDS x y z rgb
      SIZE 4 4 4 4
      TYPE F F F I
      COUNT 1 1 1 1
      WIDTH <N>
      HEIGHT 1
      VIEWPOINT 0 0 0 1 0 0 0
      POINTS <N>
      DATA binary
      <body: N records of (float x, float y, float z, int32 rgb)>

    Raises ``ValueError`` if the header doesn't match. Stride is
    computed from SIZE/COUNT (i.e., bytes per point — 16 for the
    FFFI layout).
    """
    marker = b"DATA binary\n"
    idx = pcd_bytes.find(marker)
    if idx < 0:
        raise ValueError("PCD: missing 'DATA binary' marker")
    header_end = idx + len(marker)
    header_bytes = pcd_bytes[:header_end]
    body_bytes = pcd_bytes[header_end:]
    # Compute stride from SIZE and COUNT lines.
    header_text = header_bytes.decode("ascii", errors="replace")
    size_line = next(
        (line for line in header_text.splitlines() if line.startswith("SIZE ")),
        None,
    )
    count_line = next(
        (line for line in header_text.splitlines() if line.startswith("COUNT ")),
        None,
    )
    if size_line is None or count_line is None:
        raise ValueError("PCD: missing SIZE or COUNT")
    sizes = [int(s) for s in size_line[len("SIZE "):].split()]
    counts = [int(c) for c in count_line[len("COUNT "):].split()]
    if len(sizes) != len(counts):
        raise ValueError(f"PCD: SIZE/COUNT length mismatch ({sizes} vs {counts})")
    stride = sum(s * c for s, c in zip(sizes, counts, strict=True))
    if stride <= 0:
        raise ValueError(f"PCD: invalid stride {stride}")
    total_points = len(body_bytes) // stride
    return header_bytes, body_bytes, stride, total_points


def build_pcd_chunk(
    header_bytes: bytes,
    body_bytes: bytes,
    stride: int,
    chunk_index: int,
    chunk_size_points: int,
) -> bytes:
    """Build a self-contained PCDBinary blob containing only the chunk
    at ``chunk_index``. Rewrites the WIDTH and POINTS fields in the
    header so the result is a valid standalone PCD the viewer can
    render in isolation.

    Under chunked delivery, the initial Transform's pointcloud bytes
    carry the first chunk (a working PCD all by itself). The viewer
    requests subsequent chunks via the ``get_entity_chunk`` DoCommand
    and stitches them in. Even if the viewer doesn't yet understand
    the chunks metadata, the initial first chunk still renders as a
    smaller-but-correct point cloud.
    """
    total_points = len(body_bytes) // stride
    start = chunk_index * chunk_size_points
    if start >= total_points:
        raise ValueError(
            f"chunk_index {chunk_index} out of range; "
            f"total_points={total_points} chunk_size={chunk_size_points}"
        )
    end = min(start + chunk_size_points, total_points)
    n = end - start
    body_slice = body_bytes[start * stride : end * stride]
    # Rewrite WIDTH and POINTS to match the slice length.
    header_text = header_bytes.decode("ascii", errors="replace")
    new_lines = []
    for line in header_text.split("\n"):
        if line.startswith("WIDTH "):
            new_lines.append(f"WIDTH {n}")
        elif line.startswith("POINTS "):
            new_lines.append(f"POINTS {n}")
        else:
            new_lines.append(line)
    new_header = "\n".join(new_lines).encode("ascii")
    return new_header + body_slice
