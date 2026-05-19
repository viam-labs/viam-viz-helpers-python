"""Private implementation details. Not part of the public API.

These modules house the wire-format pieces — PLY / STL / PCD parsers
and writers, the metadata Struct builder, shared constants — that
the public ``viam_visuals`` surface uses to produce wire-format
output. Nothing in this subpackage should be imported by external
callers; the surface is implementation-only and may change between
releases.

Module map:

  * :mod:`constants` — shared wire-format constants
    (``RENDERER_MESH_CONTENT_TYPE``, ``POINT_MARKER_RADIUS_MM``,
    ``SUPPORTED_MESH_CONTENT_TYPES``).
  * :mod:`metadata` — builds the ``Transform.metadata`` Struct in
    the viewer's required schema.
  * :mod:`mesh` — PLY / STL parsers and writers, the procedural
    arrow generator.
  * :mod:`pcd` — PCD parser + chunked-delivery helpers.
"""
