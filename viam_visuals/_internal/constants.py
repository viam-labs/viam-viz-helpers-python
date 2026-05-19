"""Wire-format constants shared across the library implementation."""

# The viewer renders mesh assets only when content_type is "ply".
# The RDK's STL parser is real, but on the wire to the viewer
# everything is converted to PLY (see rdk/spatialmath/mesh.go).
RENDERER_MESH_CONTENT_TYPE = "ply"

# A radius=0 sphere is a Point in the RDK's internal model but the
# viewer skips zero-radius geometries entirely. The library renders
# `Point` visuals as a small fixed-radius sphere so they show up.
POINT_MARKER_RADIUS_MM = 8.0

# Mesh file extensions the library knows how to load. PLY passes
# through unmodified; STL is converted to PLY at load time.
SUPPORTED_MESH_CONTENT_TYPES = ("ply", "stl")
