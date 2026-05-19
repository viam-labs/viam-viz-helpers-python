"""viam_visuals — typed visual scene constructors for Viam.

A small library for building Viam world-state-store scenes from
typed Python objects instead of hand-built dicts. Each shape (Box,
Sphere, Capsule, …) and animation (Spin, Pulse, Lifecycle, …) is a
dataclass that validates its parameters at construction and
serializes to the wire format the world-state-store service
consumes.

Quickstart::

    import viam_visuals as viz

    box = viz.Box("demo_box", dims_mm=(100, 200, 50),
                  color=(230, 25, 75), opacity=0.8)

    spinning_sphere = viz.Sphere("bobber",
                                 pose=viz.Pose.at(x=300),
                                 radius_mm=80,
                                 animation=viz.Spin(period_s=3))

    # Convert to the wire format the service consumes:
    items = [box.to_dict(), spinning_sphere.to_dict()]

This is the in-repo bootstrap version of the library. The public API
here is stable; the eventual extraction to a standalone repo
``github.com/viam-labs/viam-visuals`` will not change the surface.

See ``LIBRARY_PLAN.md`` in the parent module for design context and
the full delivery roadmap.
"""

from __future__ import annotations

from .animations import (
    Animation,
    AnimationLike,
    Breathe,
    Flicker,
    ForceVector,
    Lifecycle,
    Orbit,
    Oscillate,
    Pulse,
    Spin,
    Static,
    Swing,
    Trajectory,
    normalize_animation,
)
from .color import ColorLike, hsv_to_rgb, normalize_color, snap_step
from .composites import (
    BoundingBox,
    Composite,
    CoordinateFrame,
    Line,
    TrajectoryPlan,
)
from .anim_helpers import (
    orbit_pose,
    oscillate_pose,
    pulse_range,
    spin_pose,
    swing_pose,
    trajectory_pose,
)
from .pose import Pose, PoseLike, lerp_pose, normalize_pose
from .shapes import (
    Arrow,
    Box,
    Capsule,
    Frame,
    Mesh,
    Point,
    PointCloud,
    Sphere,
    Visual,
    to_dicts,
)
from . import registry
from .scene import (
    ADDED,
    GEOM_PATHS,
    METADATA_PATHS,
    PATH_BOX_DIMS_X,
    PATH_BOX_DIMS_Y,
    PATH_BOX_DIMS_Z,
    PATH_CAPSULE_LENGTH,
    PATH_CAPSULE_RADIUS,
    PATH_MESH,
    PATH_METADATA_COLOR,
    PATH_METADATA_INVISIBLE,
    PATH_METADATA_OPACITY,
    PATH_METADATA_SHOW_AXES,
    PATH_OX,
    PATH_OY,
    PATH_OZ,
    PATH_SPHERE_RADIUS,
    PATH_THETA,
    PATH_X,
    PATH_Y,
    PATH_Z,
    POSE_PATHS,
    REMOVED,
    UPDATED,
    Scene,
    SceneEvent,
    events_to_wire,
)
from .basic_geometry import build_basic_geometry
from .service import (
    DEFAULT_PARENT_FRAME,
    DEFAULT_TICK_HZ,
    DEFAULT_UUID_STRATEGY,
    SceneServiceBase,
)
from .uuid_strategy import VALID_STRATEGIES, initial_uuid, versioned_uuid


__all__ = [
    # Pose / Color / type aliases
    "Pose",
    "PoseLike",
    "lerp_pose",
    "orbit_pose",
    "oscillate_pose",
    "pulse_range",
    "spin_pose",
    "swing_pose",
    "trajectory_pose",
    "ColorLike",
    "AnimationLike",
    "hsv_to_rgb",
    "snap_step",
    "normalize_pose",
    "normalize_color",
    "normalize_animation",
    # Shape classes
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
    # Animation classes
    "Animation",
    "Static",
    "Spin",
    "Swing",
    "Oscillate",
    "Orbit",
    "Pulse",
    "Breathe",
    "Flicker",
    "Lifecycle",
    "ForceVector",
    "Trajectory",
    # Composites
    "Composite",
    "CoordinateFrame",
    "Line",
    "BoundingBox",
    "TrajectoryPlan",
    # UUID strategy
    "VALID_STRATEGIES",
    "initial_uuid",
    "versioned_uuid",
    # Service base
    "SceneServiceBase",
    "build_basic_geometry",
    "DEFAULT_TICK_HZ",
    "DEFAULT_UUID_STRATEGY",
    "DEFAULT_PARENT_FRAME",
    # Scene (mutation API)
    "Scene",
    "SceneEvent",
    "ADDED",
    "UPDATED",
    "REMOVED",
    "events_to_wire",
    # Field-mask path constants
    "POSE_PATHS",
    "GEOM_PATHS",
    "METADATA_PATHS",
    "PATH_X", "PATH_Y", "PATH_Z",
    "PATH_OX", "PATH_OY", "PATH_OZ",
    "PATH_THETA",
    "PATH_BOX_DIMS_X", "PATH_BOX_DIMS_Y", "PATH_BOX_DIMS_Z",
    "PATH_SPHERE_RADIUS",
    "PATH_CAPSULE_RADIUS", "PATH_CAPSULE_LENGTH",
    "PATH_MESH",
    "PATH_METADATA_COLOR", "PATH_METADATA_OPACITY",
    "PATH_METADATA_SHOW_AXES", "PATH_METADATA_INVISIBLE",
    # In-process registry
    "registry",
]
