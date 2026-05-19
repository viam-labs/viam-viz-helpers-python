# viam_visuals — typed visual scene constructors for Viam

A Python library for building Viam world-state-store scenes from
typed dataclasses instead of hand-built wire-format dicts.

This directory is the bootstrap in-tree version. When extracted, the
target repo is **`github.com/viam-labs/viam-viz-helpers`** with the
`pip install`-able name `viam-viz-helpers` and the import name
`viam_visuals` (unchanged from here).

## Quickstart

```python
import viam_visuals as viz

box = viz.Box("demo_box", dims_mm=(100, 200, 50),
              color=(230, 25, 75), opacity=0.8)

spinning = viz.Sphere("bobber",
                      pose=viz.Pose.at(x=300),
                      radius_mm=80,
                      animation=viz.Spin(period_s=3))

# Scene with object-based mutation API.
scene = viz.Scene(parent_frame="world")
scene.add(box, spinning)
spinning.radius_mm = 120
events = scene.update(spinning)   # diffs against committed state,
                                  # produces camelCase field-mask paths
```

Or inherit `SceneServiceBase` and get a ready-made WSS service with
the standard DoCommand verbs (list/add/remove/update/clear/preset/
snapshot/set_uuid_strategy/apply_events), animation tick loop, and
subscriber fanout for free.

## Public API surface

### Pose, Color
- `Pose`, `Pose.at(...)`, `Pose.identity()`, `PoseLike`, `lerp_pose`
- `ColorLike`, `normalize_color`, `normalize_pose`

`lerp_pose` does true quaternion SLERP on orientation (linear on
position). Conversions match RDK's `spatialmath.QuatToOV` exactly so
the renderer reconstructs continuous rotations across the OV
singularity at `|oz| = 1`.

### Shapes
- `Visual` (base) — `Box`, `Sphere`, `Capsule`, `Point`, `Arrow`,
  `Mesh`, `PointCloud`
- `to_dicts(visuals)` — convert to the wire-format list

### Animations
- `Animation` (base) — `Static`, `Spin`, `Swing`, `Oscillate`,
  `Orbit`, `Pulse`, `Breathe`, `Flicker`, `Lifecycle`, `ForceVector`,
  `Trajectory`
- `AnimationLike`, `normalize_animation`

### Composites
- `Composite` (base) — `CoordinateFrame`, `Line`, `BoundingBox`,
  `TrajectoryPlan`

`TrajectoryPlan` is the motion-plan-shaped composite: pass a list of
poses (matches CBiRRT / motion-service output after FK), get a
polyline plus per-waypoint coordinate-frame triads.

### Scene
- `Scene` — `add`, `update`, `add_or_update`, `remove`, `clear`,
  `get`, `labels`
- `SceneEvent` (`kind` ∈ `ADDED`/`UPDATED`/`REMOVED`)
- `events_to_wire(events)` — for the `apply_events` DoCommand

### Service base
- `SceneServiceBase` — inheritable WSS implementation
- `DEFAULT_TICK_HZ`, `DEFAULT_UUID_STRATEGY`, `DEFAULT_PARENT_FRAME`

### UUID strategy
- `VALID_STRATEGIES`, `initial_uuid`, `versioned_uuid`

### In-process registry
- `registry` (module) — `register`, `lookup`, `unregister`
- The driver→visualizer pattern uses `registry.lookup` to skip
  gRPC when both resources ship from the same module binary.

## Reference module

[`viam-labs/example-visualizations-python`](https://github.com/viam-labs/example-visualizations-python)
is the canonical first adopter and renderer-behavior probe.
Three models in one binary demonstrate the library's architecture:
`standalone-playground`, `playground-visualizer`,
`playground-driver`.

## Conventions and gotchas

Read [`../LESSONS.md`](../LESSONS.md) — every finding has file:line
evidence from the renderer's actual behavior. Highlights:

- Field-mask paths MUST be camelCase, not snake_case.
- PCD headers must match `pointcloud.ToPCD` byte-for-byte
  (`VERSION .7`, not `VERSION 0.7`; no leading `#` comments).
- Mesh/PCD file coordinates are in METERS, not millimeters.
- The viewer renders only PLY meshes (STL → PLY at load time).
- Transform.metadata uses the `viamrobotics/visualization` schema
  (`colors`, `opacities`, `color_format`, `show_axes_helper`,
  `invisible`).
- The renderer caches REMOVED UUIDs — re-add events for the same
  label silently drop. Rotate UUIDs on lifecycle/flicker re-add.

## License

Apache 2.0 — see [LICENSE](../LICENSE).
