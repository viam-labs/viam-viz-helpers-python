# viam-viz-helpers

Typed scene constructors for Viam world-state-store services and the 3D scene viewer.

The Viam 3D viewer subscribes to `rdk:service:world_state_store` services for transform updates. The wire format is a closed set of geometry primitives plus a custom metadata schema, with quirks around field-mask paths, UUID rotation, and PCD encoding that aren't obvious from the proto definitions. This library lets module authors construct scenes from typed dataclasses (`Box`, `Sphere`, `Pose`, `Color`, `Animation`, …) and inherit a ready-made WSS service base, while the library handles the wire format and renderer-side quirks underneath.

- **GitHub:** [viam-labs/viam-viz-helpers-python](https://github.com/viam-labs/viam-viz-helpers-python)
- **PyPI name:** `viam-viz-helpers` (not yet published — install from git, see below)
- **Import name:** `viam_visuals`
- **Go sibling:** [viam-labs/viam-viz-helpers-go](https://github.com/viam-labs/viam-viz-helpers-go)
- **License:** Apache-2.0

## Install

Not yet on PyPI. Install directly from this repo:

```
pip install git+https://github.com/viam-labs/viam-viz-helpers-python.git@main
```

In a Viam module's `requirements.txt`:

```
viam-viz-helpers @ git+https://github.com/viam-labs/viam-viz-helpers-python.git@main
```

For local iteration alongside a consuming module, clone this repo as a sibling and editable-install it into the module's venv:

```
pip install -e ../viam-viz-helpers-python
```

## Quickstart

A minimal world-state-store service that publishes three primitives and animates one:

```python
import math
import viam_visuals as viz
from viam_visuals import SceneServiceBase

class MyScene(SceneServiceBase):
    async def reconfigure(self, config, deps):
        self.bobber = viz.Sphere(
            "bobber",
            pose=viz.Pose.at(x=300, z=100),
            radius_mm=80,
            color=(60, 180, 75),
        )
        self.set_scene(
            viz.Box(
                "demo_box",
                pose=viz.Pose.at(x=-200, z=100),
                dims_mm=(150, 150, 150),
                color=(230, 25, 75),
            ),
            self.bobber,
            viz.Capsule(
                "demo_capsule",
                pose=viz.Pose.at(x=100, z=100),
                radius_mm=50, length_mm=200,
                color=(0, 130, 200),
            ),
        )

    def scene_tick(self, scene, t):
        # Mutate typed Visuals in place; return Scene.update events.
        self.bobber.pose = viz.spin_pose(
            viz.Pose.at(x=300, z=100), period_s=3.0, t=t,
        )
        return scene.update(self.bobber)
```

That's the full surface for a static-plus-one-animated scene. `SceneServiceBase` provides everything else: `ListUUIDs` / `GetTransform` / `StreamTransformChanges`, subscriber fanout, the tick goroutine, and a standard DoCommand verb set (`list` / `clear` / `snapshot` / `apply_events` / …).

For a fuller example exercising every primitive type, every animation mode, presets, and the driver→visualizer split, see [`viam-labs/example-visualizations-python`](https://github.com/viam-labs/example-visualizations-python).

## Architecture

### Typed object graph

Scenes are built from typed Visual subclasses (`Box`, `Sphere`, `Capsule`, `Point`, `Arrow`, `Frame`, `Mesh`, `PointCloud`). Each carries the common fields (`label`, `pose`, `parent_frame`, `color`, `opacity`, `show_axes_helper`, `invisible`, `animation`) plus shape-specific fields (`dims_mm`, `radius_mm`, `length_mm`, `mesh_path`, …).

Composites (`CoordinateFrame`, `Line`, `BoundingBox`, `TrajectoryPlan`) expand into the underlying primitives at `Scene.add` time. `TrajectoryPlan` is the motion-plan-shaped composite — pass a list of poses (matches CBiRRT / motion-service output after forward kinematics), get back a polyline plus per-waypoint coordinate-frame triads.

### Animation specs

Eleven typed `Animation` dataclasses cover the common cases: `Static`, `Spin`, `Swing`, `Oscillate`, `Orbit`, `Pulse`, `Breathe`, `Flicker`, `Lifecycle`, `ForceVector`, `Trajectory`. Attach one to a Visual's `animation=` field; the library's default `scene_tick` dispatches `Animation.apply(visual, base, t)` every tick. `base` is a frozen snapshot of the Visual at install time so animation math composes onto a stable rest state.

### Hook surface

`SceneServiceBase` is the inheritable WSS service. Subclasses override what they need:

- `build_geometry(item, base_geom)` — only if you publish primitive types beyond the standard set (`box` / `sphere` / `capsule` / `point` / `arrow`). The library's default handles those five.
- `read_asset(path)` — only if you load mesh / pointcloud assets.
- `scene_tick(scene, t)` — per-tick animation hook. Default dispatches `Animation.apply` per Visual.
- `load_preset(name)` — only if you expose named scene presets via the `preset` DoCommand verb.
- `handle_custom_command(ctx, cmd)` — only if you add DoCommand verbs beyond the standard set.

Modules that publish only static or `Animation`-driven scenes of standard primitives can subclass `SceneServiceBase`, override `reconfigure` to call `set_scene(...)`, and be done — no other hooks required.

### Scene mutation

```python
scene = viz.Scene(parent_frame="world")
scene.add(box, sphere)            # → [SceneEvent(kind="added", ...)]
sphere.radius_mm = 120
scene.update(sphere)              # → [SceneEvent(kind="updated", paths=[...])]
scene.remove("sphere_label")      # → [SceneEvent(kind="removed", ...)]
```

`Scene.update` diffs the visual's current `to_dict()` against the committed snapshot and produces UPDATED events with camelCase field-mask paths. When the diff touches metadata (color, opacity, parent_frame, show_axes_helper, invisible), `Scene.update` emits an UPDATED with `paths=[]` — the **respawn signal**. The renderer drops `metadata.*` on UPDATED, so the library translates the empty-paths event into REMOVE + re-ADD with a fresh UUID downstream. See [`Scene.update`](viam_visuals/scene.py) and [`SceneServiceBase._apply_events`](viam_visuals/service.py) for the exact rule.

### In-process registry

When two resources live in the same module binary, a downstream resource can hold a direct Python reference to an upstream resource via `viam_visuals.registry`, skipping the gRPC stub the framework's `Dependencies` injects. See `viam_visuals/registry.py` for the API and the driver-visualizer pair in `example-visualizations-python` for the canonical pattern.

## Renderer quirks (load-bearing)

These are not opinions; they are properties of the current viewer's actual behavior, distilled from incident reports in the consuming example module's `LESSONS.md`:

- **Field-mask paths must be camelCase.** The viewer ignores snake_case paths silently. The `Path*` constants in `viam_visuals` are the source of truth.
- **`metadata.*` paths are dropped on UPDATED.** The library compensates by escalating metadata diffs to a REMOVE+ADD respawn (see `Scene.update`).
- **The renderer caches REMOVED UUIDs.** Re-adding the same UUID is silently dropped. `versioned_uuid()` rotates UUIDs to dodge the cache; `Flicker` / `Lifecycle` / metadata-respawn paths use this.
- **PCD bytes must match `pointcloud.ToPCD` byte-for-byte.** Binary, `VERSION .7` literal, no leading `#` comments.
- **Mesh/PCD coordinates are in METERS.** RDK readers multiply by 1000 to convert to the internal mm convention.
- **Only PLY meshes render.** STL is auto-converted to PLY in `Mesh.load`; raw STL is silently dropped by the viewer.
- **`Transform.metadata` uses the visualization schema, NOT the RDK fake's shape.** All five keys are required: `colors`, `color_format`, `opacities`, `show_axes_helper`, `invisible`. The library builds this correctly.

## Status

Pre-1.0. The API is stable enough for the consuming `example-visualizations-python` module but versioned releases are not yet tagged — pin to a git SHA if you need reproducibility. CI / wheel publishing are open follow-ups.

Issues, suggestions, renderer-behavior reports: [open one](https://github.com/viam-labs/viam-viz-helpers-python/issues).
