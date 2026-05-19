# CLAUDE.md — viam-viz-helpers-python

Operational context for agents working in this repository. The user-facing entry point is `README.md`; this file is the load-bearing knowledge an agent needs **before** writing code or proposing changes.

## What this is

`viam-viz-helpers-python` is a Python library that wraps the Viam world-state-store wire format. It is consumed by Viam modules that publish geometries to the 3D scene viewer (`rdk:service:world_state_store`).

- **Repo:** `viam-labs/viam-viz-helpers-python` (public, Apache-2.0).
- **PyPI name (when published):** `viam-viz-helpers`. Not on PyPI yet — install from git.
- **Import name:** `viam_visuals` (not `viamvizhelpers`; the import name preserves the in-tree history of the original example module to avoid migration churn).
- **Go sibling:** `viam-labs/viam-viz-helpers-go` — same architecture, same wire format. Cross-language behavior should stay in lockstep.
- **Canonical consumer:** [`viam-labs/example-visualizations-python`](https://github.com/viam-labs/example-visualizations-python). Read its `LESSONS.md` before debugging anything wire-format-shaped — every renderer quirk has a file:line incident there.

## File layout

```
viam_visuals/__init__.py        # Public API surface — re-exports.
viam_visuals/pose.py            # Pose dataclass, lerp_pose (quaternion SLERP).
viam_visuals/color.py           # Color, hsv_to_rgb, snap_step.
viam_visuals/shapes.py          # Visual base + Box/Sphere/Capsule/Point/Arrow/Frame/Mesh/PointCloud.
viam_visuals/animations.py      # Animation specs (11 modes) + Animation.apply.
viam_visuals/anim_helpers.py    # Pure pose-composing helpers (spin_pose, orbit_pose, …).
viam_visuals/composites.py      # Composite base + CoordinateFrame/Line/BoundingBox/TrajectoryPlan/Arrow.from_to.
viam_visuals/scene.py           # Scene + SceneEvent + diff logic + events_to_wire helper.
viam_visuals/service.py         # SceneServiceBase — inheritable WSS service.
viam_visuals/registry.py        # In-process resource registry.
viam_visuals/uuid_strategy.py   # initial_uuid / versioned_uuid / VALID_STRATEGIES.
viam_visuals/basic_geometry.py  # build_basic_geometry default for standard primitives.
viam_visuals/_internal/         # Pure helpers — constants, metadata struct, mesh / PCD I/O.

tests/                          # pytest. 131 tests; run with `pytest`.
pyproject.toml                  # Build config + project metadata.
pytest.ini                      # asyncio_mode=auto, testpaths=tests.
README.md                       # User-facing surface.
CLAUDE.md                       # This file.
```

## Architecture

### Two halves: data layer + service layer

1. **Data layer** (everything except `service.py`). Typed dataclasses, pure functions, no Viam SDK dependency. Build a Scene, mutate Visuals, call `scene.update(...)` to get diff events. Usable standalone for testing scene logic or building wire-format payloads to send to a remote service.

2. **Service layer** (`service.py`'s `SceneServiceBase`). Subclass it to inherit `list_uuids` / `get_transform` / `stream_transform_changes`, the subscriber fanout, the animation tick task, and the standard DoCommand verb set.

### Scene-centric animation (recommended path)

```python
class MyScene(SceneServiceBase):
    async def reconfigure(self, config, deps):
        self.box = viz.Box("demo", radius_mm=50, animation=viz.Spin(period_s=3))
        self.set_scene(self.box)

    def scene_tick(self, scene, t):
        # Default implementation iterates the scene and calls
        # animation.apply(visual, base, t) on each Visual. Override
        # for custom mutation; call super().scene_tick(scene, t) to
        # chain defaults.
        return super().scene_tick(scene, t)
```

`scene.update(self.box)` diffs the visual's current `to_dict()` against the snapshot taken at `set_scene` time. Three outcomes:

- **No change**: no event.
- **Pose/geom change**: UPDATED with `paths=[...]` (camelCase).
- **Metadata change** (color, opacity, parent_frame, show_axes_helper, invisible): UPDATED with `paths=[]` — the **respawn signal**. The renderer drops `metadata.*` on UPDATED, so the library translates empty-paths UPDATEDs into REMOVE+ADD with a fresh UUID via `versioned_uuid`. See `service.py::_apply_events` for the wire-layer translation.

### Legacy compute_tick path

`SceneServiceBase.compute_tick(item, base_pose, base_geom, t)` is the older per-item animation hook. Pre-existing modules can still use it; new code should use `scene_tick` + `Animation.apply` instead. The library runs whichever path the module overrides; if both, `scene_tick` wins.

### In-process registry

`viam_visuals.registry` is a thread-safe singleton dict mapping `name → instance`. Use it when two resources live in the same module binary and exchange events at tick rate — a direct Python reference is free; the framework's gRPC stub is not. See `registry.py`'s module docstring for the canonical driver→visualizer pattern.

## Renderer quirks (DO NOT FORGET)

These are properties of the current Viam 3D viewer's actual behavior. The library's correctness depends on respecting them. Cross-reference [`example-visualizations-python/LESSONS.md`](https://github.com/viam-labs/example-visualizations-python/blob/main/LESSONS.md) for file:line evidence on each.

- **Field-mask paths must be camelCase.** The viewer's `updateEntity` handler ignores snake_case silently — animations using snake_case paths produce no visible motion. The `PATH_*` constants in this library are camelCase; never edit them to snake_case without confirming the viewer accepts it. The example module's 0.0.32 release shipped snake_case and broke every animation; reverted in 0.0.33.
- **`metadata.*` paths are dropped on UPDATED.** The viewer's `updateEntity` only handles `poseInObserverFrame.pose.*` and `physicalObject.*` prefixes; `metadata.*` is silently dropped. Color/opacity/show_axes_helper/invisible changes must be communicated via REMOVE+ADD respawn (handled automatically by `Scene.update` + `_apply_events`).
- **The renderer caches REMOVED UUIDs.** Re-adding an entity with the same UUID after REMOVED silently no-ops. `versioned_uuid()` rotates UUIDs on every metadata-respawn / flicker / lifecycle re-add to dodge this. Never re-use a UUID after REMOVED.
- **PCD bytes must match `pointcloud.ToPCD` byte-for-byte.** Binary format, `VERSION .7` (literal, not `0.7`), no leading `#` comments. The viewer's parser is strict-order; the RDK reader is lax — so a PCD file might import into other RDK tools fine but render as nothing in the viewer.
- **Mesh/PCD file coordinates are in METERS.** RDK readers multiply by 1000 internally. Writing raw mm into a PLY/PCD file makes the renderer draw it 1000× too big.
- **Only PLY meshes render.** STL is parsed (RDK accepts `content_type="stl"` per `spatialmath/mesh.go:234-243`) but the viewer drops it silently. `Mesh` auto-converts STL→PLY at load time via the library's `_internal/mesh.py`. `Mesh(..., raw_stl=True)` opts out to reproduce the bug; never use this in production.
- **`Transform.metadata` uses the `viamrobotics/visualization` schema.** Five required keys: `colors` (base64 packed RGB bytes), `color_format` (number, 1=RGB), `opacities` (base64 packed alpha bytes), `show_axes_helper` (bool), `invisible` (bool). The library's `_internal/metadata.py` emits this correctly. The RDK fake at `services/worldstatestore/fake/moving_geos_world.go` uses the obsolete `{color, opacity}` shape — do not copy from it.

## API design conventions

- **Dataclasses, not constructors with positional args.** Every Visual / Animation / Composite is a `@dataclass` so authors can use field names and IDEs autocomplete the surface.
- **Pose, Color, animation can each accept either a typed dataclass or a coercible "Like" type.** `pose=Pose.at(x=100)` and `pose={"x": 100}` are equivalent; the `normalize_*` helpers do the coercion. Inside the library, dataclasses are canonical; the Like types are an ergonomics layer for callers.
- **`Animation.apply(visual, base, t)` mutates `visual` in place.** `base` is the rest-state snapshot (frozen at `set_scene` time); `visual` is the live object the Scene tracks. Apply math composes onto `base`, writes to `visual`, then `Scene.update(visual)` produces the diff event.
- **Identifiers prefixed with `_` are internal.** Test against the public surface in `__init__.py`; don't import from `_internal` outside the library itself.

## Tests

```
pip install -e ".[dev]"
pytest
```

131 tests cover the data layer (poses, scenes, animations, registry, snap_step) and exercise the library against fixed inputs. The service layer is exercised by the consuming `example-visualizations-python` module's tests (481 tests there) — significant changes to `service.py` should run that test suite too:

```
cd ../example-visualizations-python
pip install -e ".[dev]"
pip install -e ../viam-viz-helpers-python   # editable install of this repo
pytest
```

## Don't

- **Don't change `PATH_*` constants to snake_case.** The 0.0.32 example module release broke every animation by trying this. The viewer's actual behavior empirically requires camelCase regardless of what the worldstatestore proto guide says.
- **Don't bypass `versioned_uuid` on REMOVE+ADD cycles.** The renderer caches REMOVED UUIDs; re-using the same UUID after REMOVED produces an invisible entity.
- **Don't add features that change the wire format** (new metadata keys, new geometry types) without confirming the viewer accepts them. The `viamrobotics/visualization` repo (`draw/transform.go`, `protos/draw/v1/metadata.proto`) is the canonical source — NOT the RDK fake, which is stale.
- **Don't introduce a runtime dependency on the consuming example module.** This library is upstream of the example modules; the example modules import the library, never the reverse. Tests can use fixtures that mimic module behavior, but production code must not.
- **Don't tag a release** until the v1.0 API decision is made. Pre-1.0 consumers pin to a git SHA in `requirements.txt`; tagged releases are a stronger promise this library isn't ready to make yet.

## Cross-language parity

The Go sibling (`viam-labs/viam-viz-helpers-go`) implements the same architecture, same wire format, same renderer-quirk workarounds. Behavioral changes should land in both libraries together. Discrepancies are bugs.

- The Go side uses `Visual` interface + concrete struct pointers; Python uses dataclass inheritance. Different idioms, same intent.
- Python's `SceneHooks` is method overrides on `SceneServiceBase`; Go's is `type SceneHooks = any` plus optional interface checks. Different language affordances, same surface area.
- Animation math should produce bit-identical poses at the same `t` across languages (modulo floating-point ordering). If you're chasing a Python/Go discrepancy in a consuming module, this is the place to check first.

## Releasing notes

No tagged releases yet — main is the only consumable ref. When v1.0 is cut, add an entry here per release with the date, the API-affecting changes, and any renderer-quirk discoveries that drove the bump.
