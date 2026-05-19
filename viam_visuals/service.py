"""SceneServiceBase — inheritable WorldStateStore service.

A module author who wants to publish a world-state-store scene from a
typed item list subclasses :class:`SceneServiceBase`, sets ``MODEL``,
and overrides the small set of hook methods that touch the file
system or the module's geometry-type set. The base class owns all
the asyncio plumbing — state map, subscriber fanout, the
animation tick task, UUID strategy, the 10 standard DoCommand
verbs (including ``apply_events`` for the driver→visualizer wire
format), and the ``EasyResource.new`` quirk fix that makes initial
construction call ``reconfigure``.

Required hooks (subclass MUST implement):

* :meth:`build_geometry` — build the ``commonpb.Geometry`` proto for
  an item. The base class knows nothing about your primitive types;
  this is where ``build_box`` / ``build_sphere`` / ``build_mesh``
  dispatch lives.
* :meth:`read_asset` — read bytes from an asset path. File-system
  semantics belong to the module.
* :meth:`compute_tick` — given an item dict and time ``t``, return
  ``(pose_dict, geom_override_dict, field_mask_paths, metadata_override)``.
  Where the per-mode animation math lives.
* :meth:`is_animated` — return True iff this item's animation should
  drive ticks. Usually a thin wrapper over a dict-mode check.

Optional hooks (override to extend the defaults):

* :meth:`load_preset` — fetch a preset by name. Defaults to raising;
  override if your module ships presets.
* :meth:`preset_names` — return the list of valid preset names.
* :meth:`validate_item_extra` — module-specific schema validation
  (e.g., the playground's ``raw_stl`` knob).
* :meth:`handle_custom_command` — handle DoCommand verbs the base
  class doesn't know about. Return ``None`` to fall through to the
  default debug-snapshot reply.

The base class is intentionally not abstract — defaults exist so
unit tests can instantiate it without subclassing for every method.
"""

from __future__ import annotations

import asyncio
import base64
import time
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
    ClassVar,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from google.protobuf.field_mask_pb2 import FieldMask
from typing_extensions import Self
from viam.logging import getLogger
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import (
    Geometry,
    PoseInFrame,
    ResourceName,
    Transform,
)
from viam.proto.service.worldstatestore import (
    StreamTransformChangesResponse,
    TransformChangeType,
)
from viam.resource.base import ResourceBase
from viam.resource.types import Model
from viam.services.worldstatestore import WorldStateStore
from viam.utils import ValueTypes, struct_to_dict

from ._internal.metadata import build_metadata
from ._internal.mesh import extract_ply_vertex_colors
from ._internal.pcd import build_pcd_chunk, parse_pcd_binary
from .animations import Animation
from .composites import Composite
from .scene import Scene, SceneEvent, events_to_wire
from .shapes import Visual
from .uuid_strategy import VALID_STRATEGIES, initial_uuid, versioned_uuid


__all__ = ["SceneServiceBase"]


LOGGER = getLogger(__name__)


# ---------- config attribute names ----------
#
# Subclasses can override these (and the defaults below) if they want
# different attribute names, but the standard schema is what every
# adopter needs out of the box.

ATTR_TICK_HZ = "tick_hz"
ATTR_UUID_STRATEGY = "uuid_strategy"
ATTR_PARENT_FRAME = "parent_frame"
ATTR_PRESET = "preset"
ATTR_ITEMS = "items"

# Convenience module-level constants — match the class-level defaults
# on SceneServiceBase. Importable as `from viam_visuals import
# DEFAULT_TICK_HZ`.
DEFAULT_TICK_HZ: float = 30.0
DEFAULT_UUID_STRATEGY: str = "stable"
DEFAULT_PARENT_FRAME: str = "world"


# ---------- pose key → field-mask path ----------

_POSE_KEY_TO_PATH = {
    "x": "poseInObserverFrame.pose.x",
    "y": "poseInObserverFrame.pose.y",
    "z": "poseInObserverFrame.pose.z",
    "theta": "poseInObserverFrame.pose.theta",
    # ox/oy/oz aren't covered by the RDK fake's path conventions;
    # whole-pose updates are safe via reconfigure.
}


def _base_geom_for_item(item: Mapping[str, Any]) -> Dict[str, Any]:
    """Shape-specific dim/radius/length fields the animator needs.
    Library default; modules with custom sugar types can override
    via :meth:`SceneServiceBase.base_geom_for_item`."""
    t = item["type"]
    if t == "box":
        return {"dims_mm": dict(item["dims_mm"])}
    if t == "sphere":
        return {"radius_mm": float(item["radius_mm"])}
    if t == "capsule":
        return {
            "radius_mm": float(item["radius_mm"]),
            "length_mm": float(item["length_mm"]),
        }
    if t == "arrow":
        return {
            "radius_mm": float(item["radius_mm"]),
            "length_mm": float(item["length_mm"]),
        }
    return {}


class SceneServiceBase(WorldStateStore):
    """Inheritable base for WorldStateStore service modules.

    See module docstring for the contract.

    **Subclass pattern:** subclasses inherit both this base and
    ``viam.resource.easy_resource.EasyResource``; this base
    intentionally does NOT inherit ``EasyResource`` itself because
    ``EasyResource.__init_subclass__`` requires a ``MODEL`` field
    and an abstract base shouldn't declare one. The standard shape:

        from viam.resource.easy_resource import EasyResource
        from viam.resource.types import Model, ModelFamily
        from viam_visuals import SceneServiceBase

        class MyScene(SceneServiceBase, EasyResource):
            MODEL = Model(ModelFamily("acme", "viz"), "scene")
            # override build_geometry, read_asset, compute_tick,
            # is_animated as needed.
    """

    # Subclasses set MODEL to their resource model identifier.
    MODEL: ClassVar[Model]

    # Defaults — subclasses may override.
    DEFAULT_TICK_HZ: ClassVar[float] = 30.0
    DEFAULT_UUID_STRATEGY: ClassVar[str] = "stable"
    DEFAULT_PARENT_FRAME: ClassVar[str] = "world"
    DEFAULT_PRESET: ClassVar[Optional[str]] = None
    DEFAULT_CHUNK_SIZE_POINTS: ClassVar[int] = 1000
    MAX_TICK_HZ: ClassVar[float] = 30.0

    def __init__(self, name: str):
        super().__init__(name)
        self._lock = asyncio.Lock()
        self._state: Dict[str, Dict[str, Any]] = {}
        self._subscribers: List[asyncio.Queue] = []
        self._tick_task: Optional[asyncio.Task] = None
        self.tick_hz: float = self.DEFAULT_TICK_HZ
        self.uuid_strategy: str = self.DEFAULT_UUID_STRATEGY
        self.parent_frame: str = self.DEFAULT_PARENT_FRAME
        self._animation_t0: float = 0.0
        # Scene-centric API: a typed object-graph that backs the
        # service state. Subclasses install Visuals via set_scene(...)
        # and mutate them in tick(scene, t) — the library handles
        # diff'ing, field-mask path emission, and renderer-quirk
        # workarounds (metadata-only → REMOVE+ADD respawn) internally.
        self.scene: Scene = Scene(parent_frame=self.parent_frame)
        # Per-label snapshots of Visuals at install time. The default
        # scene_tick passes these as the "base" / rest state when
        # calling Animation.apply on each animated Visual; the apply
        # method computes deltas relative to base rather than mutating
        # state through itself.
        self._base_visuals: Dict[str, Visual] = {}

    # ------------------------------------------------------------------
    # Required hooks — subclass MUST implement
    # ------------------------------------------------------------------

    def build_geometry(
        self, item: Mapping[str, Any], override_geom: Mapping[str, Any]
    ) -> Geometry:
        """Build the ``commonpb.Geometry`` proto for an item.

        Subclass dispatches on ``item['type']`` to call the
        appropriate ``build_box`` / ``build_sphere`` / ``build_mesh``
        / etc. The base class doesn't know about your primitive types.

        ``override_geom`` carries per-tick geometry overrides
        (currently only ``pulse`` and chunked-pointcloud paths use
        this); pass it through to your shape builders for those that
        accept overrides.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override build_geometry()"
        )

    def read_asset(self, asset_path: str) -> bytes:
        """Read an asset's bytes. Modules typically resolve relative
        paths against their installed module directory."""
        raise NotImplementedError(
            f"{type(self).__name__} must override read_asset()"
        )

    def scene_tick(self, scene: Scene, t: float) -> Sequence[SceneEvent]:
        """Per-tick animation hook (recommended API).

        Called by the library's tick loop at ``tick_hz`` (default
        30 Hz). The default implementation iterates the scene and
        dispatches to each Visual's ``animation.apply(visual, base, t)``
        — the typed Animation dataclass (Spin, Orbit, Pulse, …)
        mutates the Visual based on its rest state. The returned
        events are broadcast to subscribers.

        Subclasses can override for custom behavior (mutate Visuals
        directly without going through Animation specs), or call
        ``super().scene_tick(scene, t)`` to chain the default
        dispatch with custom code.

        Example — custom override::

            def scene_tick(self, scene, t):
                self.my_box.pose = viz.spin_pose(self.base_pose, 3.0, t)
                return scene.update(self.my_box)

        Example — declarative (use Animation specs on Visuals,
        rely on the default dispatch)::

            self.set_scene(
                viz.Sphere("a", radius_mm=50, animation=viz.Spin(period_s=3)),
                viz.Box("b", dims_mm=(100,100,100), animation=viz.Pulse(amplitude_mm=20)),
            )
            # No scene_tick override needed; defaults dispatch each
            # Animation.apply at tick_hz.
        """
        events: List[SceneEvent] = []
        for label in scene.labels():
            v = scene.get(label)
            if v is None:
                continue
            anim = v.animation
            if not isinstance(anim, Animation):
                continue  # No spec, or legacy dict spec (compute_tick path)
            base = self._base_visuals.get(label)
            if base is None:
                continue
            try:
                anim.apply(v, base, t)
            except Exception as e:
                LOGGER.warning(
                    f"animation.apply failed for {label!r} ({type(anim).__name__}): "
                    f"{type(e).__name__}: {e}"
                )
                continue
            events.extend(scene.update(v))
        return events

    def compute_tick(
        self,
        item: Mapping[str, Any],
        base_pose: Mapping[str, float],
        base_geom: Mapping[str, Any],
        t: float,
    ) -> Tuple[
        Mapping[str, float],
        Mapping[str, Any],
        Sequence[str],
        Optional[Mapping[str, Any]],
    ]:
        """Legacy per-item animation evaluation. Return
        ``(pose_dict, geom_override_dict, field_mask_paths, metadata_override)``.

        .. deprecated::
            Use :meth:`scene_tick` (scene-centric) instead. The new
            API mutates typed Visual objects directly, avoiding the
            tuple-return shape and field-mask-path bookkeeping.

        The default implementation is "no animation" — returns the
        base pose, no geom override, no paths, no overrides.
        """
        return base_pose, {}, [], None

    def is_animated(self, item: Mapping[str, Any]) -> bool:
        """Return True iff this item's animation should drive ticks
        under the legacy :meth:`compute_tick` path. The default reads
        ``item.animation.mode`` and returns False for ``"none"`` or
        absent.

        Not consulted under the new :meth:`scene_tick` path — that
        path runs every tick and emits no events if
        ``scene.update(...)`` returns nothing.
        """
        anim = item.get("animation") or {}
        mode = anim.get("mode", "none")
        return mode != "" and mode != "none"

    def _has_scene_tick(self) -> bool:
        """True if the subclass overrode :meth:`scene_tick`, OR the
        scene has any typed Animation specs (default dispatch will
        produce work)."""
        if type(self).scene_tick is not SceneServiceBase.scene_tick:
            return True
        for label in self.scene.labels():
            v = self.scene.get(label)
            if v is not None and isinstance(v.animation, Animation):
                return True
        return False

    # ------------------------------------------------------------------
    # Optional hooks
    # ------------------------------------------------------------------

    def load_preset(self, name: str) -> Sequence[Mapping[str, Any]]:
        """Load a named preset. Subclass overrides to provide its
        presets dict; default raises ``Exception`` because the base
        class doesn't know any presets."""
        raise Exception(f"unknown preset {name!r}")

    def preset_names(self) -> Sequence[str]:
        """Names of all presets the subclass provides. Used by
        validate_config and the ``preset`` DoCommand verb."""
        return ()

    def validate_item_extra(
        self, item: Mapping[str, Any], index: int
    ) -> None:
        """Hook for module-specific item validation beyond the
        standard schema. Override to validate sugar knobs
        (e.g., the playground's ``raw_stl``). Default: no-op."""
        return None

    def base_geom_for_item(self, item: Mapping[str, Any]) -> Dict[str, Any]:
        """Override if your module adds sugar primitive types that
        carry shape-specific fields. Default handles box, sphere,
        capsule, arrow, mesh, pointcloud, point."""
        return _base_geom_for_item(item)

    async def handle_custom_command(
        self, command: Mapping[str, ValueTypes]
    ) -> Optional[Mapping[str, ValueTypes]]:
        """Handle a DoCommand verb the base class doesn't know about.
        Return ``None`` to fall through to the default debug-snapshot
        reply. Hold ``self._lock`` if you touch ``self._state``.

        Standard verbs the base class handles itself: list, add,
        remove, update, clear, preset, snapshot, set_uuid_strategy.
        """
        return None

    # ------------------------------------------------------------------
    # Lifecycle (EasyResource quirk + reconfigure + close)
    # ------------------------------------------------------------------

    @classmethod
    def new(
        cls,
        config: ComponentConfig,
        dependencies: Mapping[ResourceName, ResourceBase],
    ) -> Self:
        """``EasyResource.new`` doesn't call ``reconfigure`` for service
        models, so we do it explicitly. Without this the service
        starts with no items and the tick task never launches."""
        instance = super().new(config, dependencies)
        instance.reconfigure(config, dependencies)
        return instance

    @classmethod
    def validate_config(
        cls, config: ComponentConfig
    ) -> Tuple[Sequence[str], Sequence[str]]:
        """Validate the standard schema fields. Subclasses can extend
        by overriding and chaining via ``super().validate_config(...)``.

        Returns ``(required_deps, optional_deps)`` per the Viam SDK
        contract."""
        attrs = struct_to_dict(config.attributes)

        if ATTR_TICK_HZ in attrs:
            hz = float(attrs[ATTR_TICK_HZ])
            if hz <= 0 or hz > cls.MAX_TICK_HZ:
                raise Exception(
                    f"{ATTR_TICK_HZ} must be in (0, {cls.MAX_TICK_HZ}]"
                )
        if ATTR_UUID_STRATEGY in attrs:
            s = str(attrs[ATTR_UUID_STRATEGY])
            if s not in VALID_STRATEGIES:
                raise Exception(
                    f"{ATTR_UUID_STRATEGY} must be one of {VALID_STRATEGIES}, got {s!r}"
                )
        return [], []

    def reconfigure(
        self,
        config: ComponentConfig,
        dependencies: Mapping[ResourceName, ResourceBase],
    ):
        attrs = struct_to_dict(config.attributes)
        tick_hz = float(attrs.get(ATTR_TICK_HZ, self.DEFAULT_TICK_HZ))
        uuid_strategy = str(
            attrs.get(ATTR_UUID_STRATEGY, self.DEFAULT_UUID_STRATEGY)
        )
        parent_frame = str(
            attrs.get(ATTR_PARENT_FRAME, self.DEFAULT_PARENT_FRAME)
        )

        # Pick item source: explicit items > preset > default preset.
        raw_items = attrs.get(ATTR_ITEMS)
        if raw_items:
            items: List[Dict[str, Any]] = [dict(it) for it in raw_items]
        else:
            preset_name = attrs.get(ATTR_PRESET, self.DEFAULT_PRESET)
            if preset_name is None:
                items = []
            else:
                items = [dict(it) for it in self.load_preset(str(preset_name))]

        self.reconfigure_with(
            items,
            tick_hz=tick_hz,
            uuid_strategy=uuid_strategy,
            parent_frame=parent_frame,
        )

    def set_scene(
        self,
        *visuals: Union[Visual, Composite],
        tick_hz: Optional[float] = None,
        uuid_strategy: Optional[str] = None,
        parent_frame: Optional[str] = None,
    ) -> None:
        """Install typed Visual / Composite objects as the new scene.
        Composites expand to their constituent Visuals; each is
        tracked in :attr:`self.scene` so subclasses can keep
        references and mutate them on each :meth:`tick`.

        Broadcasts REMOVED for any prior state and ADDED for the new
        state, then restarts the tick task if this service uses
        animation.

        Example::

            def reconfigure(self, config, deps):
                self.my_box = viz.Box(
                    "demo_box", viz.Pose.at(z=100),
                    dims_mm=(150, 150, 150),
                    color=(230, 25, 75),
                )
                self.set_scene(self.my_box)

        For services that build wire-format dicts directly (no typed
        objects), use :meth:`reconfigure_with` instead.
        """
        parent = parent_frame if parent_frame is not None else self.parent_frame
        # Build a fresh Scene so subscribers' initial-burst sees the
        # post-mutation snapshot.
        self.scene = Scene(parent_frame=parent)
        self.scene.add(*visuals)
        # Snapshot each Visual's rest state for the default
        # scene_tick dispatch (Animation.apply takes base, t).
        import copy as _copy
        self._base_visuals = {
            label: _copy.deepcopy(self.scene.get(label))
            for label in self.scene.labels()
        }
        items = [entry.committed for entry in self.scene._state.values()]
        self.reconfigure_with(
            items,
            tick_hz=tick_hz,
            uuid_strategy=uuid_strategy,
            parent_frame=parent,
        )

    def reconfigure_with(
        self,
        items: Sequence[Mapping[str, Any]],
        tick_hz: Optional[float] = None,
        uuid_strategy: Optional[str] = None,
        parent_frame: Optional[str] = None,
    ) -> None:
        """Install ``items`` as the new scene, broadcasting REMOVED
        for prior state and ADDED for the new state, restarting the
        tick task if any items animate.

        Subclasses whose scene is built in code (not from config
        attributes) should override :meth:`reconfigure` to construct
        the item list and call this method directly, instead of
        going through the ``items`` / ``preset`` config plumbing.

        ``None`` arguments use the class defaults (``DEFAULT_TICK_HZ``
        etc.).
        """
        self.tick_hz = float(self.DEFAULT_TICK_HZ if tick_hz is None else tick_hz)
        self.uuid_strategy = str(
            self.DEFAULT_UUID_STRATEGY if uuid_strategy is None else uuid_strategy
        )
        self.parent_frame = str(
            self.DEFAULT_PARENT_FRAME if parent_frame is None else parent_frame
        )

        # Cancel any prior tick task.
        if self._tick_task is not None:
            self._tick_task.cancel()
            self._tick_task = None

        # Rebuild state from scratch with REMOVED/ADDED broadcast.
        prior_transforms = [s["transform"] for s in self._state.values()]
        self._state = {}
        for it in items:
            self._install_item(dict(it))

        for t in prior_transforms:
            self._broadcast(StreamTransformChangesResponse(
                change_type=TransformChangeType.TRANSFORM_CHANGE_TYPE_REMOVED,
                transform=t,
            ))
        for s in self._state.values():
            self._broadcast(StreamTransformChangesResponse(
                change_type=TransformChangeType.TRANSFORM_CHANGE_TYPE_ADDED,
                transform=s["transform"],
            ))

        # Reset animation clock and (re)start the tick task if the
        # subclass overrides ``tick(scene, t)`` (Scene-centric path)
        # OR any item has a declarative animation spec (legacy
        # ``compute_tick`` path).
        self._animation_t0 = time.monotonic()
        wants_tick = self._has_scene_tick() or any(
            self.is_animated(s["item"]) for s in self._state.values()
        )
        if wants_tick:
            try:
                self._tick_task = asyncio.create_task(self._tick_loop())
            except RuntimeError:
                self._tick_task = None
        LOGGER.info(
            f"reconfigure: tick_hz={self.tick_hz} "
            f"uuid_strategy={self.uuid_strategy} "
            f"parent_frame={self.parent_frame} items={len(self._state)}"
        )

    async def close(self):
        if self._tick_task is not None:
            self._tick_task.cancel()
            try:
                await self._tick_task
            except asyncio.CancelledError:
                pass

    # ------------------------------------------------------------------
    # Item lifecycle helpers (sync — caller holds lock or is in
    # reconfigure where the loop isn't yet running).
    # ------------------------------------------------------------------

    def _install_item(self, item: Mapping[str, Any]) -> None:
        """Bring an item into state and build its initial Transform.
        Handles chunked-delivery setup for pointcloud items."""
        label = item["label"]
        if label in self._state:
            raise Exception(f"duplicate item label {label!r}")
        base_pose = dict(item.get("pose") or {})
        # Fill missing pose fields with identity so animation math
        # can read keys directly.
        for k, default in (("x", 0.0), ("y", 0.0), ("z", 0.0),
                           ("ox", 0.0), ("oy", 0.0), ("oz", 1.0),
                           ("theta", 0.0)):
            base_pose.setdefault(k, default)
        base_geom = self.base_geom_for_item(item)
        uuid = initial_uuid(label, self.uuid_strategy)

        chunks_info = None
        chunked_state = None
        if item.get("type") == "pointcloud" and item.get("chunked"):
            full_pcd = self.read_asset(item["pointcloud_path"])
            header_bytes, body_bytes, stride, total_points = parse_pcd_binary(full_pcd)
            chunk_size_points = int(
                item.get("chunk_size", self.DEFAULT_CHUNK_SIZE_POINTS)
            )
            if chunk_size_points <= 0:
                chunk_size_points = self.DEFAULT_CHUNK_SIZE_POINTS
            n_chunks = (total_points + chunk_size_points - 1) // chunk_size_points
            first_chunk_pcd = build_pcd_chunk(
                header_bytes, body_bytes, stride,
                chunk_index=0, chunk_size_points=chunk_size_points,
            )
            base_geom = dict(base_geom)
            base_geom["pcd_bytes"] = first_chunk_pcd
            chunks_info = {
                "chunk_size": float(chunk_size_points),
                "total": float(n_chunks),
                "total_points": float(total_points),
                "stride": float(stride),
            }
            chunked_state = {
                "header_bytes": header_bytes,
                "body_bytes": body_bytes,
                "stride": stride,
                "total_points": total_points,
                "chunk_size_points": chunk_size_points,
                "n_chunks": n_chunks,
            }
        geom_proto = self.build_geometry(item, base_geom)
        tf = self._build_transform(
            item, base_pose, geom_proto, uuid, self.parent_frame,
            chunks=chunks_info,
        )
        self._state[label] = {
            "item": dict(item),
            "base_pose": base_pose,
            "base_geom": base_geom,
            "uuid": uuid,
            "transform": tf,
            "chunks_info": chunks_info,
            "chunked_state": chunked_state,
            "visible_to_viewer": True,
        }

    def _remove_item(self, label: str) -> Optional[Transform]:
        s = self._state.pop(label, None)
        return s["transform"] if s is not None else None

    def _build_transform(
        self,
        item: Mapping[str, Any],
        pose: Mapping[str, float],
        geom: Geometry,
        uuid: bytes,
        parent_frame: str,
        chunks: Optional[Mapping[str, Any]] = None,
    ) -> Transform:
        """Assemble a ``Transform`` proto from an item + pose + geom.
        Handles the PLY-vertex-color transcoding to metadata.colors."""
        vertex_colors = None
        user_color = item.get("color")
        if user_color is None and geom.HasField("mesh"):
            vertex_colors = extract_ply_vertex_colors(geom.mesh.mesh)
        metadata = build_metadata(
            user_color,
            item.get("opacity"),
            show_axes_helper=bool(item.get("show_axes_helper", False)),
            invisible=bool(item.get("invisible", False)),
            vertex_colors=vertex_colors,
            chunks=chunks,
        )
        from viam.proto.common import Pose as _ProtoPose  # local to keep base scope tight

        proto_pose = _ProtoPose(
            x=float(pose.get("x", 0.0)),
            y=float(pose.get("y", 0.0)),
            z=float(pose.get("z", 0.0)),
            o_x=float(pose.get("ox", 0.0)),
            o_y=float(pose.get("oy", 0.0)),
            o_z=float(pose.get("oz", 1.0)),
            theta=float(pose.get("theta", 0.0)),
        )
        return Transform(
            uuid=uuid,
            reference_frame=item["label"],
            pose_in_observer_frame=PoseInFrame(
                reference_frame=item.get("parent_frame", parent_frame),
                pose=proto_pose,
            ),
            physical_object=geom,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Animation tick
    # ------------------------------------------------------------------

    async def _tick_loop(self) -> None:
        period = 1.0 / max(0.01, self.tick_hz)
        try:
            while True:
                try:
                    await self._tick_once()
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    LOGGER.warning(f"tick failed: {type(e).__name__}: {e}")
                await asyncio.sleep(period)
        except asyncio.CancelledError:
            return

    async def _tick_once(self) -> None:
        t = time.monotonic() - self._animation_t0

        # Scene-centric tick path: if the subclass overrides
        # ``scene_tick(scene, t)``, call it and apply the returned
        # events through the same machinery ``apply_events`` uses.
        # Subclass gets the typed Scene API; the library handles
        # wire format, subscriber broadcasts, and the metadata-only-
        # respawn intercept.
        if self._has_scene_tick():
            try:
                events = list(self.scene_tick(self.scene, t) or [])
            except Exception as e:
                LOGGER.warning(f"scene_tick failed: {type(e).__name__}: {e}")
                events = []
            if events:
                wire_events = events_to_wire(events)
                await self._apply_events({"events": wire_events})
            return

        async with self._lock:
            for label, s in list(self._state.items()):
                item = s["item"]
                if not self.is_animated(item):
                    continue
                pose, geom, paths, meta_override = self.compute_tick(
                    item, s["base_pose"], s["base_geom"], t,
                )
                # Scene-graph membership transitions (flicker, lifecycle).
                if meta_override and "_in_scene" in meta_override:
                    want_in_scene = bool(meta_override["_in_scene"])
                    was_in_scene = s.get("visible_to_viewer", True)
                    if want_in_scene and not was_in_scene:
                        # Rising edge: re-add. Rotate UUID by default
                        # to dodge the renderer's REMOVED-UUID cache.
                        anim_cfg = item.get("animation") or {}
                        if anim_cfg.get("rotate_uuid_on_readd", True):
                            new_uuid = versioned_uuid(label)
                            s["uuid"] = new_uuid
                        emit_uuid = s["uuid"]
                        item_for_add = item
                        if "color" in meta_override or "opacity" in meta_override:
                            item_for_add = dict(item)
                            if "color" in meta_override:
                                c = meta_override["color"]
                                item_for_add["color"] = {"r": c[0], "g": c[1], "b": c[2]}
                            if "opacity" in meta_override:
                                item_for_add["opacity"] = float(meta_override["opacity"])
                        geom_proto = self.build_geometry(item, geom)
                        new_tf = self._build_transform(
                            item_for_add, pose, geom_proto, emit_uuid, self.parent_frame,
                        )
                        s["transform"] = new_tf
                        s["visible_to_viewer"] = True
                        self._broadcast(StreamTransformChangesResponse(
                            change_type=TransformChangeType.TRANSFORM_CHANGE_TYPE_ADDED,
                            transform=new_tf,
                        ))
                        continue
                    if not want_in_scene and was_in_scene:
                        s["visible_to_viewer"] = False
                        self._broadcast(StreamTransformChangesResponse(
                            change_type=TransformChangeType.TRANSFORM_CHANGE_TYPE_REMOVED,
                            transform=s["transform"],
                        ))
                        continue
                    if not want_in_scene and not was_in_scene:
                        continue
                    # Both on — fall through to UPDATED path.
                if not paths:
                    continue
                geom_proto = self.build_geometry(item, geom)
                item_for_tf = item
                if meta_override:
                    item_for_tf = dict(item)
                    if "color" in meta_override:
                        c = meta_override["color"]
                        item_for_tf["color"] = {"r": c[0], "g": c[1], "b": c[2]}
                    if "opacity" in meta_override:
                        item_for_tf["opacity"] = float(meta_override["opacity"])
                if self.uuid_strategy == "stable":
                    new_tf = self._build_transform(
                        item_for_tf, pose, geom_proto, s["uuid"], self.parent_frame,
                    )
                    s["transform"] = new_tf
                    self._broadcast(StreamTransformChangesResponse(
                        change_type=TransformChangeType.TRANSFORM_CHANGE_TYPE_UPDATED,
                        transform=new_tf,
                        updated_fields=FieldMask(paths=list(paths)),
                    ))
                else:  # versioned
                    old_tf = s["transform"]
                    new_uuid = versioned_uuid(label)
                    new_tf = self._build_transform(
                        item_for_tf, pose, geom_proto, new_uuid, self.parent_frame,
                    )
                    s["uuid"] = new_uuid
                    s["transform"] = new_tf
                    self._broadcast(StreamTransformChangesResponse(
                        change_type=TransformChangeType.TRANSFORM_CHANGE_TYPE_REMOVED,
                        transform=old_tf,
                    ))
                    self._broadcast(StreamTransformChangesResponse(
                        change_type=TransformChangeType.TRANSFORM_CHANGE_TYPE_ADDED,
                        transform=new_tf,
                    ))

    # ------------------------------------------------------------------
    # Subscriber fanout
    # ------------------------------------------------------------------

    def _broadcast(self, msg: StreamTransformChangesResponse) -> None:
        for q in list(self._subscribers):
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                LOGGER.warning("subscriber queue full; dropping event")

    # ------------------------------------------------------------------
    # WorldStateStore service API
    # ------------------------------------------------------------------

    async def list_uuids(
        self,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[bytes]:
        async with self._lock:
            return [
                s["uuid"]
                for s in self._state.values()
                if s.get("visible_to_viewer", True)
            ]

    async def get_transform(
        self,
        uuid: bytes,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Transform:
        async with self._lock:
            for s in self._state.values():
                if s["uuid"] == uuid:
                    if not s.get("visible_to_viewer", True):
                        raise Exception(
                            f"uuid {uuid!r} is currently not in the scene "
                            "(flicker animation has it temporarily removed)"
                        )
                    return s["transform"]
        raise Exception(f"unknown uuid {uuid!r}")

    async def stream_transform_changes(
        self,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> AsyncGenerator[StreamTransformChangesResponse, None]:
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        async with self._lock:
            self._subscribers.append(q)
            for s in self._state.values():
                if not s.get("visible_to_viewer", True):
                    continue
                q.put_nowait(StreamTransformChangesResponse(
                    change_type=TransformChangeType.TRANSFORM_CHANGE_TYPE_ADDED,
                    transform=s["transform"],
                ))
        try:
            while True:
                yield await q.get()
        finally:
            async with self._lock:
                if q in self._subscribers:
                    self._subscribers.remove(q)

    # ------------------------------------------------------------------
    # DoCommand — standard verb dispatcher
    # ------------------------------------------------------------------

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Mapping[str, ValueTypes]:
        cmd = command.get("command") if command else None

        if cmd == "list":
            async with self._lock:
                return {"items": [self._item_summary(label) for label in self._state]}

        if cmd == "add":
            new_item = command.get("item")
            if not isinstance(new_item, Mapping):
                raise Exception("add requires an 'item' dict")
            self.validate_item_extra(new_item, 0)
            async with self._lock:
                if new_item["label"] in self._state:
                    raise Exception(f"item {new_item['label']!r} already exists")
                self._install_item(new_item)
                tf = self._state[new_item["label"]]["transform"]
                self._broadcast(StreamTransformChangesResponse(
                    change_type=TransformChangeType.TRANSFORM_CHANGE_TYPE_ADDED,
                    transform=tf,
                ))
                self._maybe_restart_tick()
                return {"label": new_item["label"], "uuid": tf.uuid.decode()}

        if cmd == "remove":
            label = command.get("label")
            if not label:
                raise Exception("remove requires a 'label'")
            async with self._lock:
                tf = self._remove_item(str(label))
                if tf is None:
                    return {"removed": False}
                self._broadcast(StreamTransformChangesResponse(
                    change_type=TransformChangeType.TRANSFORM_CHANGE_TYPE_REMOVED,
                    transform=tf,
                ))
                return {"removed": True}

        if cmd == "update":
            label = command.get("label")
            patch = command.get("patch")
            if not label or not isinstance(patch, Mapping):
                raise Exception("update requires 'label' and 'patch'")
            async with self._lock:
                s = self._state.get(str(label))
                if s is None:
                    raise Exception(f"unknown label {label!r}")
                updated_fields = self._apply_patch(s, patch)
                geom_proto = self.build_geometry(s["item"], s["base_geom"])
                new_tf = self._build_transform(
                    s["item"], s["base_pose"], geom_proto, s["uuid"], self.parent_frame,
                )
                s["transform"] = new_tf
                self._broadcast(StreamTransformChangesResponse(
                    change_type=TransformChangeType.TRANSFORM_CHANGE_TYPE_UPDATED,
                    transform=new_tf,
                    updated_fields=FieldMask(paths=updated_fields),
                ))
                self._maybe_restart_tick()
                return {"updated_fields": updated_fields}

        if cmd == "clear":
            async with self._lock:
                count = len(self._state)
                for tf in [s["transform"] for s in self._state.values()]:
                    self._broadcast(StreamTransformChangesResponse(
                        change_type=TransformChangeType.TRANSFORM_CHANGE_TYPE_REMOVED,
                        transform=tf,
                    ))
                self._state = {}
                return {"removed_count": count}

        if cmd == "preset":
            name = command.get("name")
            if not name:
                raise Exception("preset requires a 'name'")
            items = self.load_preset(str(name))
            async with self._lock:
                prior = [s["transform"] for s in self._state.values()]
                self._state = {}
                for it in items:
                    self._install_item(dict(it))
                for tf in prior:
                    self._broadcast(StreamTransformChangesResponse(
                        change_type=TransformChangeType.TRANSFORM_CHANGE_TYPE_REMOVED,
                        transform=tf,
                    ))
                for s in self._state.values():
                    self._broadcast(StreamTransformChangesResponse(
                        change_type=TransformChangeType.TRANSFORM_CHANGE_TYPE_ADDED,
                        transform=s["transform"],
                    ))
                self._maybe_restart_tick()
                return {"loaded": str(name), "count": len(self._state)}

        if cmd == "snapshot":
            async with self._lock:
                return {"config": {
                    "tick_hz": self.tick_hz,
                    "uuid_strategy": self.uuid_strategy,
                    "parent_frame": self.parent_frame,
                    "items": [dict(s["item"]) for s in self._state.values()],
                }}

        if cmd == "set_uuid_strategy":
            strategy = command.get("strategy")
            if strategy not in VALID_STRATEGIES:
                raise Exception(
                    f"strategy must be one of {VALID_STRATEGIES}, got {strategy!r}"
                )
            async with self._lock:
                self.uuid_strategy = str(strategy)
                return {"strategy": self.uuid_strategy}

        if cmd == "apply_events":
            return await self._apply_events(command)

        # Custom verb hook — subclass can return a Mapping for known
        # custom verbs, or None to fall through.
        custom = await self.handle_custom_command(command)
        if custom is not None:
            return custom

        # Default: debug snapshot for unrecognized / missing commands.
        return self._debug_snapshot()

    # ------------------------------------------------------------------
    # do_command helpers
    # ------------------------------------------------------------------

    def _item_summary(self, label: str) -> Mapping[str, Any]:
        s = self._state[label]
        return {
            "label": label,
            "type": s["item"]["type"],
            "uuid": s["uuid"].decode(),
            "pose": dict(s["base_pose"]),
            "animation_mode": (s["item"].get("animation") or {}).get("mode", "none"),
            "color": s["item"].get("color"),
            "opacity": s["item"].get("opacity"),
        }

    def _apply_patch(self, s: Dict[str, Any], patch: Mapping[str, Any]) -> List[str]:
        updated_fields: List[str] = []
        item = s["item"]
        if "color" in patch:
            item["color"] = dict(patch["color"]) if patch["color"] is not None else None
            updated_fields.append("metadata.color")
        if "opacity" in patch:
            item["opacity"] = (
                float(patch["opacity"]) if patch["opacity"] is not None else None
            )
            updated_fields.append("metadata.opacity")
        if "pose" in patch:
            new_pose = dict(s["base_pose"])
            new_pose.update(patch["pose"])
            s["base_pose"] = new_pose
            for k in patch["pose"]:
                fm = _POSE_KEY_TO_PATH.get(k)
                if fm is not None:
                    updated_fields.append(fm)
        if "dims_mm" in patch and item["type"] == "box":
            item["dims_mm"] = dict(patch["dims_mm"])
            s["base_geom"] = self.base_geom_for_item(item)
            updated_fields.extend([
                "physicalObject.geometryType.value.dimsMm.x",
                "physicalObject.geometryType.value.dimsMm.y",
                "physicalObject.geometryType.value.dimsMm.z",
            ])
        if "radius_mm" in patch and item["type"] in ("sphere", "capsule"):
            item["radius_mm"] = float(patch["radius_mm"])
            s["base_geom"] = self.base_geom_for_item(item)
            updated_fields.append("physicalObject.geometryType.value.radiusMm")
        if "length_mm" in patch and item["type"] == "capsule":
            item["length_mm"] = float(patch["length_mm"])
            s["base_geom"] = self.base_geom_for_item(item)
            updated_fields.append("physicalObject.geometryType.value.lengthMm")
        if "mesh_path" in patch and item["type"] == "mesh":
            item["mesh_path"] = str(patch["mesh_path"])
            updated_fields.append("physicalObject.mesh")
        if "pointcloud_path" in patch and item["type"] == "pointcloud":
            item["pointcloud_path"] = str(patch["pointcloud_path"])
            updated_fields.append("physicalObject.pointcloud")
        if "animation" in patch:
            item["animation"] = dict(patch["animation"])
        return updated_fields

    async def _apply_events(
        self, command: Mapping[str, ValueTypes],
    ) -> Mapping[str, ValueTypes]:
        """Apply a batch of SceneEvent wire-format records.

        The Scene class on the driver side produces these via
        ``Scene.add/update/remove``; the driver serializes them to the
        wire format (kind + label + item + paths) and pushes the batch
        here. Returns counters for each kind plus a list of per-event
        errors. Errors don't abort the batch — remaining events still
        apply.

        Optional ``namespace`` prefixes every label so multiple drivers
        can share one visualizer without collisions.
        """
        events = command.get("events") or []
        if not isinstance(events, list):
            raise Exception("apply_events requires 'events' as a list")
        namespace = str(command.get("namespace") or "")
        prefix = (namespace + "/") if namespace else ""

        added = updated = removed = 0
        errors: List[str] = []
        async with self._lock:
            for i, evt in enumerate(events):
                if not isinstance(evt, Mapping):
                    errors.append(f"event[{i}]: not a dict")
                    continue
                try:
                    kind = evt.get("kind")
                    raw_label = evt.get("label")
                    if not raw_label:
                        raise Exception("missing 'label'")
                    label = prefix + str(raw_label)

                    if kind == "added":
                        item = dict(evt.get("item") or {})
                        if not item:
                            raise Exception("'added' event missing 'item'")
                        item["label"] = label
                        if label in self._state:
                            raise Exception(f"label {label!r} already exists")
                        self._install_item(item)
                        tf = self._state[label]["transform"]
                        self._broadcast(StreamTransformChangesResponse(
                            change_type=TransformChangeType.TRANSFORM_CHANGE_TYPE_ADDED,
                            transform=tf,
                        ))
                        added += 1

                    elif kind == "updated":
                        s = self._state.get(label)
                        if s is None:
                            raise Exception(f"unknown label {label!r}")
                        new_item = dict(evt.get("item") or {})
                        new_item["label"] = label
                        paths = list(evt.get("paths") or [])

                        if not paths:
                            # Empty paths means a metadata-only change
                            # (color / opacity / show_axes_helper /
                            # invisible). The renderer's UPDATED handler
                            # drops metadata.* paths, so a plain UPDATED
                            # would be a no-op at the viewer. Respawn:
                            # REMOVE the entity with its current UUID,
                            # then ADD it back with a fresh UUID so the
                            # renderer re-reads metadata at spawn.
                            old_tf = s["transform"]
                            self._broadcast(StreamTransformChangesResponse(
                                change_type=TransformChangeType.TRANSFORM_CHANGE_TYPE_REMOVED,
                                transform=old_tf,
                            ))
                            new_uuid = versioned_uuid(label)
                            s["item"] = new_item
                            s["base_pose"] = dict(new_item.get("pose") or {})
                            for k, default in (("x", 0.0), ("y", 0.0), ("z", 0.0),
                                               ("ox", 0.0), ("oy", 0.0), ("oz", 1.0),
                                               ("theta", 0.0)):
                                s["base_pose"].setdefault(k, default)
                            s["base_geom"] = self.base_geom_for_item(new_item)
                            geom_proto = self.build_geometry(new_item, s["base_geom"])
                            new_tf = self._build_transform(
                                new_item, s["base_pose"], geom_proto, new_uuid,
                                self.parent_frame,
                            )
                            s["uuid"] = new_uuid
                            s["transform"] = new_tf
                            self._broadcast(StreamTransformChangesResponse(
                                change_type=TransformChangeType.TRANSFORM_CHANGE_TYPE_ADDED,
                                transform=new_tf,
                            ))
                            updated += 1
                            continue

                        s["item"] = new_item
                        s["base_pose"] = dict(new_item.get("pose") or {})
                        for k, default in (("x", 0.0), ("y", 0.0), ("z", 0.0),
                                           ("ox", 0.0), ("oy", 0.0), ("oz", 1.0),
                                           ("theta", 0.0)):
                            s["base_pose"].setdefault(k, default)
                        s["base_geom"] = self.base_geom_for_item(new_item)
                        geom_proto = self.build_geometry(new_item, s["base_geom"])
                        new_tf = self._build_transform(
                            new_item, s["base_pose"], geom_proto, s["uuid"],
                            self.parent_frame,
                        )
                        s["transform"] = new_tf
                        self._broadcast(StreamTransformChangesResponse(
                            change_type=TransformChangeType.TRANSFORM_CHANGE_TYPE_UPDATED,
                            transform=new_tf,
                            updated_fields=FieldMask(paths=paths),
                        ))
                        updated += 1

                    elif kind == "removed":
                        tf = self._remove_item(label)
                        if tf is not None:
                            self._broadcast(StreamTransformChangesResponse(
                                change_type=TransformChangeType.TRANSFORM_CHANGE_TYPE_REMOVED,
                                transform=tf,
                            ))
                            removed += 1

                    else:
                        raise Exception(f"unknown kind {kind!r}")
                except Exception as e:
                    errors.append(f"event[{i}] ({evt.get('label')!r}): {e}")
        return {
            "applied": added + updated + removed,
            "added": added,
            "updated": updated,
            "removed": removed,
            "errors": errors,
        }

    def _maybe_restart_tick(self) -> None:
        wants_tick = self._has_scene_tick() or any(
            self.is_animated(s["item"]) for s in self._state.values()
        )
        if wants_tick and (self._tick_task is None or self._tick_task.done()):
            self._animation_t0 = time.monotonic()
            try:
                self._tick_task = asyncio.create_task(self._tick_loop())
            except RuntimeError:
                self._tick_task = None

    def _debug_snapshot(self) -> Mapping[str, ValueTypes]:
        return {
            "tick_hz": self.tick_hz,
            "uuid_strategy": self.uuid_strategy,
            "parent_frame": self.parent_frame,
            "item_count": len(self._state),
            "subscriber_count": len(self._subscribers),
            "tick_running": (
                self._tick_task is not None and not self._tick_task.done()
            ),
            "animation_t0": self._animation_t0,
        }
