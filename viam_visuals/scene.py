"""Scene — typed state container with object-based mutation API.

The plan's "step 2" surface: a class that holds :class:`Visual`
instances by label, tracks the wire-format dict committed for each
one, and produces field-mask deltas when you mutate the object and
call :meth:`Scene.update`. The typical loop:

    from viam_visuals import Scene, BoundingBox, Pose, Spin

    scene = Scene(parent_frame="world")

    bbox = BoundingBox("obj_a", dims_mm=(100, 200, 50), color=(255, 0, 0))
    scene.add(bbox)

    # ...time passes, detection moves...
    bbox.pose = Pose.at(x=500, y=-200, z=100)
    bbox.color = (0, 255, 0)
    events = scene.update(bbox)
    # events == [SceneEvent(kind="updated", label="obj_a",
    #                       paths=["poseInObserverFrame.pose.x", ...,
    #                              "metadata.colors"])]

The diff is **state-based**, not patch-based: ``Scene`` snapshots the
visual's wire-format dict at ``add`` time and re-snapshots after
each ``update``. Field-mask paths come from comparing those
snapshots, so callers can mutate any subset of the object's fields
without specifying which.

This module deliberately doesn't broadcast anywhere; it produces
:class:`SceneEvent` records the caller (or a wrapping service)
consumes. A future revision of ``SceneServiceBase`` can hold a
``Scene`` internally and forward events to its subscribers, but the
class works standalone for tests and for callers writing their own
service plumbing.

Composites (CoordinateFrame, Line, BoundingBox-wireframe) are
expanded at ``add`` time — the scene tracks the constituent Visuals,
not the composite object. Mutating the composite and calling
``scene.update(frame)`` works because the composite re-expands via
``__iter__`` / ``to_visuals``; the scene diffs each constituent
against its prior snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Union

from .composites import Composite
from .shapes import Visual


__all__ = ["Scene", "SceneEvent", "events_to_wire"]


# ---- Event kinds + records ---------------------------------------------

ADDED = "added"
UPDATED = "updated"
REMOVED = "removed"


@dataclass
class SceneEvent:
    """One state-change record produced by Scene mutation methods.

    ``kind`` is one of ``"added"`` / ``"updated"`` / ``"removed"``.
    ``label`` identifies the visual. ``item_dict`` is the wire-format
    dict for ADDED/UPDATED (the current state); empty for REMOVED.
    ``paths`` is the list of field-mask paths for UPDATED events
    (always camelCase; the renderer ignores snake_case).
    """

    kind: str
    label: str
    item_dict: Mapping[str, Any] = field(default_factory=dict)
    paths: List[str] = field(default_factory=list)


@dataclass
class SceneEntry:
    """One row of scene state — the live object reference plus the
    last-committed wire-format dict used for diffing."""

    visual: Visual
    committed: Mapping[str, Any]


# ---- Field-name → field-mask path mapping ------------------------------
#
# Wire-format paths are camelCase. The viewer ignores snake_case
# paths — see LESSONS.md::snake-case-field-mask-paths-do-not-work.
#
# The renderer's ``updateEntity`` handler at
# ``@viamrobotics/motion-tools::useWorldState.svelte.ts`` honors only
# two path prefixes on UPDATED events:
#
#   * ``poseInObserverFrame.pose*`` — re-reads pose, updates the
#     entity's Pose trait.
#   * ``physicalObject*`` — re-reads geometryType.value, dispatches
#     to traits.Box / Capsule / Sphere / mesh-BufferGeometry. There
#     is no ``case === 'pointcloud'`` branch — pointcloud geometry
#     updates aren't propagated via UPDATED today.
#
# ALL ``metadata.*`` paths (color, colors, opacity, show_axes_helper,
# invisible) are dropped silently. Metadata changes propagate only
# at spawn time. To force a metadata refresh on the renderer, the
# entity must be REMOVED then re-ADDED with a fresh UUID (lifecycle-
# style label rotation, or the versioned UUID strategy on the
# visualizer).
#
# See ``LESSONS.md::renderer-honors-only-pose-and-physicalobject-on-updated``.

# ---- Public field-mask path constants ----------------------------------
#
# Subclasses writing animation hooks (the legacy ``compute_tick``
# path) need these strings to populate ``updated_fields``. They're
# also the single source of truth Scene's diff machinery uses.
#
# Importable as ``viam_visuals.PATH_X`` etc., or as dicts via
# ``viam_visuals.POSE_PATHS`` / ``GEOM_PATHS`` / ``METADATA_PATHS``.

# Pose component paths. All share the renderer's
# ``poseInObserverFrame.pose`` prefix, which triggers a full Pose
# re-read; emitting per-subfield paths is informational, but any one
# of the right prefix is sufficient.
PATH_X = "poseInObserverFrame.pose.x"
PATH_Y = "poseInObserverFrame.pose.y"
PATH_Z = "poseInObserverFrame.pose.z"
PATH_OX = "poseInObserverFrame.pose.oX"
PATH_OY = "poseInObserverFrame.pose.oY"
PATH_OZ = "poseInObserverFrame.pose.oZ"
PATH_THETA = "poseInObserverFrame.pose.theta"

# Geometry-trait paths. physicalObject.* paths re-read the geometry;
# the renderer rebuilds the appropriate trait (Box / Sphere /
# Capsule / Mesh / PointCloud).
PATH_SPHERE_RADIUS = "physicalObject.geometryType.value.radiusMm"
PATH_CAPSULE_RADIUS = "physicalObject.geometryType.value.radiusMm"
PATH_CAPSULE_LENGTH = "physicalObject.geometryType.value.lengthMm"
PATH_BOX_DIMS_X = "physicalObject.geometryType.value.dimsMm.x"
PATH_BOX_DIMS_Y = "physicalObject.geometryType.value.dimsMm.y"
PATH_BOX_DIMS_Z = "physicalObject.geometryType.value.dimsMm.z"
PATH_MESH = "physicalObject.mesh"

# Metadata paths the renderer DROPS on UPDATED events. Listed here
# for completeness — animating a value at one of these requires the
# REMOVE + re-ADD pattern (Scene + SceneServiceBase do this
# automatically via the metadata-only-update intercept).
PATH_METADATA_COLOR = "metadata.color"
PATH_METADATA_OPACITY = "metadata.opacity"
PATH_METADATA_SHOW_AXES = "metadata.show_axes_helper"
PATH_METADATA_INVISIBLE = "metadata.invisible"

POSE_PATHS: Dict[str, str] = {
    "x": PATH_X, "y": PATH_Y, "z": PATH_Z,
    "ox": PATH_OX, "oy": PATH_OY, "oz": PATH_OZ,
    "theta": PATH_THETA,
}

GEOM_PATHS: Dict[str, str] = {
    "radius_mm": PATH_SPHERE_RADIUS,  # also matches capsule
    "length_mm": PATH_CAPSULE_LENGTH,
}

METADATA_PATHS: Dict[str, str] = {
    "color": PATH_METADATA_COLOR,
    "opacity": PATH_METADATA_OPACITY,
    "show_axes_helper": PATH_METADATA_SHOW_AXES,
    "invisible": PATH_METADATA_INVISIBLE,
}

# Legacy aliases (private). Kept until callers in this file migrate.
_POSE_KEY_TO_PATH = POSE_PATHS
_TOPLEVEL_KEY_TO_PATH = GEOM_PATHS


# ---- Scene -------------------------------------------------------------


class Scene:
    """Typed scene state with object-based add / update / remove.

    A :class:`Scene` is a mapping from label to :class:`Visual` (or
    a constituent of an expanded :class:`Composite`). Calls to
    :meth:`add` / :meth:`update` / :meth:`remove` return
    :class:`SceneEvent` lists that downstream service plumbing can
    forward to WSS subscribers.
    """

    def __init__(self, parent_frame: str = "world") -> None:
        self._parent_frame = parent_frame
        self._state: Dict[str, SceneEntry] = {}

    # ---- introspection -------------------------------------------------

    @property
    def parent_frame(self) -> str:
        return self._parent_frame

    def __len__(self) -> int:
        return len(self._state)

    def __contains__(self, target: Union[str, Visual, Composite]) -> bool:
        return self._label_of(target) in self._state

    def __iter__(self) -> Iterator[Visual]:
        return iter(e.visual for e in self._state.values())

    def labels(self) -> Sequence[str]:
        """Return all current labels, sorted."""
        return sorted(self._state)

    def get(self, label: str) -> Optional[Visual]:
        """Return the live Visual for ``label``, or ``None``."""
        e = self._state.get(label)
        return e.visual if e is not None else None

    # ---- mutation ------------------------------------------------------

    def add(self, *targets: Union[Visual, Composite]) -> List[SceneEvent]:
        """Add one or more visuals. Composites expand into their
        constituent Visuals; each constituent gets its own ADDED
        event. Returns the list of events in add order.

        Raises :class:`ValueError` if any label already exists.
        Adding fails atomically — partial inserts are rolled back.
        """
        flat = _flatten(targets)
        # Pre-check for duplicates so we don't half-add.
        for v in flat:
            if v.label in self._state:
                raise ValueError(f"duplicate label {v.label!r}")
        out: List[SceneEvent] = []
        for v in flat:
            d = v.to_dict()
            self._state[v.label] = SceneEntry(visual=v, committed=d)
            out.append(SceneEvent(kind=ADDED, label=v.label, item_dict=d))
        return out

    def update(self, *targets: Union[Visual, Composite]) -> List[SceneEvent]:
        """Diff each target against its committed snapshot and return
        UPDATED events for the changed visuals. Composites expand;
        each constituent diffs independently.

        Visuals that haven't changed at all produce no event. Visuals
        whose only changes are to metadata fields (color, opacity,
        ``show_axes_helper``, ``invisible``) produce an UPDATED event
        with ``paths=[]`` — the signal to consumers that a renderer
        respawn (REMOVE + re-ADD with a fresh UUID) is required. The
        renderer's UPDATED handler drops ``metadata.*`` paths, so a
        plain UPDATED with metadata paths would be a no-op at the
        viewer; ``SceneServiceBase._apply_scene_event`` and the
        ``apply_events`` DoCommand handler both translate the
        empty-paths UPDATED into a REMOVE + re-ADD on the wire.

        Raises :class:`ValueError` if any label isn't in the scene.
        """
        flat = _flatten(targets)
        # Pre-check membership so partial updates can't leave the
        # caller guessing which targets succeeded.
        missing = [v.label for v in flat if v.label not in self._state]
        if missing:
            raise ValueError(f"unknown label(s): {missing}")
        out: List[SceneEvent] = []
        for v in flat:
            entry = self._state[v.label]
            new_dict = v.to_dict()
            if new_dict == entry.committed:
                continue  # No change at all.
            paths = _diff_paths(entry.committed, new_dict)
            # Respawn (empty Paths) wins over UPDATED when both kinds
            # of change happen simultaneously. The renderer drops
            # metadata.* / parent_frame on UPDATED, so emitting paths
            # here would visibly lose the metadata change. The
            # consumer-side respawn carries the new pose + new
            # geometry + new metadata in a single REMOVE + re-ADD,
            # losing nothing. Cost: a respawn is heavier on the wire
            # than an UPDATED, so callers mutating metadata at high
            # tick rates should snap to discrete steps (see
            # ``viam_visuals.snap_step``).
            if _requires_respawn(entry.committed, new_dict):
                paths = []
            entry.committed = new_dict
            entry.visual = v
            out.append(SceneEvent(
                kind=UPDATED, label=v.label,
                item_dict=new_dict, paths=paths,
            ))
        return out

    def add_or_update(self, *targets: Union[Visual, Composite]) -> List[SceneEvent]:
        """Upsert — ADD any visuals not currently in the scene, UPDATE
        any that exist (returning the diff event only if something
        changed). Useful for tick loops that produce a fresh visual
        list each frame without tracking the lifecycle themselves.
        """
        flat = _flatten(targets)
        out: List[SceneEvent] = []
        for v in flat:
            if v.label in self._state:
                out.extend(self.update(v))
            else:
                out.extend(self.add(v))
        return out

    def remove(self, *targets: Union[str, Visual, Composite]) -> List[SceneEvent]:
        """Remove one or more visuals by label or by object. Composite
        objects expand and remove each constituent. Visuals not in
        the scene are skipped silently — the call is idempotent.
        Returns REMOVED events for the visuals actually removed.
        """
        labels = _flatten_labels(targets)
        out: List[SceneEvent] = []
        for label in labels:
            if label in self._state:
                del self._state[label]
                out.append(SceneEvent(kind=REMOVED, label=label))
        return out

    def clear(self) -> List[SceneEvent]:
        """Remove every visual from the scene. Returns REMOVED events
        for everything that was in the scene, in label order."""
        out = [SceneEvent(kind=REMOVED, label=lab) for lab in sorted(self._state)]
        self._state = {}
        return out

    # ---- internals -----------------------------------------------------

    @staticmethod
    def _label_of(target: Union[str, Visual, Composite]) -> str:
        if isinstance(target, str):
            return target
        if isinstance(target, Visual):
            return target.label
        if isinstance(target, Composite):
            # Composites are addressed by their root visual's label
            # only when checking membership of the composite itself;
            # for actual operations, use _flatten_labels.
            visuals = target.to_visuals()
            return visuals[0].label if visuals else ""
        raise TypeError(
            f"expected str | Visual | Composite, got {type(target).__name__}"
        )


# ---- helpers -----------------------------------------------------------

def _flatten(
    targets: Sequence[Union[Visual, Composite]],
) -> List[Visual]:
    """Expand composites into their constituent Visuals. Plain
    Visuals pass through unchanged."""
    out: List[Visual] = []
    for t in targets:
        if isinstance(t, Composite):
            out.extend(t.to_visuals())
        elif isinstance(t, Visual):
            out.append(t)
        else:
            raise TypeError(
                f"expected Visual | Composite, got {type(t).__name__}"
            )
    return out


def _flatten_labels(
    targets: Sequence[Union[str, Visual, Composite]],
) -> List[str]:
    """Same as :func:`_flatten` but yields labels and accepts plain
    string labels too. Used by :meth:`Scene.remove`."""
    out: List[str] = []
    for t in targets:
        if isinstance(t, str):
            out.append(t)
        elif isinstance(t, Composite):
            for v in t.to_visuals():
                out.append(v.label)
        elif isinstance(t, Visual):
            out.append(t.label)
        else:
            raise TypeError(
                f"expected str | Visual | Composite, got {type(t).__name__}"
            )
    return out


def events_to_wire(events: Sequence[SceneEvent]) -> List[Dict[str, Any]]:
    """Serialize a list of :class:`SceneEvent` records to the dict form
    the ``apply_events`` DoCommand verb accepts.

    Use this on the driver side to ship a batch of Scene mutations to
    a visualizer::

        events = scene.update(box, sphere)
        await visualizer.do_command({
            "command": "apply_events",
            "events": events_to_wire(events),
        })
    """
    out: List[Dict[str, Any]] = []
    for e in events:
        rec: Dict[str, Any] = {"kind": e.kind, "label": e.label}
        if e.item_dict:
            rec["item"] = dict(e.item_dict)
        if e.paths:
            rec["paths"] = list(e.paths)
        out.append(rec)
    return out


def _requires_respawn(old: Mapping[str, Any], new: Mapping[str, Any]) -> bool:
    """True if the change between two wire-format item dicts touches
    any field the renderer reads only at spawn time (not on UPDATED):
    color, opacity, show_axes_helper, invisible, parent_frame.

    Used by :meth:`Scene.update` to escalate an UPDATED into the
    respawn signal (empty ``paths``) — losing those metadata
    changes by emitting a renderer-honored UPDATED instead would be
    a silent bug.
    """
    for k in ("color", "opacity", "show_axes_helper", "invisible", "parent_frame"):
        if old.get(k) != new.get(k):
            return True
    return False


def _diff_paths(
    old: Mapping[str, Any], new: Mapping[str, Any],
) -> List[str]:
    """Compute the field-mask path list describing what changed
    between two wire-format item dicts.

    Only emits paths the renderer honors on UPDATED events — see the
    module-level note on the path-matcher behavior. The library's
    state (``Scene._state``) tracks every change; only the wire-form
    paths emitted from here are limited to what the renderer reads.

    Metadata-only changes (color, opacity, show_axes_helper,
    invisible) produce an empty list. The visual's snapshot is still
    updated server-side, but no UPDATED event reaches the renderer;
    a re-spawn (REMOVED + ADDED with a fresh UUID) is required for
    the renderer to see metadata changes.
    """
    paths: List[str] = []
    # Pose: per-subfield diff. The renderer's check is
    # ``path.startsWith('poseInObserverFrame.pose')`` and re-reads
    # the full pose, so emitting per-axis paths is informational —
    # any one of them triggers a full Pose re-read.
    old_pose = old.get("pose") or {}
    new_pose = new.get("pose") or {}
    if old_pose != new_pose:
        for k, p in _POSE_KEY_TO_PATH.items():
            if old_pose.get(k) != new_pose.get(k):
                paths.append(p)
    # Top-level scalar fields the renderer rebuilds via physicalObject.*.
    for k, p in _TOPLEVEL_KEY_TO_PATH.items():
        if old.get(k) != new.get(k):
            paths.append(p)
    # Box dims_mm: per-axis diff. The renderer reads the full Box
    # geometry on any physicalObject* path.
    old_dims = old.get("dims_mm") or {}
    new_dims = new.get("dims_mm") or {}
    if old_dims != new_dims:
        for axis in ("x", "y", "z"):
            if old_dims.get(axis) != new_dims.get(axis):
                paths.append(
                    f"physicalObject.geometryType.value.dimsMm.{axis}"
                )
    # Mesh path swap: renderer re-parses the PLY and sets
    # traits.BufferGeometry.
    if old.get("mesh_path") != new.get("mesh_path"):
        paths.append("physicalObject.mesh")
    # NOTE: pointcloud_path swaps would emit physicalObject.pointcloud,
    # but the renderer's updateEntity has no ``case === 'pointcloud'``
    # branch — pointcloud updates would silently no-op. Re-spawn the
    # entity (REMOVED + ADDED with a fresh label) to update a pcd.
    return paths
