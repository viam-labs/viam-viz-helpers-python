"""Tests for viam_visuals.Scene — object-based mutation API."""

from __future__ import annotations

import pytest

from viam_visuals import (
    ADDED,
    REMOVED,
    UPDATED,
    BoundingBox,
    Box,
    Capsule,
    CoordinateFrame,
    Pose,
    Scene,
    Sphere,
)


# ---- add ---------------------------------------------------------------

def test_add_single_visual_emits_added_event():
    s = Scene()
    b = Box("b", dims_mm=(100, 200, 50))
    events = s.add(b)
    assert len(events) == 1
    assert events[0].kind == ADDED
    assert events[0].label == "b"
    assert events[0].item_dict["type"] == "box"
    assert events[0].paths == []


def test_add_multiple_visuals_emits_one_event_each():
    s = Scene()
    events = s.add(
        Box("b1", dims_mm=(100, 100, 100)),
        Sphere("s1", radius_mm=50),
        Capsule("c1", radius_mm=20, length_mm=200),
    )
    assert [e.kind for e in events] == [ADDED, ADDED, ADDED]
    assert [e.label for e in events] == ["b1", "s1", "c1"]
    assert len(s) == 3


def test_add_duplicate_label_raises():
    s = Scene()
    s.add(Box("dup", dims_mm=(1, 1, 1)))
    with pytest.raises(ValueError, match="duplicate label"):
        s.add(Sphere("dup", radius_mm=10))


def test_add_atomic_on_duplicate_in_batch():
    """If any visual in a batch collides, none are added."""
    s = Scene()
    s.add(Box("a", dims_mm=(1, 1, 1)))
    with pytest.raises(ValueError):
        s.add(Sphere("b", radius_mm=1), Box("a", dims_mm=(2, 2, 2)))
    # 'b' must not have leaked in.
    assert s.labels() == ["a"]


def test_add_composite_expands_into_constituents():
    s = Scene()
    frame = CoordinateFrame("frame")
    events = s.add(frame)
    # CoordinateFrame is anchor sphere + 3 axis capsules.
    assert len(events) == 4
    assert {e.kind for e in events} == {ADDED}
    assert set(s.labels()) == {
        "frame", "frame_axis_x", "frame_axis_y", "frame_axis_z",
    }


# ---- update ------------------------------------------------------------

def test_update_pose_emits_per_axis_paths():
    s = Scene()
    b = Box("b", dims_mm=(100, 100, 100))
    s.add(b)
    b.pose = Pose.at(x=200, y=-50)
    events = s.update(b)
    assert len(events) == 1
    assert events[0].kind == UPDATED
    assert "poseInObserverFrame.pose.x" in events[0].paths
    assert "poseInObserverFrame.pose.y" in events[0].paths
    # z and theta didn't change.
    assert "poseInObserverFrame.pose.z" not in events[0].paths
    assert "poseInObserverFrame.pose.theta" not in events[0].paths


def test_update_metadata_only_change_yields_updated_with_empty_paths():
    # Color / opacity changes are emitted as UPDATED events with
    # paths=[] — the signal that the consumer must respawn (REMOVE
    # + re-ADD with a fresh UUID) since the renderer drops
    # metadata.* paths on UPDATED. The committed snapshot still
    # reflects the new values.
    s = Scene()
    b = Box("b", dims_mm=(100, 100, 100), color=(255, 0, 0), opacity=1.0)
    s.add(b)
    b.color = (0, 255, 0)
    b.opacity = 0.5
    events = s.update(b)
    assert len(events) == 1
    assert events[0].kind == "updated"
    assert events[0].label == "b"
    assert events[0].paths == []
    assert events[0].item_dict["color"] == {"r": 0, "g": 255, "b": 0}


def test_update_no_change_yields_no_event():
    s = Scene()
    b = Box("b", dims_mm=(100, 100, 100), color=(255, 0, 0))
    s.add(b)
    events = s.update(b)  # no mutations
    assert events == []


def test_update_parent_frame_change_yields_respawn_signal():
    # parent_frame is not honored on UPDATED — the renderer reads
    # PoseInObserverFrame.reference_frame at spawn time. Treat it
    # like a metadata change: emit UPDATED with empty paths so the
    # service-side respawn intercept (REMOVE + re-ADD with fresh
    # UUID) re-anchors the entity at the renderer.
    s = Scene()
    b = Box("b", dims_mm=(100, 100, 100))
    s.add(b)
    b.parent_frame = "new_parent"
    events = s.update(b)
    assert len(events) == 1
    assert events[0].kind == "updated"
    assert events[0].paths == []
    assert events[0].item_dict["parent_frame"] == "new_parent"


def test_update_no_changes_returns_empty_list():
    s = Scene()
    b = Box("b", dims_mm=(100, 100, 100))
    s.add(b)
    assert s.update(b) == []


def test_update_sphere_radius_emits_radius_path():
    s = Scene()
    sp = Sphere("sp", radius_mm=50)
    s.add(sp)
    sp.radius_mm = 80
    events = s.update(sp)
    assert events[0].paths == ["physicalObject.geometryType.value.radiusMm"]


def test_update_capsule_length_and_radius():
    s = Scene()
    c = Capsule("c", radius_mm=20, length_mm=100)
    s.add(c)
    c.radius_mm = 25
    c.length_mm = 150
    events = s.update(c)
    paths = set(events[0].paths)
    assert "physicalObject.geometryType.value.radiusMm" in paths
    assert "physicalObject.geometryType.value.lengthMm" in paths


def test_update_box_dims_emits_per_axis_paths():
    s = Scene()
    b = Box("b", dims_mm=(100, 100, 100))
    s.add(b)
    b.dims_mm = (200, 100, 100)  # only x changed
    events = s.update(b)
    assert events[0].paths == [
        "physicalObject.geometryType.value.dimsMm.x",
    ]


def test_update_unknown_label_raises():
    s = Scene()
    b = Box("never_added", dims_mm=(1, 1, 1))
    with pytest.raises(ValueError, match="unknown label"):
        s.update(b)


def test_update_recommits_snapshot_so_second_update_diffs_against_new_state():
    """After update, the scene's committed snapshot is the post-update
    state — a second identical mutation produces no event."""
    s = Scene()
    b = Box("b", dims_mm=(100, 100, 100))
    s.add(b)
    b.pose = Pose.at(x=200)
    first = s.update(b)
    assert len(first) == 1
    # Re-running update with no further mutation: no diff.
    assert s.update(b) == []


# ---- add_or_update -----------------------------------------------------

def test_add_or_update_adds_new_visuals():
    s = Scene()
    b = Box("b", dims_mm=(100, 100, 100))
    events = s.add_or_update(b)
    assert [e.kind for e in events] == [ADDED]


def test_add_or_update_updates_existing_visuals():
    s = Scene()
    b = Box("b", dims_mm=(100, 100, 100))
    s.add(b)
    b.pose = Pose.at(x=50)
    events = s.add_or_update(b)
    assert [e.kind for e in events] == [UPDATED]


def test_add_or_update_mixed_batch():
    s = Scene()
    existing = Box("existing", dims_mm=(100, 100, 100))
    s.add(existing)
    existing.pose = Pose.at(x=200)
    new_one = Sphere("new", radius_mm=50)
    events = s.add_or_update(existing, new_one)
    kinds = [e.kind for e in events]
    assert UPDATED in kinds
    assert ADDED in kinds


# ---- remove ------------------------------------------------------------

def test_remove_by_object():
    s = Scene()
    b = Box("b", dims_mm=(1, 1, 1))
    s.add(b)
    events = s.remove(b)
    assert [e.kind for e in events] == [REMOVED]
    assert events[0].label == "b"
    assert len(s) == 0


def test_remove_by_label_string():
    s = Scene()
    s.add(Box("b", dims_mm=(1, 1, 1)))
    events = s.remove("b")
    assert [e.kind for e in events] == [REMOVED]


def test_remove_unknown_label_is_silent():
    s = Scene()
    s.add(Box("a", dims_mm=(1, 1, 1)))
    # Removing a label that isn't there → no event, no error.
    assert s.remove("does_not_exist") == []
    assert s.labels() == ["a"]


def test_remove_composite_removes_all_constituents():
    s = Scene()
    frame = CoordinateFrame("frame")
    s.add(frame)
    events = s.remove(frame)
    assert len(events) == 4
    assert {e.kind for e in events} == {REMOVED}
    assert len(s) == 0


def test_clear_removes_everything():
    s = Scene()
    s.add(Box("b1", dims_mm=(1, 1, 1)))
    s.add(Sphere("s1", radius_mm=10))
    events = s.clear()
    assert len(events) == 2
    assert {e.kind for e in events} == {REMOVED}
    assert len(s) == 0


# ---- introspection -----------------------------------------------------

def test_get_returns_live_object_reference():
    """The Visual returned by get() is the exact same instance the
    caller added — mutating it and calling update() works."""
    s = Scene()
    b = Box("b", dims_mm=(100, 100, 100))
    s.add(b)
    retrieved = s.get("b")
    assert retrieved is b
    retrieved.pose = Pose.at(x=100)
    events = s.update(b)
    assert len(events) == 1


def test_get_missing_returns_none():
    s = Scene()
    assert s.get("nope") is None


def test_contains_checks_label():
    s = Scene()
    b = Box("b", dims_mm=(1, 1, 1))
    s.add(b)
    assert "b" in s
    assert b in s
    assert "missing" not in s


def test_labels_returns_sorted():
    s = Scene()
    s.add(
        Box("zeta", dims_mm=(1, 1, 1)),
        Sphere("alpha", radius_mm=1),
        Capsule("mu", radius_mm=1, length_mm=10),
    )
    assert s.labels() == ["alpha", "mu", "zeta"]


def test_parent_frame_default():
    s = Scene()
    assert s.parent_frame == "world"


def test_parent_frame_custom():
    s = Scene(parent_frame="robot_arm:eoa")
    assert s.parent_frame == "robot_arm:eoa"


# ---- BoundingBox round-trip --------------------------------------------

def test_bounding_box_updates_pose_via_object_mutation():
    """The headline use case — bbox.pose = new_pose; scene.update(bbox).

    When pose AND metadata (color) both change, Scene escalates to
    the respawn signal (paths=[]) — emitting an UPDATED with the
    pose paths would visibly lose the color change at the renderer.
    The consumer-side respawn (REMOVE + re-ADD with fresh UUID)
    carries both the new pose and the new color in one go.
    """
    s = Scene()
    bbox = BoundingBox("obj_a", dims_mm=(100, 200, 50), color=(255, 0, 0))
    s.add(bbox)

    bbox.pose = Pose.at(x=500, y=-200, z=100)
    bbox.color = (0, 255, 0)
    events = s.update(bbox)

    assert len(events) == 1
    assert events[0].kind == UPDATED
    assert events[0].label == "obj_a"
    # Respawn signal: empty paths. The committed snapshot reflects
    # both changes; the SceneServiceBase intercept materializes them
    # as REMOVE + re-ADD with a fresh UUID at the renderer.
    assert events[0].paths == []
    assert events[0].item_dict["pose"]["x"] == 500
    assert events[0].item_dict["color"] == {"r": 0, "g": 255, "b": 0}


def test_bounding_box_pose_only_update_emits_pose_paths():
    """When only pose changes (no metadata touched), Scene emits a
    standard UPDATED with pose paths — no respawn needed."""
    s = Scene()
    bbox = BoundingBox("obj_b", dims_mm=(100, 200, 50), color=(255, 0, 0))
    s.add(bbox)

    bbox.pose = Pose.at(x=500, y=-200, z=100)
    events = s.update(bbox)

    assert len(events) == 1
    assert events[0].kind == UPDATED
    paths = set(events[0].paths)
    assert "poseInObserverFrame.pose.x" in paths
    assert "poseInObserverFrame.pose.y" in paths
    assert "poseInObserverFrame.pose.z" in paths
