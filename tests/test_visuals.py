"""Tests for the ``viam_visuals`` library — the OO API.

Covers the contract that ``Visual.to_dict()`` produces dicts
the service layer accepts unchanged, plus construction-time
validation rejects bad params before they reach the wire."""

import pytest

from viam_visuals import (
    Arrow,
    Box,
    Capsule,
    Mesh,
    Point,
    PointCloud,
    Pose,
    Sphere,
    Visual,
)


# ---------- Pose ---------------------------------------------------------

def test_pose_identity_is_default():
    p = Pose.identity()
    assert p.x == 0 and p.y == 0 and p.z == 0
    assert p.ox == 0 and p.oy == 0 and p.oz == 1.0
    assert p.theta == 0


def test_pose_at_sets_position():
    p = Pose.at(x=100, y=-50, z=25)
    d = p.to_dict()
    assert d == {"x": 100, "y": -50, "z": 25,
                 "ox": 0, "oy": 0, "oz": 1.0, "theta": 0}


def test_pose_dataclass_full_constructor():
    p = Pose(x=1, y=2, z=3, ox=0.7, oy=0.7, oz=0, theta=1.5)
    d = p.to_dict()
    assert d["ox"] == 0.7 and d["oy"] == 0.7 and d["oz"] == 0
    assert d["theta"] == 1.5


# ---------- Box ----------------------------------------------------------

def test_box_dict_matches_hand_written_shape():
    b = Box("demo_box", pose=Pose.at(x=-1600),
            dims_mm=(150, 150, 150), color=(230, 25, 75), opacity=1.0)
    d = b.to_dict()
    # Field-by-field — order doesn't matter for dict equality.
    assert d["type"] == "box"
    assert d["label"] == "demo_box"
    assert d["pose"] == {"x": -1600, "y": 0, "z": 0,
                         "ox": 0, "oy": 0, "oz": 1.0, "theta": 0}
    assert d["dims_mm"] == {"x": 150.0, "y": 150.0, "z": 150.0}
    assert d["color"] == {"r": 230, "g": 25, "b": 75}
    assert d["opacity"] == 1.0
    assert d["animation"] == {"mode": "none"}


def test_box_rejects_non_positive_dims():
    with pytest.raises(ValueError):
        Box("x", dims_mm=(150, 0, 150))
    with pytest.raises(ValueError):
        Box("x", dims_mm=(150, -1, 150))


def test_box_rejects_wrong_dims_arity():
    with pytest.raises(ValueError):
        Box("x", dims_mm=(150, 150))  # only 2 components


# ---------- Sphere -------------------------------------------------------

def test_sphere_dict_matches():
    s = Sphere("demo_sphere", radius_mm=90, color=(60, 180, 75), opacity=1.0)
    d = s.to_dict()
    assert d["type"] == "sphere"
    assert d["radius_mm"] == 90.0
    assert d["color"] == {"r": 60, "g": 180, "b": 75}


def test_sphere_rejects_non_positive_radius():
    with pytest.raises(ValueError):
        Sphere("x", radius_mm=0)


# ---------- Capsule ------------------------------------------------------

def test_capsule_dict_matches():
    c = Capsule("demo_capsule", radius_mm=50, length_mm=200,
                color=(0, 130, 200), opacity=1.0)
    d = c.to_dict()
    assert d["type"] == "capsule"
    assert d["radius_mm"] == 50.0
    assert d["length_mm"] == 200.0


def test_capsule_rejects_non_positive_dims():
    with pytest.raises(ValueError):
        Capsule("x", radius_mm=50, length_mm=0)
    with pytest.raises(ValueError):
        Capsule("x", radius_mm=0, length_mm=200)


# ---------- Point --------------------------------------------------------

def test_point_dict_has_no_shape_fields():
    p = Point("demo_point", color=(255, 225, 25), opacity=1.0)
    d = p.to_dict()
    assert d["type"] == "point"
    assert "radius_mm" not in d
    assert "dims_mm" not in d


# ---------- Arrow --------------------------------------------------------

def test_arrow_dict_matches():
    a = Arrow("demo_arrow", length_mm=220, radius_mm=12,
              color=(145, 30, 180), opacity=1.0)
    d = a.to_dict()
    assert d["type"] == "arrow"
    assert d["length_mm"] == 220.0
    assert d["radius_mm"] == 12.0


def test_arrow_rejects_non_positive_dims():
    with pytest.raises(ValueError):
        Arrow("x", length_mm=220, radius_mm=0)
    with pytest.raises(ValueError):
        Arrow("x", length_mm=0, radius_mm=12)


# ---------- Mesh ---------------------------------------------------------

def test_mesh_basic():
    m = Mesh("demo_bunny", mesh_path="assets/bunny.stl",
             color=(245, 130, 49), opacity=1.0)
    d = m.to_dict()
    assert d["type"] == "mesh"
    assert d["mesh_path"] == "assets/bunny.stl"
    assert "raw_stl" not in d


def test_mesh_raw_stl_flag_propagates():
    m = Mesh("demo_bunny_raw_stl", mesh_path="assets/bunny.stl",
            raw_stl=True, color=(245, 130, 49), opacity=1.0)
    d = m.to_dict()
    assert d["raw_stl"] is True


def test_mesh_requires_path():
    with pytest.raises(ValueError):
        Mesh("x", mesh_path="")


# ---------- PointCloud ---------------------------------------------------

def test_pointcloud_basic():
    pc = PointCloud("demo_pointcloud", pointcloud_path="assets/helix.pcd",
                    opacity=1.0)
    d = pc.to_dict()
    assert d["type"] == "pointcloud"
    assert d["pointcloud_path"] == "assets/helix.pcd"
    assert "chunked" not in d
    assert "chunk_size" not in d
    assert "color" not in d  # absence preserved when None


def test_pointcloud_chunked_flags_propagate():
    pc = PointCloud("demo_chunked", pointcloud_path="assets/helix.pcd",
                    chunked=True, chunk_size=2000, opacity=1.0)
    d = pc.to_dict()
    assert d["chunked"] is True
    assert d["chunk_size"] == 2000


def test_pointcloud_rejects_bad_chunk_size():
    with pytest.raises(ValueError):
        PointCloud("x", pointcloud_path="x", chunk_size=0)


# ---------- Cross-cutting -------------------------------------------------

def test_opacity_out_of_range_rejected():
    b = Box("x", dims_mm=(1, 1, 1), opacity=1.5)
    with pytest.raises(ValueError):
        b.to_dict()


def test_color_tuple_or_dict_both_accepted():
    b1 = Box("a", dims_mm=(1, 1, 1), color=(230, 25, 75))
    b2 = Box("b", dims_mm=(1, 1, 1), color={"r": 230, "g": 25, "b": 75})
    assert b1.to_dict()["color"] == b2.to_dict()["color"]


def test_animation_passthrough_when_set():
    s = Sphere("x", radius_mm=10,
               animation={"mode": "spin", "rpm": 15})
    d = s.to_dict()
    assert d["animation"] == {"mode": "spin", "rpm": 15}


def test_animation_defaults_to_none_mode_when_unset():
    s = Sphere("x", radius_mm=10)
    d = s.to_dict()
    assert d["animation"] == {"mode": "none"}


def test_parent_frame_propagates_when_set():
    s = Sphere("x", radius_mm=10, parent_frame="anchor")
    d = s.to_dict()
    assert d["parent_frame"] == "anchor"


def test_show_axes_helper_propagates():
    s = Sphere("x", radius_mm=10, show_axes_helper=True)
    d = s.to_dict()
    assert d["show_axes_helper"] is True


def test_invisible_propagates():
    s = Sphere("x", radius_mm=10, invisible=True)
    d = s.to_dict()
    assert d["invisible"] is True


def test_label_required():
    s = Sphere("", radius_mm=10)
    with pytest.raises(ValueError):
        s.to_dict()


# ---------- Pose accepts dict for back-compat ---------------------------

def test_pose_as_dict_accepted():
    s = Sphere("x", radius_mm=10, pose={"x": 100, "y": 50})
    d = s.to_dict()
    # Missing keys fill from identity.
    assert d["pose"] == {"x": 100.0, "y": 50.0, "z": 0.0,
                         "ox": 0.0, "oy": 0.0, "oz": 1.0, "theta": 0.0}


# ---------- Animation classes -------------------------------------------

from viam_visuals import (  # noqa: E402 — grouped with other animation imports
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
)


def test_static_emits_none_mode():
    assert Static().to_dict() == {"mode": "none"}


def test_spin_emits_period():
    assert Spin(period_s=6).to_dict() == {"mode": "spin", "period_s": 6.0}


def test_swing_emits_amplitude_and_period():
    assert Swing(amplitude_deg=75, period_s=8).to_dict() == {
        "mode": "swing", "amplitude_deg": 75.0, "period_s": 8.0,
    }


def test_swing_phase_offset_omitted_when_zero():
    d = Swing(amplitude_deg=10, period_s=2).to_dict()
    assert "phase_offset_s" not in d


def test_oscillate_rejects_bad_axis():
    with pytest.raises(ValueError):
        Oscillate(axis="w")


def test_oscillate_round_trip():
    d = Oscillate(axis="x", amplitude_mm=-10.0, period_s=3).to_dict()
    assert d == {"mode": "oscillate", "axis": "x",
                 "amplitude_mm": -10.0, "period_s": 3.0}


def test_orbit_round_trip():
    d = Orbit(radius_mm=100, period_s=4).to_dict()
    assert d == {"mode": "orbit", "radius_mm": 100.0, "period_s": 4.0}


def test_pulse_no_axis():
    d = Pulse(amplitude_mm=35, period_s=3).to_dict()
    assert d == {"mode": "pulse", "amplitude_mm": 35.0, "period_s": 3.0}


def test_pulse_with_axis():
    d = Pulse(amplitude_mm=100, period_s=4, axis="z").to_dict()
    assert d == {"mode": "pulse", "amplitude_mm": 100.0,
                 "period_s": 4.0, "axis": "z"}


def test_breathe_round_trip():
    d = Breathe(amplitude=0.55, period_s=1.5).to_dict()
    assert d == {"mode": "breathe", "amplitude": 0.55, "period_s": 1.5}


def test_flicker_default_keeps_uuid_rotation_on():
    d = Flicker(period_s=4, duty_cycle=0.55).to_dict()
    # rotate_uuid_on_readd default True → omitted (compact dict).
    # phase_offset_s is always emitted because preset tests read it
    # unconditionally.
    assert d == {"mode": "flicker", "period_s": 4.0,
                 "duty_cycle": 0.55, "phase_offset_s": 0.0}


def test_flicker_rotate_false_propagates():
    d = Flicker(period_s=4, duty_cycle=0.5, rotate_uuid_on_readd=False).to_dict()
    assert d["rotate_uuid_on_readd"] is False


def test_lifecycle_round_trip():
    d = Lifecycle(appear_s=1.0, alive_s=2.0, disappear_s=1.0,
                  gone_s=2.0, phase_offset_s=1.5).to_dict()
    assert d == {"mode": "lifecycle", "appear_s": 1.0, "alive_s": 2.0,
                 "disappear_s": 1.0, "gone_s": 2.0, "phase_offset_s": 1.5}


def test_force_vector_round_trip():
    d = ForceVector(period_s=5.0, length_amplitude_mm=80,
                    radius_amplitude_mm=5, tilt_deg=45,
                    precession_speed=1.0, color_speed=0.7).to_dict()
    assert d["mode"] == "force_vector"
    assert d["tilt_deg"] == 45.0


def test_trajectory_needs_at_least_two_waypoints():
    with pytest.raises(ValueError):
        Trajectory(waypoints=[{"x": 0}])


def test_trajectory_normalizes_pose_dicts():
    t = Trajectory(waypoints=[{"x": 0}, {"x": 100}], duration_s=4.0)
    d = t.to_dict()
    assert d["duration_s"] == 4.0
    assert d["loop"] is True
    assert len(d["waypoints"]) == 2
    # Missing keys filled from identity.
    assert d["waypoints"][0]["oz"] == 1.0


def test_visual_accepts_animation_instance():
    s = Sphere("x", radius_mm=10, animation=Spin(period_s=3))
    d = s.to_dict()
    assert d["animation"] == {"mode": "spin", "period_s": 3.0}


def test_visual_accepts_animation_dict_passthrough():
    s = Sphere("x", radius_mm=10, animation={"mode": "spin", "period_s": 3})
    d = s.to_dict()
    assert d["animation"] == {"mode": "spin", "period_s": 3}


# ---------- Composites --------------------------------------------------

from viam_visuals import (  # noqa: E402
    BoundingBox,
    Composite,
    CoordinateFrame,
    Line,
)


def test_coordinate_frame_expands_to_anchor_plus_3_axes():
    frame = CoordinateFrame("tcp", size_mm=120)
    items = frame.to_visuals()
    labels = [v.label for v in items]
    assert labels == ["tcp", "tcp_axis_x", "tcp_axis_y", "tcp_axis_z"]
    # All three axes parent to the anchor.
    for v in items[1:]:
        assert v.parent_frame == "tcp"


def test_coordinate_frame_animation_attaches_to_anchor():
    frame = CoordinateFrame("tcp", animation=Spin(period_s=2))
    items = frame.to_visuals()
    # Anchor gets the animation.
    assert items[0].animation.to_dict() == {"mode": "spin", "period_s": 2.0}
    # Axes are static (the chain composition propagates spin).
    for axis in items[1:]:
        assert axis.animation is None


def test_coordinate_frame_iterable():
    frame = CoordinateFrame("tcp")
    via_iter = list(frame)
    via_method = frame.to_visuals()
    assert [v.label for v in via_iter] == [v.label for v in via_method]


def test_line_emits_n_minus_1_segments():
    pts = [Pose.at(x=0), Pose.at(x=100), Pose.at(x=200, y=50)]
    line = Line("path", points=pts, width_mm=4, color=(0, 0, 255))
    items = line.to_visuals()
    assert len(items) == 2
    assert items[0].label == "path_seg_00"
    assert items[1].label == "path_seg_01"


def test_line_skips_coincident_points():
    pts = [Pose.at(x=0), Pose.at(x=0), Pose.at(x=100)]
    line = Line("path", points=pts, width_mm=4)
    items = line.to_visuals()
    # Only one real segment (0→100); the 0→0 coincident pair is skipped.
    assert len(items) == 1


def test_line_rejects_too_few_points():
    with pytest.raises(ValueError):
        Line("path", points=[Pose.at(x=0)])


def test_bounding_box_solid_returns_one_box():
    bb = BoundingBox("obj", dims_mm=(100, 200, 50), wireframe=False)
    items = bb.to_visuals()
    assert len(items) == 1
    assert isinstance(items[0], Box)


def test_bounding_box_wireframe_returns_12_capsules():
    bb = BoundingBox("obj", dims_mm=(100, 200, 50), wireframe=True)
    items = bb.to_visuals()
    assert len(items) == 12
    # 4 edges along each of x/y/z axes — length_mm matches the
    # box's dim on that axis.
    x_edges = [v for v in items if v.length_mm == 100]
    y_edges = [v for v in items if v.length_mm == 200]
    z_edges = [v for v in items if v.length_mm == 50]
    assert len(x_edges) == 4
    assert len(y_edges) == 4
    assert len(z_edges) == 4


def test_arrow_from_to_computes_length_and_orientation():
    a = Arrow.from_to("force", Pose.at(x=0, y=0, z=0),
                      Pose.at(x=3, y=4, z=0), radius_mm=5)
    # Length = 5, orientation = (3/5, 4/5, 0).
    assert a.length_mm == 5.0
    d = a.to_dict()
    assert d["pose"]["ox"] == 0.6
    assert d["pose"]["oy"] == 0.8
    assert d["pose"]["oz"] == 0.0


def test_arrow_from_to_rejects_coincident_points():
    with pytest.raises(ValueError):
        Arrow.from_to("x", Pose.at(x=10), Pose.at(x=10), radius_mm=5)
