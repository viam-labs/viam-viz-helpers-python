"""Tests for the imperative pose-composing animation helpers."""


import pytest

import viam_visuals as viz

# ---- spin_pose --------------------------------------------------------

def test_spin_pose_at_t_zero_is_zero_theta():
    p = viz.spin_pose(viz.Pose.at(x=10), period_s=3.0, t=0.0)
    assert p.theta == 0.0


def test_spin_pose_at_one_period_wraps_to_zero():
    # theta = (360 * 3 / 3) % 360 = 0
    p = viz.spin_pose(viz.Pose.at(), period_s=3.0, t=3.0)
    assert p.theta == 0.0


def test_spin_pose_at_quarter_period_is_90():
    p = viz.spin_pose(viz.Pose.at(), period_s=4.0, t=1.0)
    assert abs(p.theta - 90.0) < 1e-9


def test_spin_pose_ignores_base_theta():
    # spin is absolute, not a delta.
    p = viz.spin_pose(viz.Pose.at(theta=180), period_s=4.0, t=0.0)
    assert p.theta == 0.0


def test_spin_pose_preserves_position():
    base = viz.Pose.at(x=100, y=200, z=300)
    p = viz.spin_pose(base, period_s=4.0, t=1.0)
    assert (p.x, p.y, p.z) == (100, 200, 300)


# ---- orbit_pose -------------------------------------------------------

def test_orbit_pose_z_axis_at_t_zero_lands_on_positive_x():
    # phase=0 → cos=1, sin=0 → +X offset only.
    base = viz.Pose.at(x=50, y=100, z=200)
    p = viz.orbit_pose(base, period_s=4.0, radius_mm=10.0, t=0.0)
    assert abs(p.x - 60) < 1e-9
    assert abs(p.y - 100) < 1e-9
    assert p.z == 200


def test_orbit_pose_z_axis_at_quarter_period_lands_on_positive_y():
    base = viz.Pose.at(x=50, y=100)
    p = viz.orbit_pose(base, period_s=4.0, radius_mm=10.0, t=1.0)
    assert abs(p.x - 50) < 1e-9
    assert abs(p.y - 110) < 1e-9


def test_orbit_pose_y_axis_orbits_xz():
    p = viz.orbit_pose(viz.Pose.at(), period_s=4.0, radius_mm=10.0, t=0.0, axis="y")
    assert abs(p.x - 10) < 1e-9
    assert p.y == 0
    assert abs(p.z) < 1e-9


def test_orbit_pose_unknown_axis_raises():
    with pytest.raises(ValueError, match="axis"):
        viz.orbit_pose(viz.Pose.at(), period_s=4.0, radius_mm=10.0, t=0.0, axis="w")


# ---- oscillate_pose ---------------------------------------------------

def test_oscillate_pose_at_t_zero_is_base():
    base = viz.Pose.at(x=10, y=20, z=30)
    p = viz.oscillate_pose(base, period_s=4.0, amplitude_mm=50.0, t=0.0)
    assert (p.x, p.y, p.z) == (10, 20, 30)


def test_oscillate_pose_default_axis_is_y():
    p = viz.oscillate_pose(viz.Pose.at(), period_s=4.0, amplitude_mm=50.0, t=1.0)
    assert abs(p.y - 50.0) < 1e-9
    assert p.x == 0 and p.z == 0


def test_oscillate_pose_axis_x():
    p = viz.oscillate_pose(viz.Pose.at(), period_s=4.0, amplitude_mm=50.0, t=1.0, axis="x")
    assert abs(p.x - 50.0) < 1e-9


def test_oscillate_pose_axis_z():
    p = viz.oscillate_pose(viz.Pose.at(), period_s=4.0, amplitude_mm=50.0, t=1.0, axis="z")
    assert abs(p.z - 50.0) < 1e-9


# ---- swing_pose -------------------------------------------------------

def test_swing_pose_at_t_zero_is_base_theta():
    p = viz.swing_pose(viz.Pose.at(theta=30), period_s=4.0, amplitude_deg=45.0, t=0.0)
    assert p.theta == 30.0


def test_swing_pose_at_quarter_period_is_base_plus_amplitude():
    p = viz.swing_pose(viz.Pose.at(theta=30), period_s=4.0, amplitude_deg=45.0, t=1.0)
    assert abs(p.theta - 75.0) < 1e-9


def test_swing_pose_preserves_position():
    base = viz.Pose.at(x=100, y=200, z=300, theta=10)
    p = viz.swing_pose(base, period_s=4.0, amplitude_deg=15.0, t=1.0)
    assert (p.x, p.y, p.z) == (100, 200, 300)


# ---- pulse_range -----------------------------------------------------

def test_pulse_range_at_t_zero_is_midpoint():
    # sin(0) = 0 → returns the midpoint.
    assert viz.pulse_range(80, 160, period_s=2.0, t=0.0) == 120.0


def test_pulse_range_at_quarter_period_is_high():
    # sin(π/2) = 1 → returns hi.
    assert abs(viz.pulse_range(80, 160, period_s=4.0, t=1.0) - 160.0) < 1e-9


def test_pulse_range_at_three_quarter_period_is_low():
    # sin(3π/2) = -1 → returns lo.
    assert abs(viz.pulse_range(80, 160, period_s=4.0, t=3.0) - 80.0) < 1e-9


def test_pulse_range_handles_negative_amplitude_via_reversed_args():
    # Same period, just reversed lo/hi — the sinusoid is flipped.
    a = viz.pulse_range(80, 160, period_s=4.0, t=1.0)
    b = viz.pulse_range(160, 80, period_s=4.0, t=1.0)
    assert abs((a - 120) + (b - 120)) < 1e-9


# ---- trajectory_pose -------------------------------------------------

def test_trajectory_pose_at_t_zero_is_first_waypoint():
    wps = [viz.Pose.at(x=0), viz.Pose.at(x=100), viz.Pose.at(x=200)]
    p = viz.trajectory_pose(wps, duration_s=10.0, t=0.0)
    assert abs(p.x - 0) < 1e-9


def test_trajectory_pose_at_midpoint_of_segment():
    # 2 segments, 10 s total. At t = 2.5 s (half through first 5 s
    # segment) we're at midpoint between wp 0 and wp 1.
    wps = [viz.Pose.at(x=0), viz.Pose.at(x=100), viz.Pose.at(x=200)]
    p = viz.trajectory_pose(wps, duration_s=10.0, t=2.5)
    assert abs(p.x - 50) < 1e-6


def test_trajectory_pose_at_lap_end_loop_snaps_back():
    wps = [viz.Pose.at(x=0), viz.Pose.at(x=100)]
    p = viz.trajectory_pose(wps, duration_s=10.0, t=10.0, loop=True)
    # t=10 mod 10 = 0 → first waypoint.
    assert abs(p.x - 0) < 1e-6


def test_trajectory_pose_no_loop_clamps_at_final():
    wps = [viz.Pose.at(x=0), viz.Pose.at(x=100)]
    p = viz.trajectory_pose(wps, duration_s=10.0, t=20.0, loop=False)
    # Clamped to final waypoint.
    assert abs(p.x - 100) < 1e-6


def test_trajectory_pose_raises_on_too_few_waypoints():
    with pytest.raises(ValueError, match="≥ 2 waypoints"):
        viz.trajectory_pose([viz.Pose.at()], duration_s=10.0, t=0.0)
