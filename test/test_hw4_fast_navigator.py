"""Unit tests for HW4 Fast A* closed-loop navigator state machine and control."""

import math
import time
import unittest

from robot_testing.hw4_fast import HW4FastNavigator, NavigationState, wrap_angle


class TestHW4FastNavigator(unittest.TestCase):
    def setUp(self):
        self.nav = HW4FastNavigator(
            start_cm=(234.5, 63.0),
            goal_cm=(65.0, 229.5),
            position_tolerance_m=0.06,
            heading_tolerance_rad=math.radians(7.0),
            max_pose_age_s=0.5,
        )
        self.nav.plan_path()

    def test_planning_and_waypoints(self):
        self.assertIsNotNone(self.nav.path)
        self.assertEqual(len(self.nav.waypoints_m), self.nav.path.waypoint_count)
        # Fast A* path has 4 waypoints: Start, 2 intermediate clearance nodes, Goal
        self.assertEqual(len(self.nav.waypoints_m), 4)
        self.assertAlmostEqual(self.nav.waypoints_m[0][0], 2.345, places=3)
        self.assertAlmostEqual(self.nav.waypoints_m[0][1], 0.630, places=3)
        self.assertAlmostEqual(self.nav.waypoints_m[-1][0], 0.650, places=3)
        self.assertAlmostEqual(self.nav.waypoints_m[-1][1], 2.295, places=3)

        # Path length should be ~267.25 cm (< 270 cm, significantly shorter than Voronoi 327.20 cm)
        self.assertLess(self.nav.path.total_length_cm, 270.0)
        self.assertGreater(self.nav.path.min_clearance_cm, 17.88)

    def test_stale_localization_stops_robot(self):
        """Navigator must halt robot if visual pose is missing or stale."""
        # Pose time is 0 initially -> stale
        lin, ang, msg = self.nav.compute_motion_command(time.monotonic())
        self.assertEqual(lin, 0.0)
        self.assertEqual(ang, 0.0)
        self.assertEqual(self.nav.state, NavigationState.LOCALIZATION_RECOVERY)
        self.assertIn("STOPPING", msg)

    def test_waypoint_advancement(self):
        """Simulate robot at each waypoint and verify sequential advancement."""
        now = time.monotonic()
        # WP 0 is Start (2.345, 0.630)
        self.nav.update_pose(2.345, 0.630, 0.0, now)
        lin, ang, msg = self.nav.compute_motion_command(now)
        # Should advance from WP 0 to WP 1
        self.assertEqual(self.nav.current_wp_idx, 1)
        self.assertEqual(self.nav.state, NavigationState.WAYPOINT_REACHED)

    def test_orientation_vs_driving_state(self):
        """Verify robot orients when heading error is large, then drives when aligned."""
        now = time.monotonic()
        # Robot at WP 1, target is WP 2:
        wp1 = self.nav.waypoints_m[1]
        wp2 = self.nav.waypoints_m[2]
        dx = wp2[0] - wp1[0]
        dy = wp2[1] - wp1[1]
        desired_yaw = math.atan2(dy, dx)

        # 1. Robot has misaligned heading (e.g. desired_yaw + 45 deg)
        self.nav.current_wp_idx = 2
        self.nav.update_pose(wp1[0], wp1[1], desired_yaw + math.radians(45.0), now)
        lin, ang, msg = self.nav.compute_motion_command(now)
        self.assertEqual(self.nav.state, NavigationState.ORIENTING)
        self.assertEqual(lin, 0.0)
        self.assertNotEqual(ang, 0.0)

        # 2. Robot is aligned with heading (heading error < 7 deg)
        self.nav.update_pose(wp1[0], wp1[1], desired_yaw, now)
        lin, ang, msg = self.nav.compute_motion_command(now)
        self.assertEqual(self.nav.state, NavigationState.DRIVING)
        self.assertGreater(lin, 0.0)
        self.assertEqual(ang, 0.0)

    def test_goal_reached(self):
        """Verify transition to GOAL_REACHED when final waypoint is reached."""
        now = time.monotonic()
        last_idx = len(self.nav.waypoints_m) - 1
        self.nav.current_wp_idx = last_idx
        goal_wp = self.nav.waypoints_m[last_idx]

        # Robot arrives at goal
        self.nav.update_pose(goal_wp[0], goal_wp[1], 0.0, now)
        lin, ang, msg = self.nav.compute_motion_command(now)
        self.assertEqual(self.nav.state, NavigationState.GOAL_REACHED)
        self.assertEqual(lin, 0.0)
        self.assertEqual(ang, 0.0)
        self.assertTrue(self.nav.is_finished)
        self.assertIn("Goal", msg)

    def test_calibrated_slower_turn_parameters(self):
        """Verify that default navigator settings use slower turn rate and camera settle time."""
        default_nav = HW4FastNavigator()
        # Calibrated for ~4.8s per 360 deg turn (~1.309 rad/s or ~75 deg/s)
        self.assertAlmostEqual(default_nav.turn_rate, 2.0 * math.pi / 4.8, places=2)
        self.assertLess(default_nav.turn_rate, 1.50)
        self.assertEqual(default_nav.settle_time, 1.2)
        self.assertEqual(default_nav.max_pose_age, 1.2)

    def test_recovery_transition_and_reacquisition(self):
        """Verify localization recovery triggered when blind, and recovered when tag seen."""
        now = time.monotonic()
        # Initially fresh pose
        self.nav.update_pose(2.345, 0.630, 0.0, now)
        lin, ang, _ = self.nav.compute_motion_command(now)
        self.assertNotEqual(self.nav.state, NavigationState.LOCALIZATION_RECOVERY)

        # After max_pose_age passes without pose update (e.g. tag lost/blind)
        stale_time = now + 1.5
        lin, ang, msg = self.nav.compute_motion_command(stale_time)
        self.assertEqual(self.nav.state, NavigationState.LOCALIZATION_RECOVERY)
        self.assertEqual(lin, 0.0)
        self.assertEqual(ang, 0.0)
        self.assertIn("stale", msg.lower())

        # Robot acquires landmark again
        reacquired_time = stale_time + 0.1
        self.nav.update_pose(2.0, 0.70, 0.0, reacquired_time)
        lin, ang, msg = self.nav.compute_motion_command(reacquired_time)
        self.assertIn(self.nav.state, (NavigationState.ORIENTING, NavigationState.DRIVING, NavigationState.WAYPOINT_REACHED))


if __name__ == '__main__':
    unittest.main()
