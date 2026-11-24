"""Unit tests for HW4 closed-loop navigator state machine and control."""

import math
import time
import unittest

from robot_testing.hw4_control import HW4Navigator, NavigationState, wrap_angle


class TestHW4Navigator(unittest.TestCase):
    def setUp(self):
        self.nav = HW4Navigator(
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
        self.assertAlmostEqual(self.nav.waypoints_m[0][0], 2.345, places=3)
        self.assertAlmostEqual(self.nav.waypoints_m[0][1], 0.630, places=3)
        self.assertAlmostEqual(self.nav.waypoints_m[-1][0], 0.650, places=3)
        self.assertAlmostEqual(self.nav.waypoints_m[-1][1], 2.295, places=3)

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
        # When robot is at Start within tolerance:
        self.nav.update_pose(2.345, 0.630, 0.0, now)
        lin, ang, msg = self.nav.compute_motion_command(now)
        # Should advance from WP 0 to WP 1
        self.assertEqual(self.nav.current_wp_idx, 1)
        self.assertEqual(self.nav.state, NavigationState.WAYPOINT_REACHED)

    def test_orientation_vs_driving_state(self):
        """Verify robot orients when heading error is large, then drives when aligned."""
        now = time.monotonic()
        # Robot at WP 1, target is WP 2: (~0.57, ~1.26) m
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


if __name__ == '__main__':
    unittest.main()
