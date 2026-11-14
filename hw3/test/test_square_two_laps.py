"""Tests for two-lap square navigation: open-loop dead reckoning, return to (0,0), and closed-loop visual odometry."""

import math
import unittest

from hw3.square_navigation import NavigationState, SquareNavigator
from hw3.turn_calibration import ForwardCalibration, TurnCalibration
from hw3.vision_navigation import NavigationPolicy


class TestSquareTwoLaps(unittest.TestCase):
    def setUp(self):
        self.calibration = TurnCalibration(
            ccw_left_input=-0.40,
            ccw_right_input=0.37,
            cw_left_input=0.37,
            cw_right_input=-0.37,
            ccw_radps=math.pi / 2,
            cw_radps=math.pi / 2,
        )
        self.forward = ForwardCalibration(left_input=0.185, right_input=0.15, speed_mps=0.20)
        self.policy = NavigationPolicy(
            manual_motion_enabled=True,
            two_lap_square=True,
            manual_forward_pause_s=1.0,
            manual_midpoint_pause_s=1.0,
            corner_turn_duration_s=1.0,
            closed_loop_waypoint_tolerance_m=0.10,
            heading_drive_threshold_rad=0.35,
        )

    def test_square_lap1_sequence(self):
        """Verify that Lap 1 executes 4 sides and 4 turns with midpoint and stopping pauses."""
        nav = SquareNavigator(
            side_count=4,
            side_length_m=1.0,
            turn_calibration=self.calibration,
            forward_calibration=self.forward,
            policy=self.policy,
        )
        nav.start()

        # Initial tick at t=0: latches origin and begins driving side 1 along +X (no initial turn)
        cmd = nav.step(0.0)
        self.assertEqual(cmd.state, NavigationState.DRIVING)
        self.assertAlmostEqual(cmd.linear_v_mps, 0.20)
        self.assertAlmostEqual(cmd.angular_omega_radps, 0.0)
        self.assertEqual(nav.side_index, 0)
        self.assertEqual(nav.current_wp_idx, 1)

        # Drive half side (0.5m / 0.20m/s = 2.5s)
        # At 20 Hz, step 50 ticks to reach 2.5s
        now = 0.0
        for _ in range(50):
            now += 0.05
            cmd = nav.step(now)

        # Reached midpoint checkpoint
        self.assertEqual(cmd.state, NavigationState.MIDPOINT_PAUSE)
        self.assertAlmostEqual(cmd.linear_v_mps, 0.0)

        # Midpoint pause lasts 1.0s (up to t = 3.5s)
        for _ in range(20):
            now += 0.05
            cmd = nav.step(now)

        # Second half of side 1 starts
        self.assertEqual(cmd.state, NavigationState.DRIVING)
        self.assertAlmostEqual(cmd.linear_v_mps, 0.20)

    def test_two_lap_with_return_drift(self):
        """Simulate 2-lap mission with drift at end of Lap 1: exercises alignment and driving back to origin."""
        nav = SquareNavigator(
            side_count=4,
            side_length_m=1.0,
            turn_calibration=self.calibration,
            forward_calibration=self.forward,
            policy=self.policy,
        )
        nav.start()

        now = 0.0
        dt = 0.05
        max_time = 300.0

        sim_x = 0.0
        sim_y = 0.0
        sim_theta = 0.0

        states_seen = set()
        laps_seen = set()
        drift_injected = False

        while now < max_time and nav.is_active:
            cmd = nav.step(now)
            states_seen.add(cmd.state)
            laps_seen.add(nav.lap)

            # At the end of side 4 of Lap 1, inject a 30 cm offset to simulate dead-reckoning drift
            if nav.lap == 1 and nav.side_index == 4 and not drift_injected and nav.manual_final_turn_done:
                sim_x += 0.30
                sim_y -= 0.25
                sim_theta += 0.15
                drift_injected = True

            sim_theta += cmd.angular_omega_radps * dt
            sim_x += cmd.linear_v_mps * dt * math.cos(sim_theta)
            sim_y += cmd.linear_v_mps * dt * math.sin(sim_theta)

            nav.update_ekf_pose(sim_x, sim_y, sim_theta, 0.02, 0.02, now)
            now += dt

        self.assertIn(1, laps_seen)
        self.assertIn(2, laps_seen)
        self.assertIn(NavigationState.RELOCALIZING_TURN, states_seen)
        self.assertIn(NavigationState.RETURN_DRIVING, states_seen)
        self.assertIn(NavigationState.RETURN_PAUSE, states_seen)
        self.assertIn(NavigationState.REORIENTING, states_seen)
        self.assertEqual(nav.state, NavigationState.COMPLETED)
        self.assertFalse(nav.is_active)

    def test_two_lap_already_at_origin(self):
        """Simulate 2-lap mission when Lap 1 finishes already within tolerance of origin."""
        nav = SquareNavigator(
            side_count=4,
            side_length_m=1.0,
            turn_calibration=self.calibration,
            forward_calibration=self.forward,
            policy=self.policy,
        )
        nav.start()

        now = 0.0
        dt = 0.05
        max_time = 300.0

        sim_x = 0.0
        sim_y = 0.0
        sim_theta = 0.0

        states_seen = set()
        laps_seen = set()

        while now < max_time and nav.is_active:
            cmd = nav.step(now)
            states_seen.add(cmd.state)
            laps_seen.add(nav.lap)

            sim_theta += cmd.angular_omega_radps * dt
            sim_x += cmd.linear_v_mps * dt * math.cos(sim_theta)
            sim_y += cmd.linear_v_mps * dt * math.sin(sim_theta)

            nav.update_ekf_pose(sim_x, sim_y, sim_theta, 0.02, 0.02, now)
            now += dt

        self.assertIn(1, laps_seen)
        self.assertIn(2, laps_seen)
        self.assertIn(NavigationState.RETURN_PAUSE, states_seen)
        self.assertIn(NavigationState.REORIENTING, states_seen)
        self.assertEqual(nav.state, NavigationState.COMPLETED)
        self.assertFalse(nav.is_active)

    def test_octagon_remains_single_lap(self):
        """Confirm octagon (side_count=8) completes after 8 sides without Lap 2."""
        nav = SquareNavigator(
            side_count=8,
            side_length_m=0.75,
            turn_calibration=self.calibration,
            forward_calibration=self.forward,
            policy=self.policy,
        )
        nav.start()

        now = 0.0
        dt = 0.05
        while now < 100.0 and nav.is_active:
            cmd = nav.step(now)
            now += dt

        self.assertEqual(nav.side_index, 8)
        self.assertEqual(nav.lap, 1)
        self.assertEqual(nav.state, NavigationState.COMPLETED)
        self.assertFalse(nav.is_active)
