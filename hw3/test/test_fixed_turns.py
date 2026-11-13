"""Regression checks for calibrated mapping, bounded pivots, and command timing."""

import ast
import math
import json
import threading
from pathlib import Path
from types import SimpleNamespace as Record
import unittest

import numpy as np

from hw3.turn_calibration import ForwardCalibration, TurnCalibration, read_forward_calibration
from hw3.velocity_mapping import VelocityMapper
from hw3.vision_navigation import NavigationPolicy, VisualFeedback, forward_wall_clearance, tag_normal_yaw, relative_tag_turn
from hw3.ekf_slam import EkfSlam
from hw3.square_navigation import NavigationState, SquareNavigator


ROOT = Path(__file__).resolve().parents[2]


def load_method(path, class_name, method_name, namespace=None):
    """Load an actual ROS callback without importing unavailable ROS libraries.

    Args:
        path: Path to the CSV file being read.
        class_name: Name of the ROS adapter class containing the method under test.
        method_name: Name of the method to load without importing ROS.
        namespace: Optional globals supplied to the isolated method.

    Returns:
        The requested method as a callable for isolated regression checks.
    """
    tree = ast.parse((ROOT / path).read_text())
    cls = next(c for c in tree.body if isinstance(c, ast.ClassDef) and c.name == class_name)
    method = next(m for m in cls.body if isinstance(m, ast.FunctionDef) and m.name == method_name)
    for arg in method.args.args:
        arg.annotation = None
    method.returns = None
    scope = dict(namespace or {})
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), scope)
    return scope[method_name]


class TestFixedTurns(unittest.TestCase):
    def setUp(self):
        self.calibration = TurnCalibration(-0.3, 0.3, 0.3, -0.3, 0.6, 0.5)

    def test_direction_specific_fixed_rates_and_inputs(self):
        for request in (0.001, 0.2, 2.0):
            self.assertEqual(self.calibration.rate(request), 0.6)
            self.assertEqual(self.calibration.motors(request), (-0.3, 0.3))
            self.assertEqual(self.calibration.rate(-request), -0.5)
            self.assertEqual(self.calibration.motors(-request), (0.3, -0.3))
        self.assertEqual(self.calibration.motors(0), (0, 0))

    def test_measured_direction_specific_motor_inputs(self):
        calibration = TurnCalibration(-0.40, 0.37, 0.37, -0.37, math.pi / 2, math.pi / 2)
        mapper = VelocityMapper(calibration)
        ccw, cw = mapper.map_command(0, 1), mapper.map_command(0, -1)
        self.assertEqual((ccw.left, ccw.right), (-0.40, 0.37))
        self.assertEqual((cw.left, cw.right), (0.37, -0.37))
        self.assertAlmostEqual(ccw.angular_radps * 1.0, math.pi / 2)
        self.assertAlmostEqual(cw.angular_radps * 1.0, -math.pi / 2)

    def test_measured_asymmetric_quarter_turn_lasts_one_second(self):
        """Exercise the nominal 90-degree command at the 20 Hz control frequency.

        Args:
            None.

        Returns:
            None. Confirms exactly one second of [-0.40, +0.37] command prediction.
        """
        calibration = TurnCalibration(-0.40, 0.37, 0.37, -0.37, math.pi / 2, math.pi / 2)
        mapper = VelocityMapper(calibration)
        nav = SquareNavigator(turn_calibration=calibration)
        nav.start()
        nav._transition(NavigationState.STOPPING, 1)
        nav._begin_corner(1)
        moving_ticks = 0
        for tick in range(60):
            cmd = nav.step(1 + tick * 0.05)
            if cmd.angular_omega_radps:
                moving_ticks += 1
                motors = mapper.map_command(0, cmd.angular_omega_radps)
                self.assertEqual((motors.left, motors.right), (-0.40, 0.37))
                self.assertAlmostEqual(motors.angular_radps, math.pi / 2)
            if cmd.state == NavigationState.CORNER_PAUSE:
                break
        self.assertEqual(moving_ticks, 20)
        self.assertAlmostEqual(nav.corner_used, math.pi / 2)

    def test_turn_calibration_rejects_wrong_direction_signs(self):
        for motors in [(0.4, 0.37, 0.4, -0.4), (-0.4, -0.37, 0.4, -0.4),
                       (-0.4, 0.37, -0.4, 0.4), (-0.4, 0.6, 0.4, -0.4)]:
            with self.assertRaises(ValueError):
                TurnCalibration(*motors, math.pi / 2, 1.75)

    def test_unmeasured_or_invalid_calibration_is_rejected(self):
        for values in [(0.3, 0.3, 0, 0), (0.6, 0.3, 0.5, 0.5),
                       (0.3, 0.6, 0.5, 0.5), (0.3, 0.3, -1, 1), (0.3, 0.3, math.nan, 1)]:
            with self.assertRaises(ValueError):
                TurnCalibration(-values[0], values[0], values[1], -values[1], *values[2:])
        nav = SquareNavigator()
        self.assertFalse(nav.start()[0])




    def feed_vision(self, nav, now, pose, clearance=None):
        """Supply two fixed synthetic landmarks in the moving robot base frame."""
        x, y, yaw = pose
        c, s = math.cos(yaw), math.sin(yaw)
        tags = [[tag, c * (tx - x) + s * (ty - y), -s * (tx - x) + c * (ty - y)]
                for tag, tx, ty in [(1, 5.0, 4.0), (3, -2.0, 3.0)]]
        nav.update_vision(now, tags, [1, 3], clearance)

    def test_first_leg_requires_no_tags_and_reaches_turn_with_uncertain_pose(self):
        """An empty map cannot block the bounded first leg or its following turn.

        Args:
            None.

        Returns:
            None. Confirms immediate forward commands and nominal first-leg distance.
        """
        nav = SquareNavigator(turn_calibration=self.calibration)
        self.assertTrue(nav.start()[0])
        for tick in range(500):
            now = 1 + tick * 0.05
            nav.update_ekf_pose(0, 0, 0, 1.0, 1.0, now)
            cmd = nav.step(now)
            if tick == 0:
                self.assertGreater(cmd.linear_v_mps, 0)
                self.assertEqual(cmd.angular_omega_radps, 0)
            if cmd.state == NavigationState.TURNING:
                break
        self.assertEqual(nav.state, NavigationState.TURNING, nav.stop_reason)
        self.assertAlmostEqual(nav.drive_used, nav.square_size_m, delta=0.11)
        self.assertGreater(nav.step(now + 0.05).angular_omega_radps, 0)
        self.assertEqual(nav.vision.valid_stamp, -math.inf)

    def test_new_tag_wall_distance_is_logged_without_stopping_first_leg(self):
        nav = self.prepared(NavigationState.DRIVING)
        nav.update_vision(1, [[99, 0.3, 0]], [], 0.3)
        cmd = nav.step(1)
        self.assertGreater(cmd.linear_v_mps, 0)
        self.assertEqual(nav.vision.clearance, 0.3)
        self.assertTrue(nav.is_active)

    def test_square_has_four_direct_turns_and_fixed_headings(self):
        """Simulate four visually observed legs and corners, including the final heading.

        Args:
            None.

        Returns:
            None. Verifies sequence, heading, command separation, and approximate closure.
        """
        nav = SquareNavigator(turn_calibration=self.calibration, turn_settle_s=0.1)
        nav.start()
        x = y = yaw = 0.0
        states, previous, corners = set(), None, 0
        for tick in range(6000):
            now = 1 + tick * 0.05
            nav.update_ekf_pose(x, y, yaw, 0.01, 0.01, now)
            self.feed_vision(nav, now, (x, y, yaw))
            cmd = nav.step(now)
            corners += int(cmd.state == NavigationState.TURNING and previous != cmd.state)
            previous = cmd.state
            states.add(cmd.state)
            self.assertFalse(cmd.linear_v_mps and cmd.angular_omega_radps)
            x += cmd.linear_v_mps * math.cos(yaw) * 0.05
            y += cmd.linear_v_mps * math.sin(yaw) * 0.05
            yaw += cmd.angular_omega_radps * 0.05
            if not nav.is_active:
                break
        self.assertEqual(nav.state, NavigationState.COMPLETED, nav.stop_reason)
        self.assertEqual(corners, 4)
        self.assertIn(NavigationState.CORNER_PAUSE, states)
        self.assertLess(abs(math.atan2(math.sin(yaw), math.cos(yaw))), nav.heading_tolerance_rad)
        self.assertLess(math.hypot(x, y), 0.4)

    def test_complete_square_with_no_tags_and_unreliable_ekf(self):
        """Run all four sides and corners without any visual observations.

        Args:
            None.

        Returns:
            None. Verifies completion, no tag waits, four corners, and nominal command closure.
        """
        nav = SquareNavigator(turn_calibration=self.calibration, turn_settle_s=0.1)
        nav.start()
        x = y = yaw = 0.0
        corners, previous = 0, None
        for tick in range(8000):
            now = 1 + tick * 0.05
            nav.update_ekf_pose(0, 0, 0, 10, 10, now)
            cmd = nav.step(now)
            corners += int(cmd.state == NavigationState.TURNING and previous != cmd.state)
            previous = cmd.state
            self.assertFalse(cmd.linear_v_mps and cmd.angular_omega_radps)
            x += cmd.linear_v_mps * math.cos(yaw) * 0.05
            y += cmd.linear_v_mps * math.sin(yaw) * 0.05
            yaw += cmd.angular_omega_radps * 0.05
            if not nav.is_active:
                break
        self.assertEqual(nav.state, NavigationState.COMPLETED, nav.stop_reason)
        self.assertEqual(corners, 4)
        self.assertEqual(nav.vision.valid_stamp, -math.inf)
        self.assertLess(math.hypot(x, y), 0.4)
        self.assertLess(abs(math.atan2(math.sin(yaw), math.cos(yaw))), nav.heading_tolerance_rad)


    def test_entire_square_without_ekf_or_camera_messages(self):
        """Missing sensors cannot block the supervised command-predicted mission.

        Args:
            None.

        Returns:
            None. Verifies all four legs and corners finish with no sensor callbacks.
        """
        nav = SquareNavigator(turn_calibration=self.calibration, turn_settle_s=0.1)
        nav.start()
        corners, midpoints, previous = 0, 0, None
        for tick in range(8000):
            cmd = nav.step(1 + tick * 0.05)
            corners += int(cmd.state == NavigationState.TURNING and previous != cmd.state)
            midpoints += int(cmd.state == NavigationState.MIDPOINT_PAUSE and previous != cmd.state)
            previous = cmd.state
            if not nav.is_active:
                break
        self.assertEqual(nav.state, NavigationState.COMPLETED)
        self.assertEqual(corners, 4)
        self.assertEqual(midpoints, 4)
        self.assertIsNone(nav.latest_slam_pose)

    def test_manual_stop_still_cancels_motion(self):
        nav = self.prepared(NavigationState.TURNING)
        nav.step(1.05)
        nav.stop('Supervisor stop')
        cmd = nav.step(1.1)
        self.assertEqual((cmd.linear_v_mps, cmd.angular_omega_radps), (0, 0))
        self.assertEqual(nav.state, NavigationState.STOPPED)
        self.assertFalse(nav.start()[0])

    def at_midpoint(self):
        """Prepare a half-traveled leg and enter its stopped midpoint phase."""
        nav = self.prepared(NavigationState.DRIVING)
        nav.policy = NavigationPolicy(midpoint_pause_s=1.0)  # explicit fixture timing
        nav._reset_midpoint()
        nav.drive_used = nav.square_size_m / 2
        self.assertEqual(nav.step(2).state, NavigationState.MIDPOINT_PAUSE)
        return nav

    def test_default_pauses_are_three_seconds_and_never_require_feedback(self):
        """Both acquisition pauses remain stopped for three seconds, then continue.

        Args:
            None.

        Returns:
            None. Checks midpoint and post-turn timing without sensor observations.
        """
        nav = self.at_midpoint()
        nav.policy = NavigationPolicy()
        self.assertEqual(nav.policy.midpoint_pause_s, 3.0)
        self.assertEqual(nav.step(4.99).state, NavigationState.MIDPOINT_PAUSE)
        self.assertEqual(nav.step(5).state, NavigationState.DRIVING)
        self.assertGreater(nav.step(5.05).linear_v_mps, 0)
        nav = SquareNavigator(turn_calibration=self.calibration)
        self.assertEqual(nav.turn_settle_s, 3.0)
        nav.start()
        nav.side_index = 1
        nav._transition(NavigationState.CORNER_PAUSE, 1)
        self.assertEqual(nav.step(3.99).state, NavigationState.CORNER_PAUSE)
        self.assertEqual(nav.step(4).state, NavigationState.DRIVING)

    def test_midpoint_pause_accepts_ekf_undershoot_without_steering(self):
        """A shorter measured first half increases only the remaining forward distance.

        Args:
            None.

        Returns:
            None. Confirms stopped acquisition time, fixed leg origin, and zero steering.
        """
        nav = self.at_midpoint()
        origin, heading, target = nav.leg_origin, nav._heading(), nav.target
        self.assertEqual(nav.step(2.5).linear_v_mps, 0)
        self.assertFalse(nav.midpoint_done)
        nav.update_ekf_pose(0.75, 0.2, 0.1, 0.01, 0.01, 3)
        cmd = nav.step(3)
        self.assertEqual((cmd.linear_v_mps, cmd.angular_omega_radps), (0, 0))
        self.assertAlmostEqual(nav.midpoint_progress_m, 0.75)
        self.assertAlmostEqual(nav.midpoint_remaining_m, nav.square_size_m / 2 + 0.10)
        self.assertEqual((nav.leg_origin, nav._heading(), nav.target), (origin, heading, target))
        cmd = nav.step(3.05)
        self.assertGreater(cmd.linear_v_mps, 0)
        self.assertEqual(cmd.angular_omega_radps, 0)
        self.assertEqual(nav.midpoint_source, 'fresh EKF')

    def test_midpoint_ekf_overshoot_reduces_remaining_distance(self):
        nav = self.at_midpoint()
        nav.update_ekf_pose(1.1, 0, 0, 0.01, 0.01, 3)
        nav.step(3)
        self.assertAlmostEqual(nav.midpoint_remaining_m, nav.square_size_m / 2 - 0.10)
        self.assertGreater(nav.progress_correction_m, 0)
        self.assertEqual(nav.step(3.05).angular_omega_radps, 0)

    def test_midpoint_bad_or_missing_ekf_uses_finite_nominal_fallback(self):
        """Missing, stale, or uncertain observations do not prolong the midpoint pause.

        Args:
            None.

        Returns:
            None. Checks nominal half-side remainder and automatic forward continuation.
        """
        for pose in [None, (0.75, 0, 0, 0.01, 0.01, 2.1), (0.75, 0, 0, 10, 10, 3)]:
            nav = self.at_midpoint()
            if pose is not None:
                nav.update_ekf_pose(*pose)
            nav.step(3)
            self.assertEqual(nav.state, NavigationState.DRIVING)
            self.assertAlmostEqual(nav.midpoint_remaining_m, nav.square_size_m / 2)
            self.assertEqual(nav.midpoint_source, 'command prediction fallback')
            self.assertGreater(nav.step(3.05).linear_v_mps, 0)

    def test_midpoint_uses_fixed_side_axis_not_distance_to_a_new_target(self):
        nav = self.at_midpoint()
        nav.side_index = 1
        nav.update_ekf_pose(0.4, 0.8, 1.8, 0.01, 0.01, 3)
        nav.step(3)
        self.assertAlmostEqual(nav.midpoint_progress_m, 0.8)
        self.assertAlmostEqual(nav._heading(), math.pi / 2)
        self.assertEqual(nav.step(3.05).angular_omega_radps, 0)

    def test_manual_stop_during_midpoint_does_not_resume(self):
        nav = self.at_midpoint()
        nav.stop('Supervisor stop')
        cmd = nav.step(4)
        self.assertEqual((cmd.linear_v_mps, cmd.angular_omega_radps), (0, 0))
        self.assertFalse(nav.is_active)

    def prepared(self, state):
        """Create an active controller for one focused phase check.

        Args:
            state: NavigationState to enter after initializing a zero pose.

        Returns:
            SquareNavigator with measured test rates and optional corner state.
        """
        nav = SquareNavigator(turn_calibration=self.calibration, turn_settle_s=0.1)
        nav.start()
        nav.update_ekf_pose(0, 0, 0, 0.01, 0.01, 1)
        nav._transition(state, 1)
        if state == NavigationState.TURNING:
            nav._begin_corner(1)
        if state == NavigationState.DRIVING:
            nav.midpoint_done = True
        return nav

    def test_stale_pose_does_not_abort_turn(self):
        nav = self.prepared(NavigationState.TURNING)
        self.assertGreater(nav.step(1.05).angular_omega_radps, 0)
        self.assertGreater(nav.step(2).angular_omega_radps, 0)
        self.assertEqual(nav.state, NavigationState.TURNING)
        self.assertTrue(nav.is_active)





    def test_corner_is_one_nominal_quarter_turn_then_camera_pause(self):
        """Command one continuous quarter-turn without tags or intermediate scan stages.

        Args:
            None.

        Returns:
            None. Verifies approximately 90 degrees, a stopped pause, and continuation.
        """
        nav = self.prepared(NavigationState.TURNING)
        nav.turn_settle_s = 1.0
        previous = angle = 0.0
        stopped_at = None
        for tick in range(150):
            now = 1 + tick * 0.05
            angle += previous * 0.05
            cmd = nav.step(now)
            if cmd.state == NavigationState.CORNER_PAUSE and stopped_at is None:
                stopped_at = now
            if cmd.state == NavigationState.CORNER_PAUSE:
                self.assertEqual(cmd.angular_omega_radps, 0)
            previous = cmd.angular_omega_radps
            if cmd.state == NavigationState.DRIVING:
                break
        self.assertEqual(nav.state, NavigationState.DRIVING)
        self.assertAlmostEqual(angle, math.pi / 2, delta=self.calibration.ccw_radps * 0.05)
        self.assertGreaterEqual(now - stopped_at, 1.0 - 1e-8)
        self.assertEqual(nav.correction_count, 0)

    def test_same_tag_orientation_corrects_without_a_learned_map(self):
        """A shared tag measures turn error independently of global landmark positions.

        Args:
            None.

        Returns:
            None. Confirms an 80-degree observation selects a bounded left correction.
        """
        nav = self.prepared(NavigationState.STOPPING)
        nav.navigation_yaw = 0.5  # deliberately biased global heading
        nav.update_vision(2, [[7, 2, 1]], [], None, [[7, -0.4]])
        nav._begin_corner(2)
        nav.corner_used = math.pi / 2
        nav._transition(NavigationState.CORNER_PAUSE, 3)
        nav.update_vision(3.2, [[7, 1, -2]], [], None, [[7, -0.4 - math.radians(80)]])
        cmd = nav.step(3.2)
        self.assertEqual(cmd.state, NavigationState.CORRECTING)
        self.assertGreater(nav.correction_omega, 0)
        self.assertAlmostEqual(nav.correction_goal, math.radians(10))
        self.assertEqual(nav.corner_source, 'same-tag face orientation')
        self.assertEqual(nav.vision.valid_stamp, -math.inf)
        for tick in range(40):
            now = 3.25 + tick * 0.05
            if nav.state == NavigationState.CORNER_PAUSE:
                angle = math.radians(80) + nav.correction_signed
                nav.update_vision(now, [[7, 1, -2]], [], None, [[7, -0.4 - angle]])
            cmd = nav.step(now)
            if cmd.state == NavigationState.DRIVING:
                break
        self.assertEqual(nav.state, NavigationState.DRIVING)
        self.assertEqual(nav.correction_count, 1)
        self.assertEqual(nav.corner_source, 'same-tag face orientation')

    def corner_with_measured_turn(self, angle_degrees):
        """Prepare a fresh same-tag quarter-turn measurement without a global map."""
        nav = self.prepared(NavigationState.STOPPING)
        nav.update_vision(2, [[7, 2, 1]], [], None, [[7, 0.4]])
        nav._begin_corner(2)
        nav.corner_used = math.pi / 2
        nav._transition(NavigationState.CORNER_PAUSE, 3)
        nav.update_vision(3.2, [[7, 2, 1]], [], None, [[7, 0.4 - math.radians(angle_degrees)]])
        return nav

    def test_sub_ten_degree_corrections_are_skipped_in_both_directions(self):
        for observed in (81, 85, 89, 91, 95, 99):
            nav = self.corner_with_measured_turn(observed)
            cmd = nav.step(3.2)
            self.assertEqual(cmd.state, NavigationState.DRIVING)
            self.assertEqual(cmd.angular_omega_radps, 0)
            self.assertEqual(nav.correction_count, 0)

    def test_ten_degree_overshoot_uses_right_turn_only_for_correction(self):
        """An actionable overshoot selects the measured clockwise pair.

        Args:
            None.

        Returns:
            None. Confirms clockwise direction, full fixed inputs, and nominal rate.
        """
        nav = self.corner_with_measured_turn(100)
        nav.turn_calibration = TurnCalibration(-0.4, 0.37, 0.37, -0.37, math.pi / 2, math.pi / 2)
        nav.step(3.2)
        self.assertEqual(nav.state, NavigationState.CORRECTING)
        cmd = nav.step(3.25)
        self.assertLess(cmd.angular_omega_radps, 0)
        command = VelocityMapper(nav.turn_calibration).map_command(0, cmd.angular_omega_radps)
        self.assertEqual((command.left, command.right), (0.37, -0.37))
        self.assertAlmostEqual(abs(command.angular_radps), math.pi / 2)
        self.assertAlmostEqual(nav.correction_goal, math.radians(10))

    def test_remaining_correction_budget_below_ten_degrees_is_skipped(self):
        nav = self.corner_with_measured_turn(70)
        nav.correction_used = math.radians(6)
        cmd = nav.step(3.2)
        self.assertEqual(cmd.state, NavigationState.DRIVING)
        self.assertEqual(cmd.angular_omega_radps, 0)
        self.assertEqual(nav.correction_count, 0)

    def test_changed_tag_without_map_feedback_does_not_trigger_correction(self):
        nav = self.prepared(NavigationState.STOPPING)
        nav.update_vision(2, [[7, 2, 1]], [], None, [[7, -0.4]])
        nav._begin_corner(2)
        nav._transition(NavigationState.CORNER_PAUSE, 3)
        nav.update_vision(3.2, [[9, 2, 1]], [], None, [[9, -1.8]])
        self.assertEqual(nav.step(3.2).state, NavigationState.DRIVING)
        self.assertEqual(nav.correction_count, 0)

    def test_pre_turn_image_cannot_authorize_post_turn_correction(self):
        nav = self.prepared(NavigationState.STOPPING)
        nav.update_vision(2, [[7, 2, 1]], [], None, [[7, -0.4]])
        nav._begin_corner(2)
        nav.corner_used = math.pi / 2
        nav._transition(NavigationState.CORNER_PAUSE, 3)
        self.assertEqual(nav.step(3.2).state, NavigationState.DRIVING)
        self.assertEqual(nav.correction_count, 0)

    def test_bad_map_corrections_cannot_cause_repeated_spinning(self):
        """Persistent misleading visual heading consumes only the small correction budget.

        Args:
            None.

        Returns:
            None. Checks bounded added rotation and automatic next-side continuation.
        """
        nav = self.prepared(NavigationState.TURNING)
        nav.corner_used = math.pi / 2
        nav._transition(NavigationState.CORNER_PAUSE, 1)
        for tick in range(150):
            now = 1 + tick * 0.05
            self.feed_vision(nav, now, (0, 0, 0))
            nav.update_ekf_pose(0, 0, 0, 0.01, 0.01, now)
            cmd = nav.step(now)
            if cmd.state == NavigationState.DRIVING:
                break
        self.assertEqual(nav.state, NavigationState.DRIVING)
        self.assertLessEqual(nav.correction_used,
                             nav.policy.corner_correction_limit_rad + self.calibration.ccw_radps * 0.05)
        self.assertLessEqual(nav.correction_count, nav.policy.corner_max_corrections)

    def test_manual_stop_during_correction_prevents_more_motion(self):
        nav = self.prepared(NavigationState.CORRECTING)
        nav.correction_goal, nav.correction_omega = 0.2, 0.6
        self.assertGreater(nav.step(1).angular_omega_radps, 0)
        nav.stop('Supervisor stop')
        self.assertEqual(nav.step(1.05).angular_omega_radps, 0)

    def test_known_tag_wall_distance_does_not_stop_or_pivot(self):
        nav = self.prepared(NavigationState.DRIVING)
        self.feed_vision(nav, 1, (0, 0, 0), clearance=0.3)
        cmd = nav.step(1)
        self.assertGreater(cmd.linear_v_mps, 0)
        self.assertEqual(cmd.angular_omega_radps, 0)
        self.assertEqual(nav.vision.clearance, 0.3)

    def test_empty_frames_do_not_refresh_vision_or_block_forward_fallback(self):
        nav = self.prepared(NavigationState.DRIVING)
        nav.side_index = 1
        self.feed_vision(nav, 1, (0, 0, 0))
        nav.update_ekf_pose(0, 0, 0, 10, 10, 3)
        nav.update_vision(3, [], [], None)
        self.assertFalse(nav.vision.fresh(3))
        self.assertGreater(nav.step(3).linear_v_mps, 0)

    def test_heading_drift_defers_correction_until_corner(self):
        nav = self.prepared(NavigationState.DRIVING)
        nav.update_ekf_pose(0.5, 0, 0.5, 0.01, 0.01, 1.1)
        self.feed_vision(nav, 1.1, (0.5, 0, 0.5))
        cmd = nav.step(1.1)
        self.assertGreater(cmd.linear_v_mps, 0)
        self.assertEqual(cmd.angular_omega_radps, 0)

    def test_frozen_visual_distance_ends_leg_at_nominal_budget(self):
        """Unchanging observations cannot produce unbounded travel or abort the square.

        Args:
            None.

        Returns:
            None. Verifies a bounded leg followed by the next turn phase.
        """
        nav = self.prepared(NavigationState.DRIVING)
        nav.side_index = 1
        for tick in range(600):
            now = 1 + tick * 0.05
            nav.update_ekf_pose(0, 0, math.pi / 2, 0.01, 0.01, now)
            self.feed_vision(nav, now, (0, 0, math.pi / 2))
            cmd = nav.step(now)
            if nav.state == NavigationState.STOPPING:
                break
        self.assertEqual(nav.state, NavigationState.STOPPING)
        self.assertTrue(nav.is_active)
        self.assertLessEqual(nav.drive_used, nav.policy.drive_budget_factor * nav.square_size_m + 0.01)
        self.assertEqual(cmd.linear_v_mps, 0)

    def test_late_visual_progress_cannot_shorten_the_remaining_leg(self):
        """A live tag/EKF jump cannot override the frozen midpoint distance plan.

        Args:
            None.

        Returns:
            None. Confirms continued forward travel despite misleading visual progress.
        """
        nav = self.prepared(NavigationState.DRIVING)
        nav.vision.begin_leg(0)
        nav.update_vision(1, [[1, 3, 0], [3, 3, 1]], [1, 3], None)
        nav.update_ekf_pose(0.1, 0, 0, 0.01, 0.01, 1.1)
        nav.update_vision(1.1, [[1, 0.1, 0], [3, 0.1, 1]], [1, 3], None)
        cmd = nav.step(1.1)
        self.assertGreater(nav.vision.progress, nav.square_size_m)
        self.assertAlmostEqual(nav.progress, nav.drive_used)
        self.assertGreater(cmd.linear_v_mps, 0)
        self.assertEqual(cmd.angular_omega_radps, 0)
        self.assertEqual(nav.state, NavigationState.DRIVING)

    def test_false_near_complete_ekf_midpoint_retains_about_three_feet(self):
        """Reproduce the reported short second half and constrain the EKF's authority.

        Args:
            None.

        Returns:
            None. Checks a 1.77 m EKF estimate cannot reduce the remainder to 5 cm.
        """
        nav = self.at_midpoint()
        nav.update_ekf_pose(1.77, 0, 0, 0.01, 0.01, 3)
        nav.step(3)
        self.assertAlmostEqual(nav.midpoint_progress_m, 1.77)
        self.assertAlmostEqual(nav.progress_correction_m, 0.10)
        self.assertAlmostEqual(nav.midpoint_remaining_m, 0.8144)
        self.assertGreater(nav.step(3.05).linear_v_mps, 0)
        self.assertEqual(nav.state, NavigationState.DRIVING)

    def test_midpoint_correction_bounds_both_signs_and_honors_configuration(self):
        for estimated, expected in [(-5, -0.05), (5, 0.05)]:
            nav = self.at_midpoint()
            nav.policy = NavigationPolicy(midpoint_pause_s=1, midpoint_max_correction_m=0.05)
            nav.update_ekf_pose(estimated, 0, 0, 0.01, 0.01, 3)
            nav.step(3)
            self.assertAlmostEqual(nav.progress_correction_m, expected)
            self.assertAlmostEqual(nav.midpoint_remaining_m, nav.square_size_m / 2 - expected)



class TestVisionGeometry(unittest.TestCase):
    def test_oblique_wall_distance_uses_plane_not_tag_range(self):
        transform = np.eye(4)
        transform[:3, 2] = [1, 0, 0]
        transform[:3, 3] = [0.4, 1.2, 0.2]
        self.assertAlmostEqual(forward_wall_clearance(transform), 0.4)
        transform[:3, 2] = [0, 1, 0]
        self.assertIsNone(forward_wall_clearance(transform))

    def test_duplicate_or_empty_frames_cannot_refresh_visual_corrections(self):
        feedback = VisualFeedback(NavigationPolicy())
        feedback.update(1, [[1, 2, 0]], [1], None, (0, 0, 0), 0)
        for _ in range(10):
            feedback.update(1, [[1, 3, 0]], [1], None, (0, 0, 0), 0)
        feedback.update(3, [], [], None, (0, 0, 0), 0)
        self.assertEqual(feedback.valid_stamp, 1)
        self.assertFalse(feedback.fresh(3))

    def test_feedback_publisher_converts_optical_geometry_to_base(self):
        """Exercise the real ROS adapter with calibrated camera extrinsics.

        Args:
            None.

        Returns:
            None. Checks original image time, tag coordinates, and independent clearance.
        """
        from hw3.coordinates import CameraExtrinsics
        published = []
        method = load_method('hw3/hw3/nodes/ekf_slam_node.py', 'EkfSlamNode',
                             '_publish_vision_feedback', {'json': json, 'String': Record,
                             'forward_wall_clearance': forward_wall_clearance, 'tag_normal_yaw': tag_normal_yaw})
        transform = np.eye(4)
        transform[2, 3] = 0.4
        node = Record(extrinsics=CameraExtrinsics(), vision_pub=Record(publish=published.append))
        method(node, 12.5, [{'id': 3, 't_cam_tag': transform}], [3])
        payload = json.loads(published[0].data)
        self.assertEqual(payload['tags'], [[3, 0.4, 0.0]])
        self.assertEqual(payload['stamp_s'], 12.5)
        self.assertAlmostEqual(payload['forward_clearance_m'], 0.4)
        self.assertEqual(payload['tag_yaws'], [[3, 0.0]])

    def test_tf_lookup_never_substitutes_latest_for_missing_image_transform(self):
        """Missing exact-time transforms must not become fresh visual measurements.

        Args:
            None.

        Returns:
            None. Asserts that every lookup uses the original image time.
        """
        calls, stamp = [], object()
        def lookup(*args):
            calls.append(args)
            raise LookupError('Image transform unavailable')
        method = load_method('hw3/hw3/nodes/ekf_slam_node.py', 'EkfSlamNode',
                             '_lookup_tag_transform', {'TransformException': LookupError})
        self.assertIsNone(method(Record(tf_buffer=Record(lookup_transform=lookup)),
                                 'camera_frame', 1, 'tag36h11', stamp))
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(call[2] is stamp for call in calls))

    def test_quality_filter_rejects_wrong_family_and_low_margin(self):
        """Use the real quality adapter without requiring ROS packages.

        Args:
            None.

        Returns:
            None. Verifies finite quality, corrected-bit bounds, and family filtering.
        """
        method = load_method('hw3/hw3/nodes/ekf_slam_node.py', 'EkfSlamNode',
                             '_detection_quality', {'math': math})
        limits = {'quality.min_decision_margin': 20, 'quality.max_hamming': 1}
        node = Record(get_parameter=lambda name: Record(value=limits[name]))
        self.assertTrue(method(node, Record(family='tag36h11', decision_margin=50, hamming=0)))
        for family, margin, bits in [('25h9', 50, 0), ('36h11', 10, 0),
                                     ('36h11', math.nan, 0), ('36h11', 50, 2)]:
            self.assertFalse(method(node, Record(family=family, decision_margin=margin, hamming=bits)))

    def test_relative_tag_turn_rejects_face_flips_and_disagreement(self):
        self.assertAlmostEqual(relative_tag_turn({7: 0.3}, {7: 0.3 - math.pi / 2}, math.pi / 2), math.pi / 2)
        self.assertIsNone(relative_tag_turn({7: 0.3}, {7: 0.3 + math.pi / 2}, math.pi / 2))
        self.assertIsNone(relative_tag_turn({7: 0, 9: 0}, {7: -1.2, 9: -1.9}, math.pi / 2))
        self.assertIsNone(relative_tag_turn({7: 0}, {9: 0}, math.pi / 2))

    def test_tag_normal_yaw_uses_vertical_face_geometry(self):
        transform = np.eye(4)
        transform[:3, 2] = [0.6, 0.8, 0]
        self.assertAlmostEqual(tag_normal_yaw(transform), math.atan2(0.8, 0.6))
        transform[:3, 2] = [0, 0, 1]
        self.assertIsNone(tag_normal_yaw(transform))

    def test_command_yaw_uncertainty_does_not_depend_on_callback_rate(self):
        coarse, fine = EkfSlam(), EkfSlam()
        coarse.predict_commanded_velocity(0, 0.6, 1)
        for _ in range(100):
            fine.predict_commanded_velocity(0, 0.6, 0.01)
        self.assertAlmostEqual(coarse.P[2, 2], fine.P[2, 2])



class TestAppliedCommandTiming(unittest.TestCase):
    def test_stop_and_repeated_callbacks_integrate_each_interval_once(self):
        """Verify stop and repeated callbacks integrate each interval once.

        Args:
            None.

        Returns:
            None. Assertions verify the documented behavior.
        """
        path = 'hw3/hw3/nodes/ekf_slam_node.py'
        method = load_method(path, 'EkfSlamNode', '_predict_command_until')
        received = []
        node = Record(last_prediction_time=1.0, last_cmd_time=1.0,
                      command_hold_s=0.15, command_prediction=True, last_cmd=(0, 0.6),
                      ekf=Record(predict_commanded_velocity=lambda *v: received.append(v)))
        method(node, 1.05)
        method(node, 1.05)
        method(node, 1.1)
        self.assertAlmostEqual(sum(v[2] for v in received), 0.1)
        node.last_cmd, node.last_cmd_time = (0, 0), 1.1
        method(node, 1.15)
        self.assertAlmostEqual(sum(v[1] * v[2] for v in received), 0.06)

    def test_silent_commands_stop_prediction_at_motor_watchdog(self):
        """Verify silent commands stop prediction at motor watchdog.

        Args:
            None.

        Returns:
            None. Assertions verify the documented behavior.
        """
        method = load_method('hw3/hw3/nodes/ekf_slam_node.py', 'EkfSlamNode',
                             '_predict_command_until')
        received = []
        node = Record(last_prediction_time=1.0, last_cmd_time=1.0,
                      command_hold_s=0.15, command_prediction=True, last_cmd=(0, 0.6),
                      ekf=Record(predict_commanded_velocity=lambda *v: received.append(v)))
        method(node, 10.0)
        self.assertAlmostEqual(received[0][2], 0.15)
        method(node, 11.0)
        self.assertEqual(len(received), 1)


class TestFixedMapper(unittest.TestCase):
    def test_mapper_publishes_model_rate_and_exact_motor_inputs(self):
        """Verify mapper publishes model rate and exact motor inputs.

        Args:
            None.

        Returns:
            None. Assertions verify the documented behavior.
        """
        motors, applied = [], []
        twist = lambda: Record(linear=Record(x=0.0), angular=Record(z=0.0))
        method = load_method('hw3/hw3/nodes/velocity_mapping.py', 'VelocityToMotorNode',
                             'on_command', {'Twist': twist, 'Float32MultiArray': Record})
        node = Record(mapper=VelocityMapper(TurnCalibration(-0.3, 0.3, 0.3, -0.3, 0.6, 0.5)),
                      motor_pub=Record(publish=motors.append),
                      applied_pub=Record(publish=applied.append))
        for request in (0.01, -0.01, 0):
            message = twist()
            message.angular.z = request
            method(node, message)
        self.assertEqual([m.data for m in motors], [[-0.3, 0.3], [0.3, -0.3], [0, 0]])
        self.assertEqual([m.angular.z for m in applied], [0.6, -0.5, 0])

    def test_forward_motion_uses_fixed_trimmed_inputs_at_all_requested_speeds(self):
        mapper = VelocityMapper(TurnCalibration(-0.3, 0.3, 0.3, -0.3, 0.6, 0.5))
        for requested in (0.01, 0.12, 0.203, 10):
            command = mapper.map_command(requested, 0.7)
            self.assertEqual((command.left, command.right), (0.185, 0.15))
            self.assertEqual(command.angular_radps, 0)
            self.assertEqual(command.linear_mps, 0.203)

    def test_nonfinite_or_uncalibrated_reverse_requests_stop(self):
        mapper = VelocityMapper(TurnCalibration(-0.3, 0.3, 0.3, -0.3, 0.6, 0.5))
        for linear, angular in [(math.nan, 1), (1, math.inf), (-0.203, 0), (0, 0)]:
            command = mapper.map_command(linear, angular)
            self.assertEqual((command.left, command.right, command.linear_mps, command.angular_radps),
                             (0, 0, 0, 0))

    def test_measured_speed_matches_nominal_three_foot_command_time(self):
        """Verify measured speed matches nominal three foot command time.

        Args:
            None.

        Returns:
            None. Assertions verify measured calibration and controller behavior.
        """
        nav = SquareNavigator(turn_calibration=TurnCalibration(-0.3, 0.3, 0.3, -0.3, 0.6, 0.5))
        nav.start()
        commanded_distance = 0.0
        previous = 0.0
        for tick in range(150):
            commanded_distance += previous * 0.05
            cmd = nav.step(1 + tick * 0.05)
            self.assertIn(cmd.linear_v_mps, (0, 0.203))
            previous = cmd.linear_v_mps
            if nav.state == NavigationState.MIDPOINT_PAUSE:
                break
        self.assertEqual(nav.state, NavigationState.MIDPOINT_PAUSE)
        self.assertAlmostEqual(commanded_distance, 0.9144, delta=0.203 * 0.05)
        self.assertAlmostEqual(0.9144 / nav.forward.speed_mps, 4.5044334975)

    def test_mapper_publishes_actual_calibrated_speed_for_ekf(self):
        """The applied command must describe the measured motor pair, not the request.

        Args:
            None.

        Returns:
            None. Confirms the EKF receives 0.203 m/s for the exact trimmed motor inputs.
        """
        motors, applied = [], []
        twist = lambda: Record(linear=Record(x=0.0), angular=Record(z=0.0))
        method = load_method('hw3/hw3/nodes/velocity_mapping.py', 'VelocityToMotorNode',
                             'on_command', {'Twist': twist, 'Float32MultiArray': Record})
        node = Record(mapper=VelocityMapper(TurnCalibration(-0.3, 0.3, 0.3, -0.3, 0.6, 0.5)),
                      motor_pub=Record(publish=motors.append), applied_pub=Record(publish=applied.append))
        request = twist()
        request.linear.x = 0.05
        method(node, request)
        self.assertEqual(motors[0].data, [0.185, 0.15])
        self.assertEqual(applied[0].linear.x, 0.203)

    def test_forward_calibration_is_shared_and_validated(self):
        """Verify forward calibration is shared and validated.

        Args:
            None.

        Returns:
            None. Assertions verify measured calibration and controller behavior.
        """
        values = {'forward_calibration.left_input': 0.2, 'forward_calibration.right_input': 0.16,
                  'forward_calibration.speed_mps': 0.25}
        node = Record(declare_parameter=lambda name, default: Record(value=values.get(name, default)))
        calibration = read_forward_calibration(node)
        mapper = VelocityMapper(TurnCalibration(-0.3, 0.3, 0.3, -0.3, 0.6, 0.5), calibration)
        nav = SquareNavigator(forward_calibration=calibration)
        self.assertEqual(mapper.map_command(0.1, 0).linear_mps, nav.forward.speed_mps)
        self.assertEqual(nav.forward.speed_mps, 0.25)
        for values in [(0.6, 0.15, 0.203), (0.18, 0.15, 0), (0.18, math.nan, 0.203)]:
            with self.assertRaises(ValueError):
                ForwardCalibration(*values)


class TestHw3SerialDriver(unittest.TestCase):
    def test_motor_watchdog_expires_without_a_new_command(self):
        method = load_method('hw3/hw3/nodes/motor_control.py', 'MotorControllerNode',
                             '_current_command')
        node = Record(command_lock=threading.Lock(), timeout_duration=0.15,
                      last_command_time=1.0, motor_command={'T': 1, 'L': -0.3, 'R': 0.3})
        self.assertEqual(method(node, 1.1), node.motor_command)
        self.assertEqual(method(node, 1.2), {'T': 1, 'L': 0, 'R': 0})

    def test_serial_packet_retains_existing_protocol(self):
        method = load_method('hw3/hw3/nodes/motor_control.py', 'MotorControllerNode',
                             '_write', {'json': json})
        sent = []
        method(Record(robot=Record(write=sent.append)), {'T': 1, 'L': -0.3, 'R': 0.3})
        self.assertTrue(sent[0].endswith(b'\n'))
        self.assertEqual(json.loads(sent[0]), {'T': 1, 'L': -0.3, 'R': 0.3})
