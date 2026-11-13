"""Regression checks for calibrated mapping, bounded pivots, and command timing."""

import ast
import math
import json
import threading
from pathlib import Path
from types import SimpleNamespace as Record
import unittest

import numpy as np

from hw3.turn_calibration import TurnCalibration
from hw3.velocity_mapping import VelocityMapper
from hw3.vision_navigation import NavigationPolicy, VisualFeedback, forward_wall_clearance
from hw3.ekf_slam import EkfSlam
from hw3.square_navigation import FixedTurnPulse, NavigationState, SquareNavigator


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
        self.calibration = TurnCalibration(0.3, 0.3, 0.6, 0.5)

    def test_direction_specific_fixed_rates_and_inputs(self):
        for request in (0.001, 0.2, 2.0):
            self.assertEqual(self.calibration.rate(request), 0.6)
            self.assertEqual(self.calibration.motors(request), (-0.3, 0.3))
            self.assertEqual(self.calibration.rate(-request), -0.5)
            self.assertEqual(self.calibration.motors(-request), (0.3, -0.3))
        self.assertEqual(self.calibration.motors(0), (0, 0))

    def test_measured_direction_specific_motor_inputs(self):
        calibration = TurnCalibration(0.43, 0.40, 2 * math.pi / 3.55, 2 * math.pi / 3.59)
        mapper = VelocityMapper(calibration)
        ccw, cw = mapper.map_command(0, 1), mapper.map_command(0, -1)
        self.assertEqual((ccw.left, ccw.right), (-0.43, 0.43))
        self.assertEqual((cw.left, cw.right), (0.40, -0.40))
        self.assertAlmostEqual(ccw.angular_radps * 3.55, 2 * math.pi)
        self.assertAlmostEqual(cw.angular_radps * 3.59, -2 * math.pi)

    def test_unmeasured_or_invalid_calibration_is_rejected(self):
        for values in [(0.3, 0.3, 0, 0), (0.6, 0.3, 0.5, 0.5),
                       (0.3, 0.6, 0.5, 0.5), (0.3, 0.3, -1, 1), (0.3, 0.3, math.nan, 1)]:
            with self.assertRaises(ValueError):
                TurnCalibration(*values)
        nav = SquareNavigator()
        self.assertFalse(nav.start()[0])

    def test_pulse_stops_and_settles_before_restarting(self):
        pulse = FixedTurnPulse(self.calibration, math.pi / 2, 0.3, 1.0)
        self.assertEqual(pulse.step(1.0, math.pi / 2, 0.1)[0], 0.6)
        self.assertEqual(pulse.step(1.5, math.pi / 2, 0.1)[0], 0)
        self.assertEqual(pulse.step(2.4, math.pi / 2, 0.1)[0], 0)
        self.assertEqual(pulse.step(2.5, math.pi / 2, 0.1)[0], 0.6)

    def test_frozen_heading_cannot_command_a_full_revolution(self):
        pulse = FixedTurnPulse(self.calibration, math.pi / 2, math.pi / 6, 0.1)
        for tick in range(300):
            pulse.step(1 + tick * 0.05, math.pi / 2, 0.1)
            if pulse.exhausted:
                break
        self.assertTrue(pulse.exhausted)
        self.assertLess(pulse.used, math.radians(125))

    def test_feedback_stops_before_pulse_deadline(self):
        pulse = FixedTurnPulse(self.calibration, 1.0, 0.5, 0.2)
        pulse.step(1.0, 1.0, 0.1)
        self.assertEqual(pulse.step(1.1, 0.01, 0.1)[0], 0)
        self.assertEqual(pulse.step(1.4, 0.01, 0.1)[0], 0)
        self.assertTrue(pulse.done)

    def feed_vision(self, nav, now, pose, clearance=None):
        """Supply two fixed synthetic landmarks in the moving robot base frame."""
        x, y, yaw = pose
        c, s = math.cos(yaw), math.sin(yaw)
        tags = [[tag, c * (tx - x) + s * (ty - y), -s * (tx - x) + c * (ty - y)]
                for tag, tx, ty in [(1, 5.0, 4.0), (3, -2.0, 3.0)]]
        nav.update_vision(now, tags, [1, 3], clearance)

    def test_first_leg_requires_no_tags_and_reaches_scan_with_uncertain_pose(self):
        """An empty map cannot block the bounded first leg or its following scan.

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
            if cmd.state == NavigationState.SCANNING:
                break
        self.assertEqual(nav.state, NavigationState.SCANNING, nav.stop_reason)
        self.assertAlmostEqual(nav.drive_used, nav.square_size_m, delta=0.11)
        self.assertGreater(nav.step(now + 0.05).angular_omega_radps, 0)
        self.assertEqual(nav.vision.valid_stamp, -math.inf)

    def test_new_tag_can_stop_first_leg_for_close_wall(self):
        nav = self.prepared(NavigationState.DRIVING)
        nav.update_vision(1, [[99, 0.3, 0]], [], 0.3)
        cmd = nav.step(1)
        self.assertEqual(cmd.linear_v_mps, 0)
        self.assertIn('clearance', nav.stop_reason)

    def test_square_has_four_scans_and_fixed_headings(self):
        """Simulate four visually observed legs and scans, including the final heading.

        Args:
            None.

        Returns:
            None. Verifies sequence, heading, command separation, and approximate closure.
        """
        nav = SquareNavigator(turn_calibration=self.calibration, turn_settle_s=0.1)
        nav.start()
        x = y = yaw = 0.0
        states, previous, scans = set(), None, 0
        for tick in range(6000):
            now = 1 + tick * 0.05
            nav.update_ekf_pose(x, y, yaw, 0.01, 0.01, now)
            self.feed_vision(nav, now, (x, y, yaw))
            cmd = nav.step(now)
            scans += int(cmd.state == NavigationState.SCANNING and previous != cmd.state)
            previous = cmd.state
            states.add(cmd.state)
            self.assertFalse(cmd.linear_v_mps and cmd.angular_omega_radps)
            x += cmd.linear_v_mps * math.cos(yaw) * 0.05
            y += cmd.linear_v_mps * math.sin(yaw) * 0.05
            yaw += cmd.angular_omega_radps * 0.05
            if not nav.is_active:
                break
        self.assertEqual(nav.state, NavigationState.COMPLETED, nav.stop_reason)
        self.assertEqual(scans, 4)
        self.assertIn(NavigationState.LOCALIZING, states)
        self.assertLess(abs(math.atan2(math.sin(yaw), math.cos(yaw))), nav.heading_tolerance_rad)
        self.assertLess(math.hypot(x, y), 0.4)

    def test_complete_square_with_no_tags_and_unreliable_ekf(self):
        """Run all four sides and scans without any visual observations.

        Args:
            None.

        Returns:
            None. Verifies completion, no tag waits, four scans, and nominal command closure.
        """
        nav = SquareNavigator(turn_calibration=self.calibration, turn_settle_s=0.1)
        nav.start()
        x = y = yaw = 0.0
        scans, previous = 0, None
        for tick in range(8000):
            now = 1 + tick * 0.05
            nav.update_ekf_pose(0, 0, 0, 10, 10, now)
            cmd = nav.step(now)
            scans += int(cmd.state == NavigationState.SCANNING and previous != cmd.state)
            previous = cmd.state
            self.assertFalse(cmd.linear_v_mps and cmd.angular_omega_radps)
            x += cmd.linear_v_mps * math.cos(yaw) * 0.05
            y += cmd.linear_v_mps * math.sin(yaw) * 0.05
            yaw += cmd.angular_omega_radps * 0.05
            if not nav.is_active:
                break
        self.assertEqual(nav.state, NavigationState.COMPLETED, nav.stop_reason)
        self.assertEqual(scans, 4)
        self.assertEqual(nav.vision.valid_stamp, -math.inf)
        self.assertLess(math.hypot(x, y), 0.4)
        self.assertLess(abs(math.atan2(math.sin(yaw), math.cos(yaw))), nav.heading_tolerance_rad)

    def test_one_known_tag_can_correct_heading_without_a_tag_count_gate(self):
        nav = self.prepared(NavigationState.ALIGNING)
        nav.side_index = 1
        nav.update_vision(1, [[7, 2, 0]], [7], None)
        nav.update_ekf_pose(0, 0, math.pi / 2, 0.01, 0.01, 1.05)
        self.assertAlmostEqual(nav.navigation_yaw, math.pi / 2)
        self.assertEqual(nav.step(1.05).state, NavigationState.DRIVING)

    def prepared(self, state):
        """Create an active controller with a known initial pose and requested phase."""
        nav = SquareNavigator(turn_calibration=self.calibration, turn_settle_s=0.1)
        nav.start()
        nav.update_ekf_pose(0, 0, 0, 0.01, 0.01, 1)
        nav._transition(state, 1)
        return nav

    def test_stale_pose_stops_scan_and_cannot_resume(self):
        nav = self.prepared(NavigationState.SCANNING)
        self.assertGreater(nav.step(1.05).angular_omega_radps, 0)
        self.assertEqual(nav.step(2).angular_omega_radps, 0)
        self.assertEqual(nav.state, NavigationState.STOPPED)
        self.assertFalse(nav.start()[0])

    def test_inconsistent_visual_heading_has_bounded_best_effort_alignment(self):
        """Conflicting heading feedback cannot block the next leg indefinitely.

        Args:
            None.

        Returns:
            None. Verifies bounded correction angle and explicit prediction-only continuation.
        """
        nav = self.prepared(NavigationState.ALIGNING)
        nav.side_index = 1
        for tick in range(500):
            now = 1 + tick * 0.05
            nav.update_ekf_pose(0, 0, 0, 0.01, 0.01, now)
            self.feed_vision(nav, now, (0, 0, 0))
            result = nav.step(now)
            if nav.state == NavigationState.DRIVING:
                break
        self.assertEqual(nav.state, NavigationState.DRIVING)
        self.assertEqual(result.angular_omega_radps, 0)
        self.assertIn('best effort', result.status_message)

    def test_scan_has_eight_45_degree_stages_and_half_second_pauses(self):
        """Check the actual scan commands at the robot's nominal 20 Hz control rate.

        Args:
            None.

        Returns:
            None. Verifies eight bounded turns, eight pauses, and final stopped completion.
        """
        nav = self.prepared(NavigationState.SCANNING)
        nav.turn_calibration = TurnCalibration(0.43, 0.4, 2 * math.pi / 3.55, 2 * math.pi / 3.59)
        previous, angle, pause_start = 0.0, 0.0, None
        angles, pauses = [], []
        for tick in range(400):
            now = 1 + tick * 0.05
            angle += previous * 0.05
            nav.update_ekf_pose(0, 0, 0, 0.01, 0.01, now)
            command = nav.step(now)
            omega = command.angular_omega_radps
            self.assertEqual(command.linear_v_mps, 0)
            if previous and not omega:
                angles.append(angle)
                angle, pause_start = 0.0, now
            if pause_start is not None and (omega or command.state == NavigationState.LOCALIZING):
                pauses.append(now - pause_start)
                pause_start = None
            previous = omega
            if command.state == NavigationState.LOCALIZING:
                break
        self.assertEqual(nav.state, NavigationState.LOCALIZING)
        self.assertEqual(len(angles), 8)
        self.assertEqual(len(pauses), 8)
        self.assertTrue(all(pause >= 0.5 - 1e-8 for pause in pauses))
        self.assertTrue(all(abs(value - math.pi / 4) <= nav.turn_calibration.ccw_radps * 0.05
                            for value in angles))
        self.assertLessEqual(sum(angles), 2 * math.pi + nav.turn_calibration.ccw_radps * 0.05)

    def test_scan_timeout_includes_stopped_camera_pauses(self):
        """A stage pause cannot extend the overall scan deadline.

        Args:
            None.

        Returns:
            None. Verifies zero commands after timing out during a camera pause.
        """
        nav = self.prepared(NavigationState.SCANNING)
        nav.policy = NavigationPolicy(scan_timeout_s=1.5)
        for tick in range(40):
            now = 1 + tick * 0.05
            nav.update_ekf_pose(0, 0, 0, 0.01, 0.01, now)
            command = nav.step(now)
            if not nav.is_active:
                break
        self.assertGreaterEqual(nav.scan_stage_index, 1)
        self.assertEqual(nav.state, NavigationState.STOPPED)
        self.assertEqual(command.angular_omega_radps, 0)
        self.assertIn('time limit', nav.stop_reason)

    def test_scan_and_alignment_continue_without_vision(self):
        """A tag-free scan proceeds through finite settling and calibrated alignment.

        Args:
            None.

        Returns:
            None. Checks that absent observations never prevent the second side.
        """
        nav = self.prepared(NavigationState.SCANNING)
        for tick in range(800):
            now = 1 + tick * 0.05
            nav.update_ekf_pose(0, 0, 0, 10, 10, now)
            cmd = nav.step(now)
            if nav.state == NavigationState.DRIVING:
                break
        self.assertEqual(nav.state, NavigationState.DRIVING, nav.stop_reason)
        self.assertEqual(nav.side_index, 1)
        self.assertLess(nav.scan_used, 2 * math.pi + 0.04)
        self.assertEqual(nav.vision.valid_stamp, -math.inf)
        nav.update_ekf_pose(0, 0, 0, 10, 10, now + 0.05)
        self.assertGreater(nav.step(now + 0.05).linear_v_mps, 0)

    def test_drive_stops_for_wall_and_does_not_pivot_mid_side(self):
        nav = self.prepared(NavigationState.DRIVING)
        self.feed_vision(nav, 1, (0, 0, 0), clearance=0.3)
        cmd = nav.step(1)
        self.assertEqual((cmd.linear_v_mps, cmd.angular_omega_radps), (0, 0))
        self.assertIn('clearance', nav.stop_reason)

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
            None. Verifies a bounded leg followed by the next scan phase.
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

    def test_visual_progress_overrides_bad_ekf_distance(self):
        nav = self.prepared(NavigationState.DRIVING)
        nav.vision.begin_leg(0)
        nav.update_vision(1, [[1, 3, 0], [3, 3, 1]], [1, 3], None)
        nav.update_ekf_pose(0.1, 0, 0, 0.01, 0.01, 1.1)
        nav.update_vision(1.1, [[1, 1.2, 0], [3, 1.2, 1]], [1, 3], None)
        cmd = nav.step(1.1)
        self.assertAlmostEqual(nav.progress, 1.8)
        self.assertEqual(cmd.linear_v_mps, 0)
        self.assertEqual(nav.state, NavigationState.STOPPING)


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
                             'forward_wall_clearance': forward_wall_clearance})
        transform = np.eye(4)
        transform[2, 3] = 0.4
        node = Record(extrinsics=CameraExtrinsics(), vision_pub=Record(publish=published.append))
        method(node, 12.5, [{'id': 3, 't_cam_tag': transform}], [3])
        payload = json.loads(published[0].data)
        self.assertEqual(payload['tags'], [[3, 0.4, 0.0]])
        self.assertEqual(payload['stamp_s'], 12.5)
        self.assertAlmostEqual(payload['forward_clearance_m'], 0.4)

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
        node = Record(mapper=VelocityMapper(TurnCalibration(0.3, 0.3, 0.6, 0.5)),
                      motor_pub=Record(publish=motors.append),
                      applied_pub=Record(publish=applied.append))
        for request in (0.01, -0.01, 0):
            message = twist()
            message.angular.z = request
            method(node, message)
        self.assertEqual([m.data for m in motors], [[-0.3, 0.3], [0.3, -0.3], [0, 0]])
        self.assertEqual([m.angular.z for m in applied], [0.6, -0.5, 0])

    def test_straight_motion_uses_existing_linear_calibration(self):
        mapper = VelocityMapper(TurnCalibration(0.3, 0.3, 0.6, 0.5))
        command = mapper.map_command(0.2, 0.7)
        self.assertAlmostEqual(command.left, 0.08 + 0.2 / 1.772915625)
        self.assertEqual(command.left, command.right)
        self.assertEqual(command.angular_radps, 0)
        self.assertAlmostEqual(command.linear_mps, 0.2)

    def test_saturation_and_nonfinite_commands(self):
        mapper = VelocityMapper(TurnCalibration(0.3, 0.3, 0.6, 0.5))
        command = mapper.map_command(10, 0)
        self.assertEqual(command.left, 0.5)
        self.assertAlmostEqual(command.linear_mps, (0.5 - 0.08) * 1.772915625)
        self.assertEqual(mapper.map_command(math.nan, 1).left, 0)
        self.assertEqual(mapper.map_command(1, math.inf).right, 0)


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
