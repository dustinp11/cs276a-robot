"""Run-1 checks with synthetic geometry and ideal pose feedback, never hardware."""

from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import yaml

# resolve both packages without requiring ros
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent / 'hw3')]
from hw3.coordinates import CameraExtrinsics, make_transform, euler_to_rotation_matrix
from hw3.turn_calibration import ForwardCalibration, TurnCalibration
from hw3.velocity_mapping import VelocityMapper
from hw4.config import load_validated_environment, validate_environment
from hw4.grid import build_grid, rectangle_distance
from hw4.localization import Feedback, Localizer, stable_feedback, select_inliers
from hw4.navigation import Navigator
from hw4.observations import CaptureBuffer
from hw4.tags import canonical_family, detection_topic, detector_parameters, mapped_families, tag_frame
from hw4.fast import plan_path
from hw4.planner import get_mode
from hw4.preview import prepare_plans
from hw4.results import RunLog, path_metrics, save_plans
from hw4.plan_plot import map_svg, plan_description, plan_segments
from hw4.settings import load_settings


def calibrations():
    """Load the actual HW4 shared motor configuration for every synthetic run."""
    params = yaml.safe_load((ROOT / 'config/motion.yaml').read_text())['/**']['ros__parameters']
    forward = ForwardCalibration(**params['forward_calibration'])
    turns = {key: value for key, value in params['turn_calibration'].items() if key != 'enabled'}
    return forward, TurnCalibration(**turns)


def simulate(navigator, pose, duration=240.0, feedback=True, linear_scale=1.0):
    """Integrate perfect motor response and 10 Hz exact observations at 20 Hz control."""
    command, samples, minimum = (0.0, 0.0), [], math.inf
    for index in range(int(duration / 0.05)):
        now = 10.0 + index * 0.05
        x, y, yaw = pose
        x += command[0] * linear_scale * 0.05 * math.cos(yaw)
        y += command[0] * linear_scale * 0.05 * math.sin(yaw)
        yaw += command[1] * 0.05
        pose = x, y, yaw
        minimum = min(minimum, navigator.grid.clearance((x, y)))

        # feed ideal samples with original timestamps
        if feedback and index % 2 == 0:
            samples.append(Feedback(x, y, yaw, now, (0,)))
            samples = samples[-30:]
        command = navigator.tick(now, samples)
        if navigator.finished:
            return navigator.state, navigator.reason, pose, minimum

    return 'timeout', navigator.reason, pose, minimum


class RunOneTests(unittest.TestCase):
    """Exercise algorithm behavior and explicit failure cases without ROS."""

    @classmethod
    def setUpClass(cls):
        """Build shared synthetic plans once for the controller checks."""
        # share synthetic plans across controller checks
        cls.environment = load_validated_environment(ROOT / 'test/sample_environment.json')
        cls.settings = load_settings(ROOT / 'config/navigation.yaml')
        cls.policy = dict(cls.settings['navigation'], feedback_required=True, visual_feedback='throughout', tag_search_enabled=True)
        cls.grid, cls.plans = prepare_plans(cls.environment, cls.policy)
        cls.forward, cls.turn = calibrations()

    def test_config_rejects_unknown_invalid_and_duplicate_measurements(self):
        """Do not replace missing or physically invalid measurements with defaults."""
        # reject unknown values before accepting geometry
        for path, value in [(('robot', 'length_m'), None), (('robot', 'width_m'), -1),
                            (('start', 'x_m'), True), (('goal', 'y_m'), math.nan)]:
            data = deepcopy(self.environment)
            data[path[0]][path[1]] = value
            with self.assertRaises(ValueError):
                validate_environment(data)

        data = deepcopy(self.environment)
        data['landmarks'].append(data['landmarks'][0])
        with self.assertRaises(ValueError):
            validate_environment(data)

    def test_continuous_collision_checks(self):
        """Reject crossing segments even when both endpoints are outside the obstacle."""
        # check complete swept paths through obstacles
        self.assertFalse(self.grid.segment_free((0.5, 1.5), (2.5, 1.5)))
        self.assertFalse(self.grid.segment_free((0.05, 0.5), (0.05, 2.5)))
        self.assertTrue(self.grid.segment_free((0.55, 0.55), (0.55, 2.5)))
        self.assertEqual(rectangle_distance((0, 0), (2, 2), (0.8, 1.2, 0.8, 1.2)), 0)
        self.assertAlmostEqual(rectangle_distance((0, 0), (0, 2), (1, 2, 1, 2)), 1)

    def test_fast_path_preserves_endpoints_and_clearance(self):
        """Keep the implemented fast route collision-free after selecting its module."""
        # check selected mode against physical geometry
        self.assertEqual(set(self.plans), {'fast'})
        path = self.plans['fast']
        self.assertEqual(path[0], (0.55, 0.55))
        self.assertEqual(path[-1], (2.5, 2.5))
        self.assertTrue(all(self.grid.segment_free(a, b) for a, b in zip(path, path[1:])))
        self.assertGreaterEqual(path_metrics(self.grid, path)['min_body_clearance_m'], self.grid.margin)

    def test_safe_selection_reports_unimplemented_without_fallback(self):
        """A safe selection must never silently run the fast behavior."""
        with self.assertRaises(NotImplementedError):
            get_mode('safe')
        with self.assertRaises(NotImplementedError):
            prepare_plans(self.environment, dict(self.policy, mode='safe'))
        with self.assertRaises(ValueError):
            get_mode('unknown')

    def test_unreachable_goal_and_unsafe_endpoints(self):
        """Return failure rather than fabricate a route through a dividing obstacle."""
        # reject disconnected and occupied endpoint queries
        data = deepcopy(self.environment)
        data['obstacles'][0].update(x_min_m=1.2, x_max_m=1.8, y_min_m=0, y_max_m=3.048)
        with self.assertRaises(ValueError):
            plan_path(build_grid(data, 0.1), (0.55, 0.55), (2.5, 2.5))
        with self.assertRaises(ValueError):
            plan_path(self.grid, (1.5, 1.5), (2.5, 2.5))

    def test_known_map_transform_recovers_nonzero_camera_offset(self):
        """Recover world pose using independently composed synthetic camera observations."""
        # compose independent camera observations from geometry
        mount = dict(x_m=0.05, y_m=-0.02, z_m=0.21, roll_deg=0., pitch_deg=0., yaw_deg=5.)
        localizer = Localizer(self.environment, mount, self.settings['localization'])
        base = make_transform(euler_to_rotation_matrix(0, 0, 0.1), [0.55, 0.55, 0])
        camera = base @ CameraExtrinsics(**mount).t_base_cam
        measurements = [(tag, np.linalg.inv(camera) @ world) for tag, world in localizer.landmarks.items()]
        pose = localizer.estimate(measurements, 10.0)
        np.testing.assert_allclose((pose.x, pose.y, pose.yaw), (0.55, 0.55, 0.1), atol=1e-10)
        self.assertEqual(pose.stamp, 10.0)

    def test_two_conflicting_tags_have_no_majority(self):
        """Reject two conflicting tags instead of arbitrarily trusting either one."""
        # introduce conflicting independently measured tag solutions
        localizer = Localizer(self.environment, self.settings['camera_mount'], self.settings['localization'])
        measurements = []
        for index, (tag, world) in enumerate(localizer.landmarks.items()):
            base = make_transform(np.eye(3), [0.55 + index * 0.6, 0.55, 0])
            measurements.append((tag, np.linalg.inv(base @ np.linalg.inv(localizer.camera_base)) @ world))
        with self.assertRaisesRegex(ValueError, 'no unique agreeing majority'):
            localizer.estimate(measurements, 10.0)

    def test_family_ids_and_detector_spelling_remain_distinct(self):
        """Allow overlapping IDs while preserving unique map and detector identities."""
        # preserve identity even with equal numbers
        data = deepcopy(self.environment)
        obstacle_tag = dict(data['landmarks'][1], family='41h12', x_m=1.14, y_m=1.54,
                            z_m=0.187, face_yaw_deg=180.0)
        data['landmarks'].append(obstacle_tag)
        validate_environment(data)
        model = Localizer(data, self.settings['camera_mount'], self.settings['localization'])
        self.assertIn(('36h11', 1), model.landmarks)
        self.assertIn(('41h12', 1), model.landmarks)
        self.assertNotEqual(tag_frame(('36h11', 1)), tag_frame(('41h12', 1)))
        self.assertEqual(canonical_family('tagStandard41h12'), '41h12')
        self.assertEqual(mapped_families(data), ['36h11', '41h12'])

        # match the actual detector family spelling
        params = detector_parameters(data, '41h12')
        self.assertEqual(params['family'], 'Standard41h12')
        self.assertEqual(params['tag.ids'], [1])
        self.assertEqual(params['tag.frames'], ['tag_41h12_1'])
        self.assertEqual(params['tag.sizes'], [0.172])
        data['landmarks'].append(obstacle_tag)
        with self.assertRaises(ValueError):
            validate_environment(data)

    def test_both_families_recover_the_same_measured_pose(self):
        """Use overlapping IDs from both families in one independent synthetic capture."""
        # create two distinct physical tag identities
        data = deepcopy(self.environment)
        data['landmarks'].append(dict(data['landmarks'][1], family='41h12', x_m=1.14,
                                      y_m=1.54, z_m=0.187, face_yaw_deg=180.0))
        model = Localizer(data, self.settings['camera_mount'], self.settings['localization'])
        base = make_transform(euler_to_rotation_matrix(0, 0, 0.2), [0.55, 0.55, 0])
        camera = base @ np.linalg.inv(model.camera_base)
        measurements = [(key, np.linalg.inv(camera) @ world) for key, world in model.landmarks.items()]
        pose = model.estimate(measurements, 10.0)
        np.testing.assert_allclose((pose.x, pose.y, pose.yaw), (0.55, 0.55, 0.2), atol=1e-10)
        self.assertIn(('36h11', 1), pose.tags)
        self.assertIn(('41h12', 1), pose.tags)

    def test_detection_topic_tokens_are_valid_ros_identifiers(self):
        """Regress the physical-run failure caused by digit-prefixed topic segments."""
        # validate generated names used across adapters
        self.assertEqual(detection_topic('36h11'), '/detections/tag36h11')
        self.assertEqual(detection_topic('tagStandard41h12'), '/detections/tag41h12')
        for family in ('36h11', '41h12'):
            for token in detection_topic(family).split('/')[1:]:
                self.assertRegex(token, r'^[A-Za-z_][A-Za-z0-9_]*$')

    def test_detector_packets_count_as_one_capture(self):
        """Merge same-image packets without manufacturing independent stability samples."""
        # wait for matching family capture packets
        frames = CaptureBuffer()
        frames.add(100, 'camera_frame', '36h11', 'walls', 10.0)
        self.assertIsNone(frames.ready(10.05, ['36h11', '41h12']))
        frames.add(100, 'camera_frame', '41h12', 'obstacle', 10.06)
        capture = frames.ready(10.06, ['36h11', '41h12'])
        self.assertEqual(capture.stamp_ns, 100)
        self.assertEqual(set(capture.messages), {'36h11', '41h12'})
        frames.finish(100)
        frames.add(100, 'camera_frame', '41h12', 'duplicate', 10.07)
        self.assertIsNone(frames.ready(10.5, ['36h11', '41h12']))

        # permit one healthy detector after timeout
        frames.add(200, 'camera_frame', '41h12', 'obstacle only', 11.0)
        self.assertIsNotNone(frames.ready(11.2, ['36h11', '41h12']))
        frames.finish(200)
        for index in range(10):
            frames.add(300 + index, 'camera_frame', '36h11', 'packet', 12.0)
        self.assertLessEqual(len(frames.pending), 5)

    def test_stability_requires_distinct_fresh_post_stop_frames(self):
        """Repeated stale frames and pre-settle images do not unlock motion."""
        # exclude repeated stale and pre-settle samples
        pose = Feedback(0.55, 0.55, 0, 10.0)
        self.assertIsNone(stable_feedback([pose] * 3, 9.0, 10.0, self.policy))
        poses = [Feedback(0.55, 0.55, 0, stamp) for stamp in (9.8, 9.9, 10.0)]
        self.assertIsNone(stable_feedback(poses, 9.85, 10.0, self.policy))
        self.assertIsNone(stable_feedback(poses, 9.0, 12.0, self.policy))
        self.assertIsNotNone(stable_feedback(poses, 9.0, 10.0, self.policy))

    def test_no_motion_without_feedback_and_no_false_arrival(self):
        """Missing initial localization ends in failure without advancing targets."""
        # keep targets unchanged when feedback fails
        nav = Navigator(self.grid, self.plans['fast'], self.forward, self.turn, self.policy, 10)
        self.assertEqual(nav.tick(10.1, []), (0, 0))
        self.assertEqual(nav.tick(19, []), (0, 0))
        self.assertEqual(nav.state, 'failed')
        self.assertEqual(nav.index, 1)

    def test_stale_vision_and_timer_overrun_stop_active_segments(self):
        """Active motion cannot continue after missing feedback or a watchdog overrun."""
        # exercise both active motion failure conditions
        for now, samples in [(10.05, []), (10.25, [Feedback(0.55, 0.55, 0, 10.25)])]:
            nav = Navigator(self.grid, self.plans['fast'], self.forward, self.turn, self.policy, 10)
            nav.begin_segment(10, Feedback(0.55, 0.55, 0, 10), self.forward.speed_mps, 0, 1, 'test straight segment')
            self.assertEqual(nav.tick(now, samples), (0, 0))
            self.assertEqual(nav.state, 'failed')
            self.assertEqual(nav.index, 1)

    def test_continuous_turn_faces_full_plan_then_drives_without_pause(self):
        """Keep turning through intermediate angles and drive immediately when aligned."""
        # verify full turns across angular wrapping
        for direction, origin in [(1.0, math.radians(179)), (-1.0, math.radians(-179))]:
            desired = origin + direction * math.pi / 2
            target = 0.55 + math.cos(desired) * 0.5, 1.0 + math.sin(desired) * 0.5
            path = [(0.55, 1.0), target]
            nav = Navigator(self.grid, path, self.forward, self.turn, self.policy, 10)
            pose = Feedback(0.55, 1.0, origin, 10)
            command = nav.begin_turn(10, pose, direction * math.pi / 2)
            self.assertAlmostEqual(nav.turn_angle, math.pi / 2)
            for index, degrees in enumerate((15, 30, 60), 1):
                sample = Feedback(0.55, 1.0, origin + direction * math.radians(degrees), 10 + index * 0.1)
                self.assertEqual(nav.tick(10 + index * 0.1, [sample]), command)
                self.assertEqual(nav.state, 'turning')
            sample = Feedback(0.55, 1.0, desired, 10.4)
            self.assertEqual(nav.tick(10.4, [sample]), (self.forward.speed_mps, 0))
            self.assertEqual(nav.state, 'moving')
            self.assertEqual(nav.index, 1)

    def test_continuous_turn_does_not_use_timer_as_measured_arrival(self):
        """Fresh unchanged headings keep rotation active until the failure timeout."""
        # require physical progress instead of elapsed time
        nav = Navigator(self.grid, self.plans['fast'], self.forward, self.turn, self.policy, 10)
        pose = Feedback(0.55, 0.55, 0, 10)
        command = nav.begin_turn(10, pose, math.pi / 2)
        for index in range(1, int(self.policy['turn_timeout_s'] / 0.05) + 1):
            now = 10 + index * 0.05
            result = nav.tick(now, [Feedback(0.55, 0.55, 0, now)])
            if now < 10 + self.policy['turn_timeout_s']:
                self.assertEqual(result, command)
                self.assertEqual(nav.state, 'turning')
        self.assertEqual(nav.state, 'failed')
        self.assertIn('timed out', nav.reason)
        self.assertEqual(nav.index, 1)

    def test_continuous_turn_stops_when_heading_feedback_expires(self):
        """Do not finish a continuous turn blindly after losing camera feedback."""
        # retain original capture age during rotation
        nav = Navigator(self.grid, self.plans['fast'], self.forward, self.turn, self.policy, 10)
        pose = Feedback(0.55, 0.55, 0, 10)
        nav.begin_turn(10, pose, math.pi / 2)
        for index in range(1, 10):
            nav.tick(10 + index * 0.05, [pose])
        self.assertEqual(nav.state, 'failed')
        self.assertIn('heading feedback unavailable', nav.reason)
        self.assertEqual(nav.command, (0, 0))

    def test_forward_travel_lasts_one_second_before_checking(self):
        """Keep driving for a full second in open space before stopped acquisition."""
        # retain movement throughout the requested duration
        path = [(0.55, 0.55), (0.55, 2.5)]
        nav = Navigator(self.grid, path, self.forward, self.turn, self.policy, 10)
        pose = Feedback(0.55, 0.55, math.pi / 2, 10)
        command = nav.drive_segment(10, pose, path[-1])
        self.assertEqual(command, (self.forward.speed_mps, 0))
        self.assertAlmostEqual(nav.deadline, 11.0)
        for index in range(1, 20):
            now = 10 + index * 0.05
            sample = Feedback(0.55, 0.55 + self.forward.speed_mps * (now - 10), math.pi / 2, now)
            self.assertEqual(nav.tick(now, [sample]), command)
        sample = Feedback(0.55, 0.55 + self.forward.speed_mps, math.pi / 2, 11)
        self.assertEqual(nav.tick(11, [sample]), (0, 0))
        self.assertEqual(nav.state, 'waiting')
        self.assertEqual(nav.index, 1)

    def test_forward_segment_shortens_near_waypoint_and_clearance(self):
        """Preserve robot clearance rather than blindly driving a full second."""
        # stop short of corners when necessary
        path = [(0.55, 1.5), (0.85, 1.5)]
        nav = Navigator(self.grid, path, self.forward, self.turn, self.policy, 10)
        pose = Feedback(0.55, 1.5, 0, 10)
        self.assertEqual(nav.drive_segment(10, pose, path[-1]), (self.forward.speed_mps, 0))
        self.assertLess(nav.commanded_distance, 0.30)
        self.assertLess(nav.deadline - 10, 1.0)
        self.assertTrue(nav.forward_sweep_free(pose, nav.commanded_distance))

    def test_fast_default_moves_immediately_without_tag_feedback(self):
        """Configured start placement starts motion without acquiring a visual pose."""
        # begin the planned turn without observations
        policy = dict(self.policy, feedback_required=False)
        nav = Navigator(self.grid, [(0.55, 0.55), (0.55, 2.5)], self.forward, self.turn, policy, 10)
        self.assertNotEqual(nav.tick(10.05, []), (0, 0))
        self.assertEqual(nav.state, 'turning')
        for index in range(2, 25):
            nav.tick(10 + index * 0.05, [])
        self.assertEqual(nav.state, 'moving')
        self.assertEqual(nav.command, (self.forward.speed_mps, 0))
        self.assertEqual(nav.pose_source, 'command prediction')
        self.assertIsNone(nav.measured)

    def test_optional_visual_correction_is_used_or_explicitly_rejected(self):
        """Missing or inconsistent corrections must not stall the requested fast mode."""
        # accept plausible observations and reject jumps
        policy = dict(self.policy, feedback_required=False)
        for x, y, accepted in [(0.56, 0.55, True), (2.5, 2.5, False)]:
            nav = Navigator(self.grid, self.plans['fast'], self.forward, self.turn, policy, 10)
            samples = [Feedback(x, y, 0, stamp, (0,)) for stamp in (10.0, 10.1, 10.2)]
            self.assertNotEqual(nav.tick(10.2, samples), (0, 0))
            self.assertEqual(nav.measured is not None, accepted)
            if not accepted:
                self.assertIn('visual correction rejected', nav.feedback_note)
                self.assertFalse(nav.finished)

    def test_missing_tags_search_then_finish_predicted_route_as_unverified(self):
        """Exercise bounded searches and predicted waypoint progress with no observations."""
        # exercise the full route without observations
        policy = dict(self.policy, feedback_required=False)
        grid = deepcopy(self.grid)
        grid.margin = self.environment['robot']['safety_margin_m']
        nav = Navigator(grid, self.plans['fast'], self.forward, self.turn, policy, 10)
        state, reason, pose, clearance = simulate(nav, (0.55, 0.55, 0), feedback=False)
        self.assertEqual(state, 'unverified', reason)
        self.assertIn('goal not visually confirmed', reason)
        self.assertEqual(nav.index, len(nav.path))
        self.assertEqual(nav.unverified_waypoints, list(range(1, len(nav.path))))
        self.assertIsNone(nav.measured)
        self.assertGreaterEqual(clearance, grid.margin)
        self.assertLessEqual(math.dist(pose[:2], nav.path[-1]), self.policy['goal_tolerance_m'])
        self.assertEqual(nav.command, (0, 0))

    def test_search_stops_for_tags_and_requires_post_stop_confirmation(self):
        """Keep the target while reacquiring a stopped pose and then reorienting."""
        # start recovery without changing the active waypoint
        policy = dict(self.policy, feedback_required=False)
        nav = Navigator(self.grid, [(0.55, 0.55), (0.55, 2.5)], self.forward, self.turn, policy, 10)
        nav.begin_search(10)
        for i in range(1, 11):
            nav.tick(10 + i * 0.05, [])
        self.assertEqual(nav.index, 1)
        self.assertEqual(nav.state, 'searching')
        self.assertGreater(nav.command[1], 0)

        # brake on detection and ignore pre-stop samples
        nav.tick(10.55, [Feedback(0.55, 0.55, 0.9, 10.5)])
        self.assertEqual(nav.command, (0, 0))
        self.assertEqual(nav.search.phase, 'observing')
        after, yaw = nav.search.after, nav.expected[2]
        samples = [Feedback(0.55, 0.55, yaw, after + dt) for dt in (0.01, 0.11, 0.21)]
        for i in range(12, 40):
            now = 10 + i * 0.05
            nav.tick(now, [p for p in samples if p.stamp <= now])
            if nav.state != 'searching':
                break
        self.assertIsNotNone(nav.measured)
        self.assertEqual(nav.index, 1)
        self.assertEqual(nav.state, 'turning')
        self.assertAlmostEqual(nav.turn_target_yaw, math.pi / 2)
        self.assertEqual(nav.unverified_waypoints, [])

    def test_exhausted_search_preserves_heading_and_target_before_driving(self):
        """Track the full predicted sweep and reorient toward the same waypoint."""
        policy = dict(self.policy, feedback_required=False)
        nav = Navigator(self.grid, [(0.55, 0.55), (0.55, 2.5)], self.forward, self.turn, policy, 10, 0.3)
        nav.begin_search(10)
        for i in range(1, 401):
            nav.tick(10 + i * 0.05, [])
            if nav.state != 'searching':
                break
        self.assertEqual(nav.index, 1)
        self.assertEqual(nav.state, 'turning')
        self.assertGreaterEqual(nav.search.rotated, 2 * math.pi)
        self.assertLess(nav.search.rotated, 2 * math.pi + self.turn.ccw_radps * 0.051)
        self.assertAlmostEqual(nav.turn_origin.yaw, nav.search.expected[2])
        self.assertAlmostEqual(nav.turn_target_yaw, math.pi / 2)
        self.assertLess(10 + i * 0.05, 10 + policy['search_timeout_s'])

    def test_near_goal_mode_ignores_bad_vision_through_initial_traversal(self):
        """Ignore camera poses and tag searches until predicted arrival in the approach zone."""
        # inject wrong observations throughout the initial path
        policy = dict(self.policy, feedback_required=False, visual_feedback='near_goal')
        path = [(0.55, 0.55), (0.55, 1.0), (0.55, 2.5)]
        nav = Navigator(self.grid, path, self.forward, self.turn, policy, 10, math.pi / 2)
        for i in range(1, 151):
            now = 10 + i * 0.05
            samples = [Feedback(2.5, 0.55, 0, now - age) for age in (0.3, 0.2, 0.1)]
            nav.tick(now, samples)
            self.assertIsNone(nav.measured)
            self.assertFalse(nav.finished)
            self.assertNotEqual(nav.state, 'searching')
            if nav.goal_feedback_active:
                break
        self.assertTrue(nav.goal_feedback_active)
        self.assertEqual(nav.index, 2)
        self.assertEqual(nav.unverified_waypoints, [1])
        self.assertLessEqual(math.dist(nav.expected[:2], path[-1]), policy['visual_enable_distance_m'])
        self.assertEqual(nav.command, (0, 0))

    def test_goal_feedback_stays_enabled_and_shortens_fixed_power_steps(self):
        """Keep corrections active after a pose update moves the estimated robot farther away."""
        # enter and latch the predicted approach zone
        policy = dict(self.policy, feedback_required=False, visual_feedback='near_goal')
        path = [(0.55, 0.55), (0.55, 2.5)]
        nav = Navigator(self.grid, path, self.forward, self.turn, policy, 10, math.pi / 2)
        nav.expected = (0.55, 2.21, math.pi / 2)
        self.assertTrue(nav.visual_feedback_enabled())
        pose = Feedback(0.55, 2.1, math.pi / 2, 11)
        self.assertIsNone(nav.apply_checkpoint_pose(pose))
        self.assertTrue(nav.visual_feedback_enabled())
        self.assertAlmostEqual(nav.approach_step_limit(pose, path[-1]), policy['goal_max_step_m'])
        self.assertEqual(nav.drive_segment(11, pose, path[-1]), (self.forward.speed_mps, 0))
        self.assertAlmostEqual(nav.deadline - 11, policy['goal_max_step_m'] / self.forward.speed_mps)
        closer = Feedback(0.55, 2.42, math.pi / 2, 12)
        self.assertAlmostEqual(nav.approach_step_limit(closer, path[-1]), 0.04)

    def test_throughout_mode_stops_at_goal_approach_and_corrects_from_vision(self):
        """Stop near the goal and replace prediction with a fresh stopped pose."""
        # shorten the drive at the approach boundary
        policy = dict(self.policy, feedback_required=False, visual_feedback='throughout')
        path = [(0.55, 0.55), (0.55, 2.5)]
        nav = Navigator(self.grid, path, self.forward, self.turn, policy, 10, math.pi / 2)
        start = Feedback(0.55, 2.1, math.pi / 2, 10)
        nav.expected = start.x, start.y, start.yaw
        self.assertAlmostEqual(nav.approach_step_limit(start, path[-1]), 0.1)
        nav.drive_segment(10, start, path[-1])
        for i in range(1, 5):
            nav.tick(10 + i * 0.05, [])
        self.assertEqual(nav.state, 'verifying')
        self.assertTrue(nav.goal_feedback_active)
        self.assertEqual(nav.command, (0, 0))

        # correct a short physical stop before more travel
        after = nav.after
        samples = [Feedback(0.55, 2.12, math.pi / 2, after + dt) for dt in (0.01, 0.11, 0.21)]
        for i in range(5, 35):
            now = 10 + i * 0.05
            nav.tick(now, [pose for pose in samples if pose.stamp <= now])
            if nav.state == 'moving':
                break
        self.assertEqual(nav.state, 'moving')
        self.assertTrue(nav.goal_feedback_active)
        self.assertAlmostEqual(nav.segment_pose.y, 2.12)
        self.assertAlmostEqual(nav.deadline - nav.segment_start, policy['goal_max_step_m'] / self.forward.speed_mps)
        self.assertEqual(nav.index, 1)
        self.assertEqual(nav.unverified_waypoints, [])

    def test_end_only_mode_ignores_initial_vision_then_corrects_without_search(self):
        """Use observations only after predicted arrival; prohibit recovery spins."""
        # follow the route without accepting initial observations
        policy = dict(self.policy, feedback_required=False, visual_feedback='at_goal', tag_search_enabled=False)
        path = [(0.55, 0.55), (0.55, 1.0), (0.55, 2.5)]
        nav = Navigator(self.grid, path, self.forward, self.turn, policy, 10, math.pi / 2)
        for i in range(1, 151):
            now = 10 + i * 0.05
            bad = [Feedback(2.5, 0.55, 0, now - age) for age in (0.3, 0.2, 0.1)]
            nav.tick(now, bad)
            self.assertIsNone(nav.measured)
            self.assertIsNone(nav.search)
            if nav.goal_feedback_active:
                break
        self.assertTrue(nav.goal_feedback_active)
        self.assertLessEqual(math.dist(nav.expected[:2], path[-1]), policy['goal_tolerance_m'])
        self.assertEqual(nav.state, 'verifying')

        # correct an observed short stop using ordinary travel turns
        after = nav.after
        samples = [Feedback(0.55, 2.12, math.pi / 2, after + dt) for dt in (0.01, 0.11, 0.21)]
        for j in range(i + 1, i + 40):
            now = 10 + j * 0.05
            nav.tick(now, [p for p in samples if p.stamp <= now])
            if nav.state == 'moving':
                break
        self.assertEqual(nav.state, 'moving')
        self.assertAlmostEqual(nav.segment_pose.y, 2.12)
        self.assertIsNone(nav.search)

    def test_end_only_mode_with_no_tags_never_searches(self):
        """Stop unverified at the predicted end when final observations are unavailable."""
        policy = dict(self.policy, feedback_required=False, visual_feedback='at_goal', tag_search_enabled=False)
        path = [(0.55, 0.55), (0.55, 2.5)]
        nav = Navigator(self.grid, path, self.forward, self.turn, policy, 10, math.pi / 2)
        state, reason, pose, gap = simulate(nav, (0.55, 0.55, math.pi / 2), feedback=False)
        self.assertEqual(state, 'unverified', reason)
        self.assertIsNone(nav.search)
        self.assertIsNone(nav.measured)

    def test_goal_reference_uses_wall_tag_three_without_second_tag(self):
        """Recover a single wall-tag pose while preserving consensus requirements elsewhere."""
        # compose the measured reference in the real map
        environment = load_validated_environment(ROOT / 'config/environment.json')
        model = Localizer(environment, self.settings['camera_mount'], self.settings['localization'])
        key = ('36h11', 3)
        base = make_transform(euler_to_rotation_matrix(0, 0, math.pi / 3), [0.65, 2.1, 0])
        transform = np.linalg.inv(base @ np.linalg.inv(model.camera_base)) @ model.landmarks[key]
        pose = model.estimate_tag([(key, transform)], key, 10)
        np.testing.assert_allclose((pose.x, pose.y, pose.yaw), (0.65, 2.1, math.pi / 3), atol=1e-10)
        self.assertEqual(pose.tags, (key,))
        with self.assertRaisesRegex(ValueError, 'insufficient tags'):
            model.estimate([(key, transform)], 10)
        with self.assertRaisesRegex(ValueError, 'unavailable'):
            model.estimate_tag([(('41h12', 3), transform)], key, 10)

    def test_goal_correction_uses_reference_samples_instead_of_fused_pose(self):
        """Use tag three to correct a short stop despite conflicting multi-tag feedback."""
        # prefer the independent reference only in approach
        policy = dict(self.policy, feedback_required=False, visual_feedback='near_goal', tag_search_enabled=False)
        path = [(0.55, 0.55), (0.55, 2.5)]
        nav = Navigator(self.grid, path, self.forward, self.turn, policy, 10, math.pi / 2)
        nav.expected = (0.55, 2.22, math.pi / 2)
        self.assertTrue(nav.visual_feedback_enabled())
        nav.begin_verification(10)
        after = nav.after
        goal = [Feedback(0.55, 2.12, math.pi / 2, after + dt, (('36h11', 3),))
                for dt in (0.01, 0.11, 0.21)]
        fused = [Feedback(0.70, 2.46, math.pi / 2, p.stamp) for p in goal]
        for i in range(1, 40):
            now = 10 + i * 0.05
            nav.tick(now, [p for p in fused if p.stamp <= now], [p for p in goal if p.stamp <= now])
            if nav.state == 'moving':
                break
        self.assertEqual(nav.state, 'moving')
        self.assertAlmostEqual(nav.segment_pose.y, 2.12)
        self.assertIn('36h11:3', nav.pose_source)
        self.assertEqual(nav.unverified_waypoints, [])
        self.assertIsNone(nav.search)

    def test_missing_reference_cannot_be_replaced_by_other_visible_tags(self):
        """Stop with an explicit missing-reference result and never spin to search."""
        policy = dict(self.policy, feedback_required=False, visual_feedback='near_goal', tag_search_enabled=False)
        nav = Navigator(self.grid, [(0.55, 0.55), (0.55, 2.5)], self.forward, self.turn, policy, 10)
        nav.expected = (0.55, 2.22, math.pi / 2)
        self.assertTrue(nav.visual_feedback_enabled())
        nav.begin_verification(10)
        for i in range(1, 180):
            now = 10 + i * 0.05
            fused = [Feedback(0.55, 2.49, math.pi / 2, now - age) for age in (0.3, 0.2, 0.1)]
            nav.tick(now, fused, goal_samples=[])
            if nav.finished:
                break
        self.assertEqual(nav.state, 'unverified')
        self.assertIn('36h11:3 unavailable', nav.reason)
        self.assertIsNone(nav.measured)
        self.assertIsNone(nav.search)
        self.assertEqual(nav.command, (0, 0))

    def test_off_mode_never_uses_available_vision_or_searches(self):
        """Finish predicted traversal unverified even when ideal feedback is available."""
        policy = dict(self.policy, feedback_required=False, visual_feedback='off')
        path = [(0.55, 0.55), (0.55, 1.0), (0.55, 2.5)]
        nav = Navigator(self.grid, path, self.forward, self.turn, policy, 10, math.pi / 2)
        state, reason, pose, gap = simulate(nav, (0.55, 0.55, math.pi / 2))
        self.assertEqual(state, 'unverified', reason)
        self.assertIsNone(nav.measured)
        self.assertIsNone(nav.search)
        self.assertEqual(nav.unverified_waypoints, [1, 2])

    def test_reported_short_stop_triggers_measured_goal_correction(self):
        """Replay the observed 34 cm goal error instead of accepting predicted arrival."""
        # reproduce the measured obstacle-side stopping position
        data = deepcopy(self.environment)
        data['robot'].update(length_m=0.16, width_m=0.16)
        data['obstacles'][0].update(x_min_m=1.14, x_max_m=1.76, y_min_m=1.23, y_max_m=1.85)
        grid = build_grid(data)
        path = [(2.345, 0.63), (0.95, 1.0), (0.85, 1.2), (0.75, 1.54)]
        policy = dict(self.policy, feedback_required=False)
        nav = Navigator(grid, path, self.forward, self.turn, policy, 10, math.pi)
        nav.index, nav.expected = 3, (0.75384, 1.56843, 1.8344)
        nav.begin_verification(10)
        samples = [Feedback(0.9412, 1.2560, 2.2664, stamp, (('41h12', 1),))
                   for stamp in (11.05, 11.25, 11.45)]
        self.assertEqual(nav.tick(11.5, samples), (self.forward.speed_mps, 0))
        self.assertEqual(nav.index, 3)
        self.assertEqual(nav.state, 'moving')
        self.assertAlmostEqual(nav.expected[0], 0.9412)
        self.assertIn('checkpoint correction accepted', nav.feedback_note)
        self.assertTrue(nav.segment_is_clear(samples[-1].xy, path[-1]))

    def test_margin_recovery_must_move_outward_and_respect_collision_floor(self):
        """A checkpoint inside the safety margin cannot authorize motion closer to obstacles."""
        # preserve measured clearance while leaving the margin
        grid = build_grid(self.environment)
        x = 1.2 - grid.radius - 0.08
        path = [(0.55, 0.55), (0.55, 1.5)]
        nav = Navigator(grid, path, self.forward, self.turn, self.policy, 10)
        pose = Feedback(x, 1.5, math.pi, 11)
        nav.expected = pose.x, pose.y, pose.yaw
        self.assertIsNone(nav.apply_checkpoint_pose(pose))
        self.assertTrue(nav.segment_is_clear(pose.xy, (x - 0.1, 1.5)))
        self.assertFalse(nav.segment_is_clear(pose.xy, (x + 0.01, 1.5)))
        unsafe = Feedback(1.2 - grid.radius - 0.01, 1.5, math.pi, 12)
        self.assertIn('too close to collision', nav.apply_checkpoint_pose(unsafe))

    def test_goal_requires_all_fresh_frames_inside_goal_tolerance(self):
        """A close average cannot hide one confirming observation outside tolerance."""
        # hold the target until every frame agrees
        path = [(0.55, 0.55), (0.55, 1.0)]
        nav = Navigator(self.grid, path, self.forward, self.turn, self.policy, 10)
        nav.expected = 0.55, 1.0, math.pi / 2
        nav.begin_verification(10)
        tolerance = self.policy['goal_tolerance_m']
        samples = [Feedback(0.55 + dx, 1.0, math.pi / 2, stamp, (0,))
                   for dx, stamp in zip((tolerance - 0.01, tolerance - 0.009, tolerance + 0.001),
                                       (11.1, 11.3, 11.5))]
        self.assertEqual(nav.tick(11.55, samples), (0, 0))
        self.assertEqual(nav.state, 'verifying')
        self.assertEqual(nav.index, 1)

        # finish only after new measured confirmation
        samples = [Feedback(0.55 + dx, 1.0, math.pi / 2, stamp, (0,))
                   for dx, stamp in zip((0.01, 0.015, 0.02), (11.6, 11.8, 12.0))]
        self.assertEqual(nav.tick(12.05, samples), (0, 0))
        self.assertEqual(nav.state, 'complete')
        self.assertEqual(nav.index, 2)

    def test_goal_refinement_rejects_the_logged_twenty_four_centimeter_jump(self):
        """Do not steer toward a new position cluster after already approaching the goal."""
        # reject the observed localization cluster switch
        grid = build_grid(self.environment)
        path = [(0.55, 0.55), (0.75, 1.54)]
        nav = Navigator(grid, path, self.forward, self.turn, self.policy, 10)
        nav.expected = 0.775, 1.527, math.radians(-22.9)
        nav.measured = Feedback(0.67, 1.571, math.radians(112.4), 10)
        bad = Feedback(0.508, 1.522, math.radians(28.3), 12)
        error = nav.apply_checkpoint_pose(bad)
        self.assertIn('position limit', error)
        self.assertEqual(nav.expected[:2], (0.775, 1.527))

    def test_logged_bad_tag_is_excluded_from_an_agreeing_majority(self):
        """Keep the wall pair and discard the obstacle estimate from a real capture."""
        # reproduce the supplied rotation frame geometry
        candidates = [(('36h11', 1), (0.758578, 2.291003, -0.134222)),
                      (('36h11', 7), (0.780137, 2.321378, -0.101039)),
                      (('41h12', 2), (0.265884, 2.717304, -0.126320))]
        kept = select_inliers(candidates, self.settings['localization'])
        self.assertEqual([tag for tag, pose in kept], [('36h11', 1), ('36h11', 7)])

        # do not manufacture consensus when every tag disagrees
        candidates[1] = (('36h11', 7), (0.800, 1.608, -0.101039))
        with self.assertRaisesRegex(ValueError, 'no unique agreeing majority'):
            select_inliers(candidates, self.settings['localization'])

    def test_single_tag_cannot_confirm_its_own_accuracy(self):
        """Reject lone estimates, including the bad single-tag frames in the rotation."""
        candidates = [(('41h12', 2), (0.265884, 2.717304, -0.126320))]
        with self.assertRaisesRegex(ValueError, 'insufficient tags'):
            select_inliers(candidates, self.settings['localization'])

    def test_heading_outliers_and_ambiguous_clusters_are_rejected(self):
        """Handle heading wrap and refuse equally plausible overlapping clusters."""
        # compare orientations across the angle wrapping boundary
        candidates = [(('36h11', 1), (0.7, 2.3, math.radians(179))),
                      (('36h11', 3), (0.72, 2.3, math.radians(-179))),
                      (('36h11', 7), (0.71, 2.3, 0))]
        kept = select_inliers(candidates, self.settings['localization'])
        self.assertEqual(len(kept), 2)
        candidates = [(('36h11', i), (x, 2.3, 0)) for i, x in enumerate((0.5, 0.65, 0.8))]
        with self.assertRaisesRegex(ValueError, 'no unique agreeing majority'):
            select_inliers(candidates, self.settings['localization'])

    def test_waypoint_verification_cannot_use_pre_stop_or_stale_frames(self):
        """Repeated or old observations do not confirm an intermediate waypoint."""
        # preserve the current waypoint on timeout
        path = [(0.55, 0.55), (0.55, 1.0), (0.55, 2.5)]
        nav = Navigator(self.grid, path, self.forward, self.turn, self.policy, 10)
        nav.expected = 0.55, 1.0, math.pi / 2
        nav.begin_verification(10)
        old = [Feedback(0.55, 1.0, math.pi / 2, stamp, (0,)) for stamp in (9.7, 9.8, 9.9)]
        self.assertEqual(nav.tick(12, old), (0, 0))
        self.assertEqual(nav.index, 1)
        self.assertEqual(nav.tick(20, old), (0, 0))
        self.assertEqual(nav.index, 1)
        self.assertEqual(nav.state, 'failed')

    def test_measured_checkpoints_correct_a_slower_than_predicted_robot(self):
        """A robot moving at 65 percent of nominal speed must still reach measured goal."""
        # exercise repeated corrections instead of predicted completion
        policy = dict(self.policy, feedback_required=False)
        grid = deepcopy(self.grid)
        grid.margin = self.environment['robot']['safety_margin_m']
        nav = Navigator(grid, self.plans['fast'], self.forward, self.turn, policy, 10)
        state, reason, pose, clearance = simulate(nav, (0.55, 0.55, 0), linear_scale=0.65)
        self.assertEqual(state, 'complete', reason)
        self.assertLessEqual(math.dist(pose[:2], self.plans['fast'][-1]), policy['goal_tolerance_m'])
        self.assertGreaterEqual(clearance, grid.margin)

    def test_ideal_feedback_completes_fast_route(self):
        """Check fast-mode control under exact localization and ideal motor response."""
        # check ideal completion without asserting hardware accuracy
        for name, path in self.plans.items():
            with self.subTest(mode=name):
                grid = deepcopy(self.grid)
                grid.margin = self.environment['robot']['safety_margin_m']
                nav = get_mode(name).create_navigator(grid, path, self.forward, self.turn, self.policy, 10)
                state, reason, pose, clearance = simulate(nav, (0.55, 0.55, 0.0))
                self.assertEqual(state, 'complete', reason)
                self.assertLessEqual(math.dist(pose[:2], path[-1]), self.policy['position_tolerance_m'])
                self.assertGreaterEqual(clearance, grid.margin)
                nav.finish('stopped', 'stop after arrival')
                self.assertEqual(nav.state, 'complete')

    def test_stop_heading_mismatch_and_forward_budget(self):
        """Never resume a stopped mission or repeat blocked progress indefinitely."""
        # reject incorrect initial heading placement
        nav = Navigator(self.grid, self.plans['fast'], self.forward, self.turn, self.policy, 10)
        samples = [Feedback(0.55, 0.55, math.pi, t) for t in (10.0, 10.1, 10.2)]
        self.assertEqual(nav.tick(10.25, samples), (0, 0))
        self.assertEqual(nav.state, 'failed')
        self.assertIn('heading', nav.reason)

        # preserve targets after explicit cancellation
        nav = Navigator(self.grid, self.plans['fast'], self.forward, self.turn, self.policy, 10)
        nav.finish('stopped', 'test cancellation')
        self.assertEqual(nav.tick(10.25, samples), (0, 0))
        self.assertEqual(nav.state, 'stopped')
        self.assertEqual(nav.index, 1)

        # fail when commands exceed progress allowance
        nav = Navigator(self.grid, self.plans['fast'], self.forward, self.turn, self.policy, 10)
        nav.segment_budget, nav.commanded_distance = 0.1, 0.1
        pose = Feedback(0.55, 0.55, 0, 10)
        self.assertEqual(nav.drive_segment(10, pose, (1.0, 0.55)), (0, 0))
        self.assertEqual(nav.state, 'failed')
        self.assertIn('budget', nav.reason)

    def test_motor_mapping_preserves_requested_forward_and_measured_turns(self):
        """Exercise actual straight, signed turn, and stop motor mapping."""
        # preserve user inputs and existing turns
        mapper = VelocityMapper(self.turn, self.forward)
        self.assertAlmostEqual(2 * math.pi / self.turn.ccw_radps, 3.55)
        self.assertAlmostEqual(2 * math.pi / self.turn.cw_radps, 3.59)
        for command, pair in [((self.forward.speed_mps, 0), (0.43, 0.43)), ((0, 1), (-0.43, 0.43)),
                              ((0, -1), (0.4, -0.4)), ((0, 0), (0., 0.))]:
            applied = mapper.map_command(*command)
            self.assertEqual((applied.left, applied.right), pair)
            if command[0] > 0:
                self.assertEqual(applied.linear_mps, 0.567333)

    def test_plan_preview_reports_signed_turns_and_distinct_tag_labels(self):
        """Expose turn direction and family/ID labels before executing a plan."""
        # compare geometric instructions against known headings
        data = deepcopy(self.environment)
        data['start']['yaw_deg'] = 90.0
        data['landmarks'].append(dict(data['landmarks'][1], family='41h12'))
        path = [(0.55, 0.55), (1.0, 0.55), (1.0, 1.0)]
        rows = plan_segments(data, path)
        self.assertAlmostEqual(rows[0]['heading_deg'], 0.0)
        self.assertAlmostEqual(rows[0]['turn_deg'], -90.0)
        self.assertAlmostEqual(rows[1]['turn_deg'], 90.0)
        description = plan_description(data, {'fast': path})
        self.assertIn('PLANNED PATH ONLY', description)
        self.assertIn('x_cm', description)
        self.assertIn('heading 90.0 deg', description)

        # retain labels for overlapping numeric identities
        import xml.etree.ElementTree as ET
        svg = map_svg(self.grid, data, {'fast': path})
        ET.fromstring(svg)
        for label in ('36h11:1', '41h12:1', 'START', 'GOAL', 'W1', 'OBSTACLE'):
            self.assertIn(label, svg)

    def test_artifact_snapshot_and_session_reuse_protection(self):
        """Preserve inputs and prevent a second mission from overwriting the first."""
        # protect recorded evidence from accidental overwriting
        with tempfile.TemporaryDirectory() as directory:
            paths = {'environment.json': ROOT / 'test/sample_environment.json'}
            log = RunLog(directory, paths)
            save_plans(directory, self.grid, self.environment, self.plans)
            log.status('failed', 'synthetic test failure', goal_reached=False)
            log.close()
            self.assertEqual(json.loads((Path(directory) / 'status.json').read_text())['goal_reached'], False)
            self.assertTrue((Path(directory) / 'planned_paths.svg').is_file())
            self.assertTrue((Path(directory) / 'planned_path.txt').is_file())
            with self.assertRaises(FileExistsError):
                RunLog(directory, paths)


if __name__ == '__main__':
    unittest.main()
