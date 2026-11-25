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
from hw4.localization import Feedback, Localizer, stable_feedback
from hw4.navigation import Navigator
from hw4.observations import CaptureBuffer
from hw4.tags import canonical_family, detection_topic, detector_parameters, mapped_families, tag_frame
from hw4.fast import plan_path
from hw4.planner import get_mode
from hw4.preview import prepare_plans
from hw4.results import RunLog, path_metrics, save_plans
from hw4.settings import load_settings


def calibrations():
    """Load the actual HW4 shared motor configuration for every synthetic run."""
    params = yaml.safe_load((ROOT / 'config/motion.yaml').read_text())['/**']['ros__parameters']
    forward = ForwardCalibration(**params['forward_calibration'])
    turns = {key: value for key, value in params['turn_calibration'].items() if key != 'enabled'}
    return forward, TurnCalibration(**turns)


def simulate(navigator, pose, duration=240.0):
    """Integrate perfect motor response and 10 Hz exact observations at 20 Hz control."""
    command, samples, minimum = (0.0, 0.0), [], math.inf
    for index in range(int(duration / 0.05)):
        now = 10.0 + index * 0.05
        x, y, yaw = pose
        x += command[0] * 0.05 * math.cos(yaw)
        y += command[0] * 0.05 * math.sin(yaw)
        yaw += command[1] * 0.05
        pose = x, y, yaw
        minimum = min(minimum, navigator.grid.clearance((x, y)))

        # feed ideal samples with original timestamps
        if index % 2 == 0:
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
        cls.policy = cls.settings['navigation']
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

    def test_disagreeing_mapped_tags_are_rejected(self):
        """Two conflicting mapped-tag solutions cannot establish trustworthy feedback."""
        # introduce conflicting independently measured tag solutions
        localizer = Localizer(self.environment, self.settings['camera_mount'], self.settings['localization'])
        measurements = []
        for index, (tag, world) in enumerate(localizer.landmarks.items()):
            base = make_transform(np.eye(3), [0.55 + index * 0.6, 0.55, 0])
            measurements.append((tag, np.linalg.inv(base @ np.linalg.inv(localizer.camera_base)) @ world))
        with self.assertRaises(ValueError):
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

    def test_stale_vision_and_timer_overrun_stop_active_pulses(self):
        """Active motion cannot continue after missing feedback or a watchdog overrun."""
        # exercise both active motion failure conditions
        for now, samples in [(10.05, []), (10.25, [Feedback(0.55, 0.55, 0, 10.25)])]:
            nav = Navigator(self.grid, self.plans['fast'], self.forward, self.turn, self.policy, 10)
            nav.begin_pulse(10, Feedback(0.55, 0.55, 0, 10), self.forward.speed_mps, 0, 1, 'test pulse')
            self.assertEqual(nav.tick(now, samples), (0, 0))
            self.assertEqual(nav.state, 'failed')
            self.assertEqual(nav.index, 1)

    def test_continuous_turn_stops_at_measured_thirty_degrees(self):
        """Hold the turn command until new measured heading reaches its stage target."""
        # verify both directions across angular wrapping
        for direction, origin in [(1.0, math.radians(179)), (-1.0, math.radians(-179))]:
            nav = Navigator(self.grid, self.plans['fast'], self.forward, self.turn, self.policy, 10)
            pose = Feedback(0.55, 0.55, origin, 10)
            command = nav.begin_turn(10, pose, direction * math.pi / 2)
            self.assertEqual(nav.state, 'turning')
            self.assertAlmostEqual(nav.turn_angle, math.pi / 6)
            sample = Feedback(0.55, 0.55, origin + direction * math.radians(10), 10.05)
            self.assertEqual(nav.tick(10.05, [sample]), command)
            sample = Feedback(0.55, 0.55, origin + direction * math.radians(30), 10.1)
            self.assertEqual(nav.tick(10.1, [sample]), (0, 0))
            self.assertEqual(nav.state, 'waiting')
            self.assertEqual(nav.index, 1)

    def test_continuous_turn_does_not_use_timer_as_measured_arrival(self):
        """Fresh unchanged headings keep rotation active until the failure timeout."""
        # require physical progress instead of elapsed time
        nav = Navigator(self.grid, self.plans['fast'], self.forward, self.turn, self.policy, 10)
        pose = Feedback(0.55, 0.55, 0, 10)
        command = nav.begin_turn(10, pose, math.pi / 2)
        for index in range(1, 61):
            now = 10 + index * 0.05
            result = nav.tick(now, [Feedback(0.55, 0.55, 0, now)])
            if index < 60:
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
        self.assertEqual(nav.drive_pulse(10, pose, (1.0, 0.55)), (0, 0))
        self.assertEqual(nav.state, 'failed')
        self.assertIn('budget', nav.reason)

    def test_motor_mapping_preserves_requested_forward_and_hw3_turns(self):
        """Exercise actual straight, signed turn, and stop motor mapping."""
        # preserve user inputs and existing turns
        mapper = VelocityMapper(self.turn, self.forward)
        for command, pair in [((self.forward.speed_mps, 0), (0.4, 0.4)), ((0, 1), (-0.35, 0.35)),
                              ((0, -1), (0.37, -0.37)), ((0, 0), (0., 0.))]:
            applied = mapper.map_command(*command)
            self.assertEqual((applied.left, applied.right), pair)
            if command[0] > 0:
                self.assertEqual(applied.linear_mps, 0.567333)

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
            with self.assertRaises(FileExistsError):
                RunLog(directory, paths)


if __name__ == '__main__':
    unittest.main()
