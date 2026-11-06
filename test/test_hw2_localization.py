import copy
import json
import math
from pathlib import Path
import tempfile
import threading
import unittest
from types import SimpleNamespace

import numpy as np

from robot_testing.hw2_localization import (
    Detection, LandmarkMap, LocalizationError, LocalizationService, Observation,
    PoseEstimator, PoseProvider, angle_distance, quaternion_transform, rotation,
    transform,
)


class LocalizationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / 'landmarks.json'
        config = Path(__file__).resolve().parents[1] / 'config/landmarks.json'
        self.data = json.loads(config.read_text())
        self.data['surveyed'] = True
        self.data['landmarks'][0].update(x_m=2., y_m=0., facing_deg=180.)
        self.data['landmarks'][1].update(x_m=2., y_m=1., facing_deg=180.)
        self.landmarks = self.load()
        self.estimator = PoseEstimator(self.landmarks)
        self.observation = Observation(100_000_000_000, 'camera_frame', (
            Detection(0, 'tag36h11', 0, 80.),
        ))
        self.front_transform = transform(np.diag([1., -1., -1.]), (0., 0., 2.))

    def load(self):
        self.path.write_text(json.dumps(self.data))
        return LandmarkMap(self.path)

    def observation_for(self, ids):
        return Observation(100_000_000_000, 'camera_frame',
                           tuple(Detection(i, '36h11', 0, 80.) for i in ids))

    def lookup_for_pose(self, x, y, yaw, landmarks=None):
        landmarks = landmarks or self.landmarks
        base = transform(rotation(0., 0., yaw), (x, y, 0.))
        camera = base @ np.linalg.inv(landmarks.camera_to_base)
        values = {tag.frame: np.linalg.inv(camera) @ tag.map_transform
                  for tag in landmarks.landmarks.values()}
        return lambda frame, observation: values[frame]

    def test_independently_known_front_facing_geometry(self):
        result = self.estimator.estimate(self.observation, lambda *_: self.front_transform)
        np.testing.assert_allclose((result.x, result.y, result.yaw), (0., 0., 0.), atol=1e-10)
        self.assertEqual(result.stamp_ns, self.observation.stamp_ns)
        self.assertEqual(result.tag_ids, (0,))

    def test_translated_rotated_robot_and_offset_tilted_camera(self):
        self.data['camera_mount'].update(x_m=.12, y_m=-.04, z_m=.31,
                                         yaw_deg=9., pitch_deg=8., roll_deg=-3.)
        landmarks = self.load()
        estimator = PoseEstimator(landmarks)
        rng = np.random.default_rng(276)
        for _ in range(20):
            x, y, yaw = rng.uniform(-.25, .25, 3)
            result = estimator.estimate(self.observation, self.lookup_for_pose(x, y, yaw, landmarks))
            np.testing.assert_allclose((result.x, result.y, result.yaw), (x, y, yaw), atol=1e-10)

    def test_facing_angle_is_the_outward_normal(self):
        for facing in (0., 90., 180., 270.):
            self.data['landmarks'][0]['facing_deg'] = facing
            tag = self.load().landmarks[('36h11', 0)]
            expected = (math.cos(math.radians(facing)), math.sin(math.radians(facing)), 0.)
            np.testing.assert_allclose(tag.map_transform[:3, 2], expected, atol=1e-10)
            np.testing.assert_allclose(tag.map_transform[:3, 1], (0., 0., 1.))

    def test_multitag_pose_and_disagreement(self):
        observation = self.observation_for((0, 1))
        lookup = self.lookup_for_pose(.2, -.1, .2)
        result = self.estimator.estimate(observation, lookup)
        self.assertEqual(result.tag_ids, (0, 1))
        np.testing.assert_allclose((result.x, result.y, result.yaw), (.2, -.1, .2), atol=1e-10)
        wrong = self.lookup_for_pose(.8, -.1, .2)
        with self.assertRaisesRegex(LocalizationError, 'disagree'):
            self.estimator.estimate(observation, lambda f, o: lookup(f, o) if f == 'tag_0' else wrong(f, o))

    def test_circular_heading_near_pi(self):
        self.data['landmarks'][0].update(x_m=-2.)
        self.data['landmarks'][1].update(x_m=-2.)
        landmarks = self.load()
        estimator = PoseEstimator(landmarks)
        left = self.lookup_for_pose(0., 0., math.pi-.02, landmarks)
        right = self.lookup_for_pose(0., 0., -math.pi+.02, landmarks)
        result = estimator.estimate(self.observation_for((0, 1)),
                                    lambda f, o: left(f, o) if f == 'tag_0' else right(f, o))
        self.assertLess(angle_distance(result.yaw, math.pi), .021)

    def test_unknown_weak_corrupt_and_missing_transforms_are_rejected(self):
        for detection in (Detection(9, '36h11', 0, 80.), Detection(0, '25h9', 0, 80.),
                          Detection(0, '36h11', 2, 80.), Detection(0, '36h11', 0, 1.),
                          Detection(0, '36h11', 0, float('nan'))):
            observation = Observation(self.observation.stamp_ns, 'camera_frame', (detection,))
            with self.assertRaises(LocalizationError):
                self.estimator.estimate(observation, lambda *_: self.front_transform)
        def missing(*_):
            raise LookupError('TF not received yet')
        with self.assertRaises(LocalizationError):
            self.estimator.estimate(self.observation, missing)
        with self.assertRaises(LocalizationError):
            self.estimator.estimate(Observation(100, 'wrong_frame', self.observation.detections), missing)
        for quaternion in ((0., 0., 0., 0.), (0., 0., float('nan'), 1.)):
            with self.assertRaises(LocalizationError):
                quaternion_transform((0., 0., 2.), quaternion)

    def test_nonplanar_and_behind_camera_geometry_is_rejected(self):
        for base in (transform(rotation(0., 0., 0.), (0., 0., 1.)),
                     transform(rotation(.8, 0., 0.), (0., 0., 0.))):
            tag = self.landmarks.landmarks[('36h11', 0)]
            camera = base @ np.linalg.inv(self.landmarks.camera_to_base)
            with self.assertRaises(LocalizationError):
                self.estimator.estimate(self.observation, lambda *_: np.linalg.inv(camera) @ tag.map_transform)
        behind = self.front_transform.copy()
        behind[2, 3] = -2.
        with self.assertRaises(LocalizationError):
            self.estimator.estimate(self.observation, lambda *_: behind)

    def test_invalid_map_is_rejected(self):
        original = copy.deepcopy(self.data)
        mutations = (
            lambda d: d.update(surveyed=False),
            lambda d: d['landmarks'].append(copy.deepcopy(d['landmarks'][0])),
            lambda d: d['landmarks'][0].update(x_m=float('nan')),
            lambda d: d['landmarks'][0].update(id=-1),
            lambda d: d.update(landmarks=[]),
            lambda d: d['quality'].update(max_disagreement_m=0),
        )
        for mutation in mutations:
            self.data = copy.deepcopy(original)
            mutation(self.data)
            with self.assertRaises(LocalizationError):
                self.load()

    def provider(self, lookup=None, timeout=0.):
        return PoseProvider(self.estimator, lookup or (lambda *_: self.front_transform),
                            lambda: 100_100_000_000, timeout_s=timeout)

    def test_observe_does_not_estimate_until_requested(self):
        calls = []
        provider = self.provider(lambda *args: calls.append(args) or self.front_transform)
        provider.observe(self.observation)
        self.assertEqual(calls, [])
        result = provider.request()
        self.assertEqual(result.stamp_ns, self.observation.stamp_ns)
        self.assertEqual(calls[0][0], 'tag_0')
        self.assertIs(calls[0][1], self.observation)

    def test_missing_stale_future_and_empty_frames_fail(self):
        provider = self.provider()
        with self.assertRaisesRegex(LocalizationError, 'No detection'):
            provider.request()
        for stamp in (99_000_000_000, 101_000_000_000):
            provider = self.provider()
            provider.observe(Observation(stamp, 'camera_frame', self.observation.detections))
            with self.assertRaisesRegex(LocalizationError, 'stale or future'):
                provider.request()
        provider = self.provider()
        provider.observe(self.observation)
        provider.observe(Observation(100_010_000_000, 'camera_frame', ()))
        with self.assertRaises(LocalizationError):
            provider.request()

    def test_delayed_tf_is_retried_without_replacing_the_stamp(self):
        calls = []
        def delayed(*args):
            calls.append(args)
            if len(calls) < 3:
                raise LookupError('not yet')
            return self.front_transform
        provider = self.provider(delayed, timeout=.2)
        provider.observe(self.observation)
        self.assertEqual(provider.request().stamp_ns, self.observation.stamp_ns)
        self.assertEqual(len(calls), 3)

    def test_new_detection_can_arrive_while_request_waits(self):
        provider = self.provider(timeout=.3)
        timer = threading.Timer(.02, provider.observe, (self.observation,))
        timer.start()
        try:
            self.assertEqual(provider.request().tag_ids, (0,))
        finally:
            timer.join()

    def test_pose_that_expires_during_estimation_is_not_returned(self):
        clock = [100_100_000_000]
        def lookup(*_):
            clock[0] = 100_900_000_000
            return self.front_transform
        provider = PoseProvider(self.estimator, lookup, lambda: clock[0], timeout_s=0.)
        provider.observe(self.observation)
        with self.assertRaisesRegex(LocalizationError, 'expired'):
            provider.request()

    def test_ros_lookup_uses_camera_to_tag_at_exact_observation_time(self):
        service = LocalizationService.__new__(LocalizationService)
        service.node = SimpleNamespace(get_clock=lambda: SimpleNamespace(clock_type='ROS'))
        service.time_type = lambda **kwargs: SimpleNamespace(**kwargs)
        service.tf_exception = LookupError
        calls = []
        def lookup(*args):
            calls.append(args)
            return SimpleNamespace(transform=SimpleNamespace(
                translation=SimpleNamespace(x=0., y=0., z=2.),
                rotation=SimpleNamespace(x=1., y=0., z=0., w=0.)))
        service.tf_buffer = SimpleNamespace(lookup_transform=lookup)
        np.testing.assert_allclose(service.lookup('tag_0', self.observation), self.front_transform)
        self.assertEqual(calls[0][:2], ('camera_frame', 'tag_0'))
        self.assertEqual(calls[0][2].nanoseconds, self.observation.stamp_ns)
        self.assertEqual(calls[0][2].clock_type, 'ROS')

    def test_service_response_and_failure(self):
        service = LocalizationService.__new__(LocalizationService)
        service.provider = self.provider()
        pose = SimpleNamespace(header=SimpleNamespace(frame_id='', stamp=SimpleNamespace(sec=0, nanosec=0)),
                               pose=SimpleNamespace(position=SimpleNamespace(x=0., y=0., z=0.),
                                                    orientation=SimpleNamespace(x=0., y=0., z=0., w=0.)))
        response = SimpleNamespace(success=False, message='', pose=pose, tag_ids=[])
        failed = service.on_request(None, copy.deepcopy(response))
        self.assertFalse(failed.success)
        self.assertEqual(failed.pose.header.frame_id, '')
        service.provider.observe(self.observation)
        result = service.on_request(None, response)
        self.assertTrue(result.success)
        self.assertEqual(result.pose.header.frame_id, 'map')
        self.assertEqual(result.pose.header.stamp.sec, 100)
        self.assertEqual(result.tag_ids, [0])
        self.assertAlmostEqual(result.pose.pose.orientation.w, 1.)


if __name__ == '__main__':
    unittest.main()
