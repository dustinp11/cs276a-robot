"""Unit tests for HW4 visual localization."""

import math
import unittest
import numpy as np

from robot_testing.hw2_localization import (
    Detection,
    LandmarkMap,
    Observation,
    transform,
)
from robot_testing.hw4_localization import HW4PoseEstimator


class TestHW4Localization(unittest.TestCase):
    def setUp(self):
        self.map = LandmarkMap('config/hw4_landmarks.json')
        self.estimator = HW4PoseEstimator(self.map)

    def test_landmark_loading_and_families(self):
        """Verify all 8 wall tags and 4 obstacle tags are loaded correctly."""
        self.assertGreaterEqual(len(self.map.landmarks), 12)
        # Check wall tags (36h11)
        wall_ids = {0, 1, 3, 4, 5, 7, 8, 9}
        obs_ids = {1, 2, 3, 4}
        for wid in wall_ids:
            self.assertIn(('36h11', wid), self.map.landmarks)
            self.assertIn(wid, self.estimator.wall_tag_ids)
        for oid in obs_ids:
            self.assertIn(('41h12', oid), self.map.landmarks)
            self.assertIn(oid, self.estimator.obstacle_tag_ids)

    def test_tag_classification(self):
        self.assertIn('Wall Tag 5 (36h11)', self.estimator.classify_tag('36h11', 5))
        self.assertIn('Obstacle Tag 2 (41h12)', self.estimator.classify_tag('41h12', 2))

    def test_pose_estimation_from_wall_tag(self):
        """Simulate robot at (1.0, 1.0) facing east looking at Tag 0 (x=3.0, y=0.95, facing 180)."""
        # Wall tag 0 is at (3.0, 0.95, 0.15), facing 180 degrees (-X direction)
        # If robot is at (1.0, 0.95) facing east (0 degrees):
        # Camera is facing tag 0 directly at distance 2.0 m along camera Z axis
        tag0 = self.map.landmarks[('36h11', 0)]
        # Camera pose in map frame:
        # Robot base: pos=(1.0, 0.95, 0.0), yaw=0.0
        # camera_to_base has xyz=(0, 0, 0.2), rpy=(0, 0, 0)
        # So camera in map: (1.0, 0.95, 0.2)
        robot_x, robot_y, robot_yaw = 1.0, 0.95, 0.0
        c, s = math.cos(robot_yaw), math.sin(robot_yaw)
        R_base = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        T_base = transform(R_base, [robot_x, robot_y, 0.0])
        T_camera = T_base @ np.linalg.inv(self.map.camera_to_base)

        # Camera to tag transform:
        T_camera_tag = np.linalg.inv(T_camera) @ tag0.map_transform

        obs = Observation(
            100_000_000_000,
            'camera_frame',
            (Detection(0, '36h11', 0, 85.0),),
        )
        lookup = lambda frame, obs: T_camera_tag

        estimate = self.estimator.estimate(obs, lookup)
        self.assertAlmostEqual(estimate.x, robot_x, places=3)
        self.assertAlmostEqual(estimate.y, robot_y, places=3)
        self.assertAlmostEqual(estimate.yaw, robot_yaw, places=3)
        self.assertEqual(estimate.tag_ids, (0,))

    def test_multi_tag_consensus_wall_and_obstacle(self):
        """Simulate robot observing both a wall tag and an obstacle tag simultaneously."""
        robot_x, robot_y, robot_yaw = 0.8, 1.54, 0.0  # looking towards east
        # Obstacle tag 1 is at (1.14, 1.54), facing 180 (west)
        # Wall tag 0 is at (3.00, 0.95), facing 180 (west)
        c, s = math.cos(robot_yaw), math.sin(robot_yaw)
        R_base = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        T_base = transform(R_base, [robot_x, robot_y, 0.0])
        T_camera = T_base @ np.linalg.inv(self.map.camera_to_base)

        tag_obs1 = self.map.landmarks[('41h12', 1)]
        tag_wall0 = self.map.landmarks[('36h11', 0)]

        T_cam_obs1 = np.linalg.inv(T_camera) @ tag_obs1.map_transform
        T_cam_wall0 = np.linalg.inv(T_camera) @ tag_wall0.map_transform

        obs = Observation(
            100_000_000_000,
            'camera_frame',
            (
                Detection(1, '41h12', 0, 90.0),
                Detection(0, '36h11', 0, 80.0),
            ),
        )
        def lookup(frame, observation):
            if '1' in frame and ('41h12' in frame or frame == 'tag_41h12_1'):
                return T_cam_obs1
            return T_cam_wall0

        estimate = self.estimator.estimate(obs, lookup)
        self.assertAlmostEqual(estimate.x, robot_x, places=2)
        self.assertAlmostEqual(estimate.y, robot_y, places=2)
        self.assertAlmostEqual(estimate.yaw, robot_yaw, places=2)
        self.assertIn(0, estimate.tag_ids)
        self.assertIn(1, estimate.tag_ids)

    def test_standard41h12_family_alias(self):
        """Verify detector family 'Standard41h12' or 'tagStandard41h12' matches '41h12'."""
        robot_x, robot_y, robot_yaw = 0.8, 1.54, 0.0
        c, s = math.cos(robot_yaw), math.sin(robot_yaw)
        R_base = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        T_base = transform(R_base, [robot_x, robot_y, 0.0])
        T_camera = T_base @ np.linalg.inv(self.map.camera_to_base)
        tag_obs1 = self.map.landmarks[('41h12', 1)]
        T_cam_obs1 = np.linalg.inv(T_camera) @ tag_obs1.map_transform

        for fam in ['Standard41h12', 'tagStandard41h12', '41h12']:
            obs = Observation(
                100_000_000_000,
                'camera_frame',
                (Detection(1, fam, 0, 90.0),),
            )
            estimate = self.estimator.estimate(obs, lambda f, o: T_cam_obs1)
            self.assertAlmostEqual(estimate.x, robot_x, places=2)
            self.assertEqual(estimate.tag_ids, (1,))


if __name__ == '__main__':
    unittest.main()
