"""Unit tests for Visual Motion Estimation using AprilTags (Step 2)."""

import math
import unittest
import numpy as np

from hw3.coordinates import CameraExtrinsics, euler_to_rotation_matrix, make_transform
from hw3.visual_odometry import VisualMotionEstimator


class TestVisualOdometry(unittest.TestCase):
    def setUp(self):
        self.extrinsics = CameraExtrinsics(x_m=0.0, y_m=0.0, z_m=0.2)
        self.vo = VisualMotionEstimator(extrinsics=self.extrinsics, max_dt_s=1.0)

    def test_forward_motion_single_tag(self):
        """Test relative forward displacement estimation from a stationary tag."""
        # Frame 1: Robot at x=0. Tag is at x=2.0 in front of camera
        # In optical coordinates: [0, 0, 2.0]
        det1 = [{
            'id': 0,
            'family': '36h11',
            't_cam_tag': make_transform(np.eye(3), [0.0, 0.0, 2.0]),
            'margin': 40.0,
            'hamming': 0,
        }]
        res1 = self.vo.process_frame(stamp_s=10.0, detections=det1)
        self.assertFalse(res1.valid, "First frame should establish baseline only")

        # Frame 2: Robot moved forward by 0.2 meters (now at x=0.2)
        # Tag is now at x=1.8 in front of camera: [0, 0, 1.8] in optical frame
        det2 = [{
            'id': 0,
            'family': '36h11',
            't_cam_tag': make_transform(np.eye(3), [0.0, 0.0, 1.8]),
            'margin': 40.0,
            'hamming': 0,
        }]
        res2 = self.vo.process_frame(stamp_s=10.1, detections=det2)

        self.assertTrue(res2.valid)
        self.assertAlmostEqual(res2.dx, 0.2, places=3)
        self.assertAlmostEqual(res2.dy, 0.0, places=3)
        self.assertAlmostEqual(res2.dtheta, 0.0, places=3)
        self.assertEqual(res2.matched_tag_ids, (0,))

    def test_pure_rotation_single_tag(self):
        """Test relative yaw rotation estimation from a stationary tag."""
        # Stationary tag at (x=2.0, y=0.0) in world
        # Frame 1: Robot at origin with yaw=0
        # t_cam_tag at t1:
        t_base_tag1 = make_transform(np.eye(3), [2.0, 0.0, 0.0])
        t_cam_tag1 = self.extrinsics.t_cam_base @ t_base_tag1

        det1 = [{
            'id': 1,
            'family': '36h11',
            't_cam_tag': t_cam_tag1,
            'margin': 40.0,
            'hamming': 0,
        }]
        self.vo.process_frame(stamp_s=1.0, detections=det1)

        # Frame 2: Robot rotates CCW in place by 30 degrees (+pi/6) at (0, 0)
        angle = math.radians(30.0)
        r_world_r2 = euler_to_rotation_matrix(0.0, 0.0, angle)
        t_world_r2 = make_transform(r_world_r2, [0.0, 0.0, 0.0])

        # In new robot frame: T_r2_l = inv(T_world_r2) @ T_world_l
        t_base_tag2 = np.linalg.inv(t_world_r2) @ t_base_tag1
        t_cam_tag2 = self.extrinsics.t_cam_base @ t_base_tag2

        det2 = [{
            'id': 1,
            'family': '36h11',
            't_cam_tag': t_cam_tag2,
            'margin': 40.0,
            'hamming': 0,
        }]
        res = self.vo.process_frame(stamp_s=1.1, detections=det2)

        self.assertTrue(res.valid)
        self.assertAlmostEqual(res.dx, 0.0, places=4)
        self.assertAlmostEqual(res.dy, 0.0, places=4)
        self.assertAlmostEqual(res.dtheta, angle, places=4)

    def test_multiple_tags_fusion(self):
        """Test multiple tags fusion combines estimates and reduces uncertainty."""
        # Two tags: tag 0 straight ahead (Z=2.0), tag 1 slightly to the left (X=-0.5, Z=2.0)
        det1 = [
            {'id': 0, 't_cam_tag': make_transform(np.eye(3), [0.0, 0.0, 2.0]), 'margin': 50.0},
            {'id': 1, 't_cam_tag': make_transform(np.eye(3), [-0.5, 0.0, 2.0]), 'margin': 50.0},
        ]
        self.vo.process_frame(stamp_s=1.0, detections=det1)

        # Both shift forward by 0.3m (Z becomes 1.7m)
        det2 = [
            {'id': 0, 't_cam_tag': make_transform(np.eye(3), [0.0, 0.0, 1.7]), 'margin': 50.0},
            {'id': 1, 't_cam_tag': make_transform(np.eye(3), [-0.5, 0.0, 1.7]), 'margin': 50.0},
        ]
        res = self.vo.process_frame(stamp_s=1.1, detections=det2)

        self.assertTrue(res.valid)
        self.assertAlmostEqual(res.dx, 0.3, places=3)
        self.assertEqual(len(res.matched_tag_ids), 2)
        # Covariance variance with 2 tags should be lower than single tag base variance
        self.assertLess(res.covariance[0, 0], (self.vo.base_trans_noise_m * 2.0) ** 2)

    def test_outlier_rejection_multiple_tags(self):
        """Test that an outlier tag observation with inconsistent jump is rejected."""
        det1 = [
            {'id': 0, 't_cam_tag': make_transform(np.eye(3), [0.0, 0.0, 2.0]), 'margin': 40.0},
            {'id': 1, 't_cam_tag': make_transform(np.eye(3), [-0.5, 0.0, 2.0]), 'margin': 40.0},
            {'id': 2, 't_cam_tag': make_transform(np.eye(3), [0.5, 0.0, 2.0]), 'margin': 40.0},
        ]
        self.vo.process_frame(stamp_s=1.0, detections=det1)

        # Tags 0 and 1 agree with dx=0.2m (Z=1.8), Tag 2 erroneously jumps by 1.5m (Z=0.5)
        det2 = [
            {'id': 0, 't_cam_tag': make_transform(np.eye(3), [0.0, 0.0, 1.8]), 'margin': 40.0},
            {'id': 1, 't_cam_tag': make_transform(np.eye(3), [-0.5, 0.0, 1.8]), 'margin': 40.0},
            {'id': 2, 't_cam_tag': make_transform(np.eye(3), [0.5, 0.0, 0.5]), 'margin': 40.0},
        ]
        res = self.vo.process_frame(stamp_s=1.1, detections=det2)

        self.assertTrue(res.valid)
        self.assertAlmostEqual(res.dx, 0.2, places=2)
        # Outlier tag 2 should be excluded from inliers
        self.assertNotIn(2, res.matched_tag_ids)

    def test_no_common_tags_available(self):
        """Test that no common tags results in valid=False without fabricated motion."""
        det1 = [{'id': 0, 't_cam_tag': make_transform(np.eye(3), [0.0, 0.0, 2.0])}]
        self.vo.process_frame(stamp_s=1.0, detections=det1)

        # Disjoint set of tags in next frame
        det2 = [{'id': 5, 't_cam_tag': make_transform(np.eye(3), [0.0, 0.0, 2.0])}]
        res = self.vo.process_frame(stamp_s=1.1, detections=det2)

        self.assertFalse(res.valid)
        self.assertEqual(res.dx, 0.0)
        self.assertEqual(res.dy, 0.0)
        self.assertIn("No common landmarks", res.status_message)


if __name__ == '__main__':
    unittest.main()
