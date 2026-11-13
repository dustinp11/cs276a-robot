"""Data Recording, Replay, and Export Infrastructure (Step 4).

Allows recording timestamped AprilTag detections and commanded velocities,
replaying them through Visual Odometry and EKF-SLAM without hardware,
and exporting estimated trajectories and landmark maps to CSV.
"""

import csv
import json
import math
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from .coordinates import CameraExtrinsics, make_transform, quaternion_to_rotation_matrix, wrap_angle
from .ekf_slam import EkfSlam
from .visual_odometry import VisualMotionEstimator


class DataRecorder:
    """Records AprilTag observations and motor commands to file."""

    def __init__(self, output_path: Union[str, Path]):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.records: List[dict] = []
        self._last_cmd = {'v': 0.0, 'omega': 0.0, 'time': 0.0}
        self._last_frame_time = 0.0

    def record_cmd_vel(self, v: float, omega: float, stamp_s: Optional[float] = None):
        """Update latest commanded velocity."""
        t = time.time() if stamp_s is None else float(stamp_s)
        self._last_cmd = {'v': float(v), 'omega': float(omega), 'time': t}

    def record_frame(
        self,
        stamp_s: float,
        detections: List[dict],
        cmd_v: Optional[float] = None,
        cmd_omega: Optional[float] = None,
    ):
        """Record an observation frame.

        Args:
            stamp_s: Timestamp in seconds.
            detections: List of detections with id, t_cam_tag or position/orientation, margin, etc.
            cmd_v: Optional linear velocity at this instant.
            cmd_omega: Optional angular velocity at this instant.

        Returns:
            None. State changes and publications are performed in place.
        """
        dt = 0.0 if self._last_frame_time == 0.0 else max(0.0, stamp_s - self._last_frame_time)
        self._last_frame_time = stamp_s

        v = self._last_cmd['v'] if cmd_v is None else float(cmd_v)
        omega = self._last_cmd['omega'] if cmd_omega is None else float(cmd_omega)

        cleaned_detections = []
        for det in detections:
            item = {
                'id': int(det['id']),
                'family': str(det.get('family', '36h11')),
                'margin': float(det.get('margin', 30.0)),
                'hamming': int(det.get('hamming', 0)),
            }
            if 't_cam_tag' in det:
                t = np.asarray(det['t_cam_tag'], dtype=float)
                item['t_cam_tag'] = t.tolist()
                item['position'] = t[:3, 3].tolist()
            elif 'position' in det:
                pos = list(det['position'])
                item['position'] = pos
                quat = det.get('quaternion', [0.0, 0.0, 0.0, 1.0])
                r = quaternion_to_rotation_matrix(*quat)
                item['t_cam_tag'] = make_transform(r, pos).tolist()
            cleaned_detections.append(item)

        record = {
            'timestamp_s': float(stamp_s),
            'dt_s': float(dt),
            'cmd_v': v,
            'cmd_omega': omega,
            'num_detections': len(cleaned_detections),
            'detections': cleaned_detections,
        }
        self.records.append(record)

    def save(self, path: Optional[Union[str, Path]] = None):
        """Save all records to JSON file."""
        target = self.output_path if path is None else Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, 'w', encoding='utf-8') as f:
            json.dump({'records': self.records}, f, indent=2)


class DataReplayer:
    """Replays recorded observations through Visual Odometry and EKF-SLAM."""

    def __init__(
        self,
        record_path: Union[str, Path],
        vo: Optional[VisualMotionEstimator] = None,
        ekf: Optional[EkfSlam] = None,
        extrinsics: Optional[CameraExtrinsics] = None,
    ):
        """Load recorded observations and configure the offline estimation pipeline.

        Args:
            record_path: Path to a recorded JSON observation run.
            vo: Optional VisualMotionEstimator instance; None constructs a default estimator.
            ekf: Optional EkfSlam instance; None constructs a default filter.
            extrinsics: CameraExtrinsics optical-to-base mount transform; None uses the default mount.

        Returns:
            None. State changes and publications are performed in place.
        """
        self.record_path = Path(record_path)
        with open(self.record_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.records = data.get('records', [])
        self.extrinsics = extrinsics if extrinsics is not None else CameraExtrinsics()
        self.vo = vo if vo is not None else VisualMotionEstimator(extrinsics=self.extrinsics)
        self.ekf = ekf if ekf is not None else EkfSlam()

    def run(
        self,
        use_vo: bool = True,
        use_cmd_fallback: bool = True,
    ) -> dict:
        """Run through all records in sequence.

        Args:
            use_vo: Whether valid visual motion estimates drive prediction.
            use_cmd_fallback: Whether recorded velocity commands may provide fallback prediction.

        Returns:
            Dictionary containing:
            - 'trajectory': list of (t, x, y, theta, sigma_x, sigma_y, sigma_theta)
            - 'landmarks': dict of tag_id -> (x, y, sigma_x, sigma_y, cov_xy)
            - 'vo_history': list of vo estimates
        """
        trajectory = []
        vo_history = []

        for rec in self.records:
            t = rec['timestamp_s']
            dt = rec.get('dt_s', 0.1)
            v = rec.get('cmd_v', 0.0)
            omega = rec.get('cmd_omega', 0.0)
            raw_dets = rec.get('detections', [])

            # Format detections for VO
            vo_dets = []
            meas_for_ekf = []
            for d in raw_dets:
                t_cam = np.array(d['t_cam_tag'], dtype=float)
                vo_dets.append({
                    'id': d['id'],
                    'family': d.get('family', '36h11'),
                    't_cam_tag': t_cam,
                    'margin': d.get('margin', 30.0),
                    'hamming': d.get('hamming', 0),
                })
                # Tag relative in robot base frame
                t_base_tag = self.extrinsics.camera_to_base_transform(t_cam)
                meas_for_ekf.append({
                    'id': d['id'],
                    'x_rel': float(t_base_tag[0, 3]),
                    'y_rel': float(t_base_tag[1, 3]),
                })

            # Process VO
            vo_est = self.vo.process_frame(t, vo_dets)
            vo_history.append(vo_est)

            # EKF Prediction
            if use_vo and vo_est.valid:
                self.ekf.predict_visual_odometry(
                    vo_est.dx, vo_est.dy, vo_est.dtheta, vo_est.covariance
                )
            elif use_cmd_fallback and dt > 0.0 and (abs(v) > 1e-4 or abs(omega) > 1e-4):
                self.ekf.predict_commanded_velocity(v, omega, dt)

            # EKF Measurement Update
            for m in meas_for_ekf:
                self.ekf.update_landmark(m['id'], m['x_rel'], m['y_rel'])

            pose = self.ekf.get_robot_pose()
            cov = pose.covariance
            trajectory.append({
                'timestamp_s': t,
                'x': pose.x,
                'y': pose.y,
                'theta': pose.theta,
                'sigma_x': math.sqrt(max(0.0, cov[0, 0])),
                'sigma_y': math.sqrt(max(0.0, cov[1, 1])),
                'sigma_theta': math.sqrt(max(0.0, cov[2, 2])),
            })

        # Final landmarks
        landmarks = {}
        for tag_id, lm in self.ekf.get_all_landmarks().items():
            l_cov = lm.covariance
            landmarks[tag_id] = {
                'tag_id': tag_id,
                'x': lm.x,
                'y': lm.y,
                'sigma_x': math.sqrt(max(0.0, l_cov[0, 0])),
                'sigma_y': math.sqrt(max(0.0, l_cov[1, 1])),
                'cov_xy': float(l_cov[0, 1]),
            }

        return {
            'trajectory': trajectory,
            'landmarks': landmarks,
            'vo_history': vo_history,
        }


def export_trajectory_csv(trajectory: List[dict], output_path: Union[str, Path]):
    """Export robot trajectory list to CSV.

    Args:
        trajectory: Ordered robot state records containing xy positions in meters.
        output_path: Destination CSV or JSON file path.

    Returns:
        None. State changes and publications are performed in place.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp_s', 'x_m', 'y_m', 'theta_rad', 'sigma_x_m', 'sigma_y_m', 'sigma_theta_rad'])
        for item in trajectory:
            writer.writerow([
                f"{item['timestamp_s']:.4f}",
                f"{item['x']:.5f}",
                f"{item['y']:.5f}",
                f"{item['theta']:.5f}",
                f"{item['sigma_x']:.5f}",
                f"{item['sigma_y']:.5f}",
                f"{item['sigma_theta']:.5f}",
            ])


def export_landmarks_csv(landmarks: Dict[int, dict], output_path: Union[str, Path]):
    """Export landmark dictionary to CSV.

    Args:
        landmarks: Tag-ID-indexed landmark records with positions in meters.
        output_path: Destination CSV or JSON file path.

    Returns:
        None. State changes and publications are performed in place.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['tag_id', 'x_m', 'y_m', 'sigma_x_m', 'sigma_y_m', 'cov_xy'])
        for tag_id, item in sorted(landmarks.items()):
            writer.writerow([
                item['tag_id'],
                f"{item['x']:.5f}",
                f"{item['y']:.5f}",
                f"{item['sigma_x']:.5f}",
                f"{item['sigma_y']:.5f}",
                f"{item['cov_xy']:.6f}",
            ])
