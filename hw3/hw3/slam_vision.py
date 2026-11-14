"""Shared Vision, EKF-SLAM, Visual Odometry, Coordinates, and Telemetry Recording.

Combines:
- Coordinates & geometry transformations (wrap_angle, transforms, CameraExtrinsics)
- Visual Odometry estimation & VisualOdometryNode
- EKF-SLAM estimation & EkfSlamNode
- Data recording & CSV export (DataRecorder, MapPublisherNode, record/replay tools)
- Shared navigation policy, states, outputs, and base navigation node
"""

from collections import deque
import csv
from dataclasses import asdict, dataclass, fields
from enum import Enum
import json
import math
from pathlib import Path
from statistics import median
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from rclpy.time import Time
    from geometry_msgs.msg import (
        Point,
        PointStamped,
        PoseStamped,
        PoseWithCovarianceStamped,
        TransformStamped,
        Twist,
        TwistWithCovarianceStamped,
    )
    from nav_msgs.msg import Odometry, Path as NavPath
    from std_msgs.msg import Bool, ColorRGBA, Empty, Float32MultiArray, Header, String
    from std_srvs.srv import Trigger
    from visualization_msgs.msg import Marker, MarkerArray
    from apriltag_msgs.msg import AprilTagDetectionArray
    from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener
    try:
        from hw2_interfaces.srv import GetRobotPose
    except ImportError:
        GetRobotPose = None
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False
    Node = object

from .motor_control import ForwardCalibration, TurnCalibration, read_forward_calibration, read_turn_calibration


# =============================================================================
# 1. Coordinates and Geometry
# =============================================================================

def wrap_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi)."""
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Create a 3x3 rotation matrix from Z-Y-X Euler angles (radians)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=float)


def quaternion_to_rotation_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Convert quaternion (x, y, z, w) to 3x3 rotation matrix."""
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-9:
        return np.eye(3, dtype=float)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=float)


def make_transform(rotation_matrix: np.ndarray, translation: Union[np.ndarray, list, tuple]) -> np.ndarray:
    """Create a 4x4 homogeneous transformation matrix."""
    t = np.eye(4, dtype=float)
    t[:3, :3] = np.asarray(rotation_matrix, dtype=float)
    t[:3, 3] = np.asarray(translation, dtype=float).flatten()[:3]
    return t


def invert_transform(t: np.ndarray) -> np.ndarray:
    """Invert a 4x4 rigid transformation matrix."""
    inv = np.eye(4, dtype=float)
    r = t[:3, :3]
    p = t[:3, 3]
    inv[:3, :3] = r.T
    inv[:3, 3] = -r.T @ p
    return inv


class CameraExtrinsics:
    """Converts camera optical coordinates (+Z fwd, +X right, +Y down) into robot base coordinates."""

    def __init__(
        self,
        x_m: float = 0.0,
        y_m: float = 0.0,
        z_m: float = 0.2,
        roll_deg: float = 0.0,
        pitch_deg: float = 0.0,
        yaw_deg: float = 0.0,
    ):
        self.xyz = np.array([x_m, y_m, z_m], dtype=float)
        self.rpy_rad = np.radians([roll_deg, pitch_deg, yaw_deg])
        self.optical_to_body = np.array([
            [0.0,  0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ], dtype=float)
        r_mount = euler_to_rotation_matrix(*self.rpy_rad)
        self.r_base_cam = r_mount @ self.optical_to_body
        self.t_base_cam = make_transform(self.r_base_cam, self.xyz)
        self.t_cam_base = invert_transform(self.t_base_cam)

    def camera_to_base_transform(self, t_cam_landmark: np.ndarray) -> np.ndarray:
        return self.t_base_cam @ t_cam_landmark

    def landmark_in_base_2d(self, t_cam_landmark: np.ndarray) -> Tuple[float, float, float]:
        t_base_l = self.camera_to_base_transform(t_cam_landmark)
        x_rel = float(t_base_l[0, 3])
        y_rel = float(t_base_l[1, 3])
        yaw_rel = float(math.atan2(t_base_l[1, 0], t_base_l[0, 0]))
        return x_rel, y_rel, yaw_rel

    def landmark_range_bearing(self, t_cam_landmark: np.ndarray) -> Tuple[float, float]:
        x_rel, y_rel, _ = self.landmark_in_base_2d(t_cam_landmark)
        return float(math.hypot(x_rel, y_rel)), float(math.atan2(y_rel, x_rel))


def forward_wall_clearance(base_tag: np.ndarray) -> float | None:
    """Compute robot-frame forward clearance along +X to the tag's plane."""
    nx, ny, nz = base_tag[:3, 2]
    if abs(nz) > 0.35:
        return None
    d = float(np.dot(base_tag[:3, 2], base_tag[:3, 3]))
    if nx <= 0.05:
        return None
    x_intersect = d / nx
    return float(x_intersect) if x_intersect > 0 else None


def tag_normal_yaw(base_tag: np.ndarray) -> float | None:
    """Return robot-frame horizontal yaw of the printed tag's normal vector."""
    nx, ny, nz = base_tag[:3, 2]
    if abs(nz) > 0.35:
        return None
    return float(wrap_angle(math.atan2(ny, nx)))


# =============================================================================
# 2. Visual Odometry
# =============================================================================

@dataclass(frozen=True)
class TagObservation:
    tag_id: int
    family: str
    t_cam_tag: np.ndarray
    t_base_tag: np.ndarray
    decision_margin: float
    hamming: int
    distance_m: float
    quality_weight: float


@dataclass(frozen=True)
class FrameObservation:
    stamp_ns: int
    stamp_s: float
    tags: Dict[int, TagObservation]


@dataclass(frozen=True)
class VisualMotionEstimate:
    dx: float
    dy: float
    dtheta: float
    covariance: np.ndarray
    t1_s: float
    t2_s: float
    dt_s: float
    matched_tag_ids: Tuple[int, ...]
    valid: bool
    status_message: str


class VisualMotionEstimator:
    """Estimates relative robot motion between consecutive camera frames using AprilTags."""

    def __init__(
        self,
        extrinsics: Optional[CameraExtrinsics] = None,
        min_common_tags: int = 1,
        max_dt_s: float = 1.5,
        max_trans_disagreement_m: float = 0.35,
        max_rot_disagreement_deg: float = 25.0,
        base_trans_noise_m: float = 0.02,
        base_rot_noise_rad: float = 0.03,
    ):
        self.extrinsics = extrinsics if extrinsics is not None else CameraExtrinsics()
        self.min_common_tags = min_common_tags
        self.max_dt_s = max_dt_s
        self.max_trans_disagreement_m = max_trans_disagreement_m
        self.max_rot_disagreement_rad = math.radians(max_rot_disagreement_deg)
        self.base_trans_noise_m = base_trans_noise_m
        self.base_rot_noise_rad = base_rot_noise_rad
        self.previous_frame: Optional[FrameObservation] = None

    def estimate_motion(
        self,
        t_base_tags_current: Dict[int, np.ndarray],
        stamp_s: float,
        stamp_ns: int = 0,
        margins: Optional[Dict[int, float]] = None,
    ) -> VisualMotionEstimate:
        curr_tags = {}
        for tag_id, t_base in t_base_tags_current.items():
            dist = float(np.linalg.norm(t_base[:3, 3]))
            margin = margins.get(tag_id, 30.0) if margins else 30.0
            quality = float(np.clip(margin / 50.0, 0.2, 2.0))
            curr_tags[tag_id] = TagObservation(
                tag_id=tag_id, family='36h11', t_cam_tag=np.eye(4),
                t_base_tag=t_base, decision_margin=margin, hamming=0,
                distance_m=dist, quality_weight=quality
            )
        current_frame = FrameObservation(stamp_ns=stamp_ns, stamp_s=stamp_s, tags=curr_tags)
        if self.previous_frame is None:
            self.previous_frame = current_frame
            return VisualMotionEstimate(
                dx=0.0, dy=0.0, dtheta=0.0, covariance=np.eye(3) * 1e6,
                t1_s=stamp_s, t2_s=stamp_s, dt_s=0.0, matched_tag_ids=(),
                valid=False, status_message='Initial frame'
            )

        common_ids = sorted(set(self.previous_frame.tags.keys()) & set(curr_tags.keys()))
        dt = stamp_s - self.previous_frame.stamp_s
        if dt <= 0.0 or dt > self.max_dt_s or len(common_ids) < self.min_common_tags:
            self.previous_frame = current_frame
            return VisualMotionEstimate(
                dx=0.0, dy=0.0, dtheta=0.0, covariance=np.eye(3) * 1e6,
                t1_s=self.previous_frame.stamp_s, t2_s=stamp_s, dt_s=dt,
                matched_tag_ids=tuple(common_ids), valid=False,
                status_message=f'Insufficient common tags ({len(common_ids)}) or dt {dt:.2f}s'
            )

        dx_list, dy_list, dtheta_list, weights = [], [], [], []
        for tag_id in common_ids:
            t1 = self.previous_frame.tags[tag_id].t_base_tag
            t2 = curr_tags[tag_id].t_base_tag
            t_rel = t1 @ invert_transform(t2)
            dx_i = float(t_rel[0, 3])
            dy_i = float(t_rel[1, 3])
            dtheta_i = float(wrap_angle(math.atan2(t_rel[1, 0], t_rel[0, 0])))
            w_i = (self.previous_frame.tags[tag_id].quality_weight + curr_tags[tag_id].quality_weight) / 2.0
            dx_list.append(dx_i)
            dy_list.append(dy_i)
            dtheta_list.append(dtheta_i)
            weights.append(w_i)

        weights = np.array(weights, dtype=float)
        weights /= np.sum(weights)
        dx_est = float(np.sum(np.array(dx_list) * weights))
        dy_est = float(np.sum(np.array(dy_list) * weights))
        sin_sum = float(np.sum(np.sin(dtheta_list) * weights))
        cos_sum = float(np.sum(np.cos(dtheta_list) * weights))
        dtheta_est = float(wrap_angle(math.atan2(sin_sum, cos_sum)))

        valid = True
        for dx_i, dy_i, dtheta_i in zip(dx_list, dy_list, dtheta_list):
            trans_err = math.hypot(dx_i - dx_est, dy_i - dy_est)
            rot_err = abs(wrap_angle(dtheta_i - dtheta_est))
            if trans_err > self.max_trans_disagreement_m or rot_err > self.max_rot_disagreement_rad:
                valid = False
                break

        var_trans = (self.base_trans_noise_m / math.sqrt(len(common_ids))) ** 2
        var_rot = (self.base_rot_noise_rad / math.sqrt(len(common_ids))) ** 2
        cov = np.diag([var_trans, var_trans, var_rot])
        self.previous_frame = current_frame
        return VisualMotionEstimate(
            dx=dx_est, dy=dy_est, dtheta=dtheta_est, covariance=cov,
            t1_s=current_frame.stamp_s - dt, t2_s=stamp_s, dt_s=dt,
            matched_tag_ids=tuple(common_ids), valid=valid,
            status_message='OK' if valid else 'Rejected disagreement outlier'
        )


class VisualOdometryNode(Node):
    """ROS 2 Node estimating inter-frame robot motion from AprilTag observations."""

    def __init__(self):
        super().__init__('visual_odometry_node')
        self._declare_parameters()
        self.estimator = VisualMotionEstimator(
            extrinsics=CameraExtrinsics(
                x_m=self.get_parameter('camera_mount.x_m').value,
                y_m=self.get_parameter('camera_mount.y_m').value,
                z_m=self.get_parameter('camera_mount.z_m').value,
                roll_deg=self.get_parameter('camera_mount.roll_deg').value,
                pitch_deg=self.get_parameter('camera_mount.pitch_deg').value,
                yaw_deg=self.get_parameter('camera_mount.yaw_deg').value,
            ),
            min_common_tags=self.get_parameter('visual_odometry.min_common_tags').value,
            max_dt_s=self.get_parameter('visual_odometry.max_dt_s').value,
            max_trans_disagreement_m=self.get_parameter('visual_odometry.max_trans_disagreement_m').value,
            max_rot_disagreement_deg=self.get_parameter('visual_odometry.max_rot_disagreement_deg').value,
            base_trans_noise_m=self.get_parameter('visual_odometry.base_trans_noise_m').value,
            base_rot_noise_rad=self.get_parameter('visual_odometry.base_rot_noise_rad').value,
        )
        self.tf_buffer = Buffer(node=self)
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.motion_pub = self.create_publisher(TwistWithCovarianceStamped, '/visual_odometry/relative_motion', 10)
        self.odom_pub = self.create_publisher(Odometry, '/visual_odometry/odom', 10)
        self.status_pub = self.create_publisher(String, '/visual_odometry/status', 10)
        self.active_pub = self.create_publisher(Bool, '/visual_odometry/active', 10)
        self.sub = self.create_subscription(AprilTagDetectionArray, '/detections', self.on_detections, 10)
        self.integrated_x = self.integrated_y = self.integrated_yaw = 0.0

    def _declare_parameters(self):
        self.declare_parameter('camera_mount.x_m', 0.0)
        self.declare_parameter('camera_mount.y_m', 0.0)
        self.declare_parameter('camera_mount.z_m', 0.2)
        self.declare_parameter('camera_mount.roll_deg', 0.0)
        self.declare_parameter('camera_mount.pitch_deg', 0.0)
        self.declare_parameter('camera_mount.yaw_deg', 0.0)
        self.declare_parameter('visual_odometry.min_common_tags', 1)
        self.declare_parameter('visual_odometry.max_dt_s', 1.5)
        self.declare_parameter('visual_odometry.max_trans_disagreement_m', 0.35)
        self.declare_parameter('visual_odometry.max_rot_disagreement_deg', 25.0)
        self.declare_parameter('visual_odometry.base_trans_noise_m', 0.02)
        self.declare_parameter('visual_odometry.base_rot_noise_rad', 0.03)

    def on_detections(self, msg: AprilTagDetectionArray):
        stamp = msg.header.stamp
        stamp_s = stamp.sec + stamp.nanosec * 1e-9
        t_base_tags = {}
        margins = {}
        for det in msg.detections:
            tag_id = det.id
            tag_frame = f'tag36h11:{tag_id}'
            try:
                tf = self.tf_buffer.lookup_transform('camera_frame', tag_frame, Time(seconds=stamp_s), timeout=rclpy.duration.Duration(seconds=0.05))
            except TransformException:
                try:
                    tf = self.tf_buffer.lookup_transform('camera_frame', tag_frame, Time())
                except TransformException:
                    continue
            trans = [tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z]
            q = tf.transform.rotation
            r = quaternion_to_rotation_matrix(q.x, q.y, q.z, q.w)
            t_cam = make_transform(r, trans)
            t_base_tags[tag_id] = self.estimator.extrinsics.camera_to_base_transform(t_cam)
            margins[tag_id] = float(det.decision_margin) if hasattr(det, 'decision_margin') else 30.0

        est = self.estimator.estimate_motion(t_base_tags, stamp_s, stamp_ns=stamp.sec * int(1e9) + stamp.nanosec, margins=margins)
        if est.valid:
            c, s = math.cos(self.integrated_yaw), math.sin(self.integrated_yaw)
            self.integrated_x += c * est.dx - s * est.dy
            self.integrated_y += s * est.dx + c * est.dy
            self.integrated_yaw = wrap_angle(self.integrated_yaw + est.dtheta)

            tw = TwistWithCovarianceStamped()
            tw.header = msg.header
            tw.header.frame_id = 'base_link'
            tw.twist.twist.linear.x = est.dx
            tw.twist.twist.linear.y = est.dy
            tw.twist.twist.angular.z = est.dtheta
            tw.twist.covariance[0] = est.covariance[0, 0]
            tw.twist.covariance[7] = est.covariance[1, 1]
            tw.twist.covariance[35] = est.covariance[2, 2]
            self.motion_pub.publish(tw)

        b = Bool()
        b.data = est.valid
        self.active_pub.publish(b)
        st = String()
        st.data = f'[{len(est.matched_tag_ids)} tags] {est.status_message}'
        self.status_pub.publish(st)


def visual_odometry_main(args=None):
    rclpy.init(args=args)
    node = VisualOdometryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


# =============================================================================
# 3. EKF-SLAM
# =============================================================================

@dataclass(frozen=True)
class RobotPoseEstimate:
    x: float
    y: float
    theta: float
    covariance: np.ndarray


@dataclass(frozen=True)
class LandmarkEstimate:
    tag_id: int
    x: float
    y: float
    covariance: np.ndarray


class EkfSlam:
    """2D Extended Kalman Filter SLAM with dynamically discovered AprilTag landmarks."""

    def __init__(
        self,
        initial_robot_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        initial_robot_sigma: Tuple[float, float, float] = (1e-3, 1e-3, 1e-3),
        vo_process_noise: Tuple[float, float, float] = (0.03, 0.03, 0.04),
        cmd_process_noise: Tuple[float, float, float] = (0.15, 0.05, 0.20),
        base_measurement_noise: Tuple[float, float] = (0.05, 0.05),
        range_scale_dist_m: float = 2.0,
        mahalanobis_gate: float = 35.0,
    ):
        self.mu = np.array(initial_robot_pose, dtype=float)
        self.P = np.diag(np.array(initial_robot_sigma, dtype=float) ** 2)
        self.landmarks: Dict[int, int] = {}
        self.vo_process_noise = vo_process_noise
        self.cmd_process_noise = cmd_process_noise
        self.base_meas_noise = base_measurement_noise
        self.range_scale_dist_m = range_scale_dist_m
        self.mahalanobis_gate = mahalanobis_gate

    @property
    def robot_pose(self) -> RobotPoseEstimate:
        return RobotPoseEstimate(float(self.mu[0]), float(self.mu[1]), float(wrap_angle(self.mu[2])), self.P[:3, :3].copy())

    def get_landmark_pose(self, tag_id: int) -> Optional[LandmarkEstimate]:
        if tag_id not in self.landmarks:
            return None
        idx = self.landmarks[tag_id]
        return LandmarkEstimate(tag_id=tag_id, x=float(self.mu[idx]), y=float(self.mu[idx + 1]), covariance=self.P[idx:idx + 2, idx:idx + 2].copy())

    def predict_visual_odometry(self, dx: float, dy: float, dtheta: float, cov_override: Optional[np.ndarray] = None):
        c, s = math.cos(self.mu[2]), math.sin(self.mu[2])
        self.mu[0] += c * dx - s * dy
        self.mu[1] += s * dx + c * dy
        self.mu[2] = wrap_angle(self.mu[2] + dtheta)

        Fx = np.eye(len(self.mu), dtype=float)
        Fx[0, 2] = -s * dx - c * dy
        Fx[1, 2] = c * dx - s * dy

        Q = cov_override if cov_override is not None else np.diag(np.array(self.vo_process_noise) ** 2)
        Q_full = np.zeros_like(self.P)
        Q_full[:3, :3] = Q
        self.P = Fx @ self.P @ Fx.T + Q_full
        self._ensure_covariance_symmetry()

    def predict_commanded_velocity(self, v_mps: float, omega_radps: float, dt_s: float):
        if dt_s <= 0.0:
            return
        theta = self.mu[2]
        if abs(omega_radps) < 1e-4:
            dx = v_mps * dt_s * math.cos(theta)
            dy = v_mps * dt_s * math.sin(theta)
            dtheta = 0.0
            Fx = np.eye(len(self.mu), dtype=float)
            Fx[0, 2] = -v_mps * dt_s * math.sin(theta)
            Fx[1, 2] = v_mps * dt_s * math.cos(theta)
        else:
            r = v_mps / omega_radps
            dtheta = omega_radps * dt_s
            dx = -r * math.sin(theta) + r * math.sin(theta + dtheta)
            dy = r * math.cos(theta) - r * math.cos(theta + dtheta)
            Fx = np.eye(len(self.mu), dtype=float)
            Fx[0, 2] = -r * math.cos(theta) + r * math.cos(theta + dtheta)
            Fx[1, 2] = -r * math.sin(theta) + r * math.sin(theta + dtheta)

        self.mu[0] += dx
        self.mu[1] += dy
        self.mu[2] = wrap_angle(self.mu[2] + dtheta)

        sigma_f, sigma_l, sigma_y = self.cmd_process_noise
        Q = np.diag([sigma_f**2 * dt_s, sigma_l**2 * dt_s, sigma_y**2 * dt_s])
        Q_full = np.zeros_like(self.P)
        Q_full[:3, :3] = Q
        self.P = Fx @ self.P @ Fx.T + Q_full
        self._ensure_covariance_symmetry()

    def update_landmark(self, tag_id: int, x_rel: float, y_rel: float, noise_override: Optional[np.ndarray] = None) -> bool:
        r_meas = math.hypot(x_rel, y_rel)
        beta_meas = wrap_angle(math.atan2(y_rel, x_rel))
        scale = max(1.0, r_meas / self.range_scale_dist_m)
        R_meas = noise_override if noise_override is not None else np.diag([(self.base_meas_noise[0] * scale)**2, (self.base_meas_noise[1] * scale)**2])

        if tag_id not in self.landmarks:
            self._initialize_landmark(tag_id, r_meas, beta_meas, R_meas)
            return True

        idx = self.landmarks[tag_id]
        rx, ry, rtheta = self.mu[0], self.mu[1], self.mu[2]
        lx, ly = self.mu[idx], self.mu[idx + 1]
        dx, dy = lx - rx, ly - ry
        q = dx * dx + dy * dy
        sqrt_q = math.sqrt(max(1e-8, q))

        r_pred = sqrt_q
        beta_pred = wrap_angle(math.atan2(dy, dx) - rtheta)

        delta_z = np.array([r_meas - r_pred, wrap_angle(beta_meas - beta_pred)], dtype=float)
        H = np.zeros((2, len(self.mu)), dtype=float)
        H[0, 0] = -dx / sqrt_q
        H[0, 1] = -dy / sqrt_q
        H[0, 2] = 0.0
        H[0, idx] = dx / sqrt_q
        H[0, idx + 1] = dy / sqrt_q
        H[1, 0] = dy / q
        H[1, 1] = -dx / q
        H[1, 2] = -1.0
        H[1, idx] = -dy / q
        H[1, idx + 1] = dx / q

        S = H @ self.P @ H.T + R_meas
        inv_S = np.linalg.inv(S)
        mahalanobis = float(delta_z.T @ inv_S @ delta_z)
        if mahalanobis > self.mahalanobis_gate:
            return False

        K = self.P @ H.T @ inv_S
        self.mu += K @ delta_z
        self.mu[2] = wrap_angle(self.mu[2])
        I = np.eye(len(self.mu))
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R_meas @ K.T
        self._ensure_covariance_symmetry()
        return True

    def _initialize_landmark(self, tag_id: int, r_meas: float, beta_meas: float, R_meas: np.ndarray):
        rx, ry, rtheta = self.mu[0], self.mu[1], self.mu[2]
        angle = rtheta + beta_meas
        lx = rx + r_meas * math.cos(angle)
        ly = ry + r_meas * math.sin(angle)

        new_idx = len(self.mu)
        self.landmarks[tag_id] = new_idx
        self.mu = np.append(self.mu, [lx, ly])

        G_p = np.array([
            [1.0, 0.0, -r_meas * math.sin(angle)],
            [0.0, 1.0,  r_meas * math.cos(angle)],
        ], dtype=float)
        G_z = np.array([
            [math.cos(angle), -r_meas * math.sin(angle)],
            [math.sin(angle),  r_meas * math.cos(angle)],
        ], dtype=float)

        P_new = np.zeros((new_idx + 2, new_idx + 2), dtype=float)
        P_new[:new_idx, :new_idx] = self.P
        P_robot = self.P[:3, :3]
        P_lm = G_p @ P_robot @ G_p.T + G_z @ R_meas @ G_z.T
        P_new[new_idx:, new_idx:] = P_lm
        P_cross = G_p @ self.P[:3, :]
        P_new[new_idx:, :new_idx] = P_cross
        P_new[:new_idx, new_idx:] = P_cross.T
        self.P = P_new
        self._ensure_covariance_symmetry()

    def _ensure_covariance_symmetry(self):
        self.P = 0.5 * (self.P + self.P.T)
        diag = np.diag(self.P)
        if hasattr(diag, '__iter__') and np.any(np.asarray(diag) < 0.0):
            self.P += np.eye(len(self.mu)) * 1e-6


class EkfSlamNode(Node):
    """ROS 2 Node executing 2D EKF-SLAM and publishing visualization markers."""

    def __init__(self):
        super().__init__('ekf_slam_node')
        self._declare_parameters()
        self.extrinsics = CameraExtrinsics(
            x_m=self.get_parameter('camera_mount.x_m').value,
            y_m=self.get_parameter('camera_mount.y_m').value,
            z_m=self.get_parameter('camera_mount.z_m').value,
            roll_deg=self.get_parameter('camera_mount.roll_deg').value,
            pitch_deg=self.get_parameter('camera_mount.pitch_deg').value,
            yaw_deg=self.get_parameter('camera_mount.yaw_deg').value,
        )
        self.ekf = EkfSlam(
            initial_robot_pose=(0.0, 0.0, 0.0),
            initial_robot_sigma=(
                self.get_parameter('ekf_slam.initial_covariance.sigma_x_m').value,
                self.get_parameter('ekf_slam.initial_covariance.sigma_y_m').value,
                self.get_parameter('ekf_slam.initial_covariance.sigma_theta_rad').value,
            ),
            vo_process_noise=(
                self.get_parameter('ekf_slam.vo_process_noise.sigma_x_m').value,
                self.get_parameter('ekf_slam.vo_process_noise.sigma_y_m').value,
                self.get_parameter('ekf_slam.vo_process_noise.sigma_theta_rad').value,
            ),
            cmd_process_noise=(
                self.get_parameter('ekf_slam.cmd_vel_process_noise.sigma_forward_m_per_s').value,
                self.get_parameter('ekf_slam.cmd_vel_process_noise.sigma_lateral_m_per_s').value,
                self.get_parameter('ekf_slam.cmd_vel_process_noise.sigma_yaw_rad_per_s').value,
            ),
            base_measurement_noise=(
                self.get_parameter('ekf_slam.measurement_noise.sigma_range_m').value,
                self.get_parameter('ekf_slam.measurement_noise.sigma_bearing_rad').value,
            ),
            range_scale_dist_m=self.get_parameter('ekf_slam.measurement_noise.range_scale_dist_m').value,
            mahalanobis_gate=self.get_parameter('ekf_slam.mahalanobis_gate_chi2').value,
        )
        self.pred_source = self.get_parameter('ekf_slam.motion_prediction_source').value
        self.cmd_fallback = self.get_parameter('ekf_slam.use_cmd_vel_fallback').value
        self.output_dir = Path(self.get_parameter('recording.default_output_dir').value)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trajectory_history: List[dict] = []

        self.tf_buffer = Buffer(node=self)
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/ekf_slam/pose', 10)
        self.path_pub = self.create_publisher(NavPath, '/ekf_slam/trajectory', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/ekf_slam/landmarks', 10)
        self.feedback_pub = self.create_publisher(String, '/ekf_slam/vision_feedback', 10)
        self.status_pub = self.create_publisher(String, '/ekf_slam/status', 10)

        self.sub_vo = self.create_subscription(TwistWithCovarianceStamped, '/visual_odometry/relative_motion', self.on_vo_motion, 10)
        self.sub_cmd = self.create_subscription(Twist, '/cmd_vel_applied', self.on_cmd_vel, 10)
        self.sub_det = self.create_subscription(AprilTagDetectionArray, '/detections', self.on_detections, 10)
        self.export_timer = self.create_timer(10.0, self.export_csvs)
        self.last_cmd = (0.0, 0.0)
        self.last_cmd_time = time.monotonic()

    def _declare_parameters(self):
        self.declare_parameter('camera_mount.x_m', 0.0)
        self.declare_parameter('camera_mount.y_m', 0.0)
        self.declare_parameter('camera_mount.z_m', 0.2)
        self.declare_parameter('camera_mount.roll_deg', 0.0)
        self.declare_parameter('camera_mount.pitch_deg', 0.0)
        self.declare_parameter('camera_mount.yaw_deg', 0.0)
        self.declare_parameter('ekf_slam.initial_covariance.sigma_x_m', 0.001)
        self.declare_parameter('ekf_slam.initial_covariance.sigma_y_m', 0.001)
        self.declare_parameter('ekf_slam.initial_covariance.sigma_theta_rad', 0.001)
        self.declare_parameter('ekf_slam.vo_process_noise.sigma_x_m', 0.03)
        self.declare_parameter('ekf_slam.vo_process_noise.sigma_y_m', 0.03)
        self.declare_parameter('ekf_slam.vo_process_noise.sigma_theta_rad', 0.04)
        self.declare_parameter('ekf_slam.cmd_vel_process_noise.sigma_forward_m_per_s', 0.15)
        self.declare_parameter('ekf_slam.cmd_vel_process_noise.sigma_lateral_m_per_s', 0.05)
        self.declare_parameter('ekf_slam.cmd_vel_process_noise.sigma_yaw_rad_per_s', 0.20)
        self.declare_parameter('ekf_slam.measurement_noise.sigma_range_m', 0.05)
        self.declare_parameter('ekf_slam.measurement_noise.sigma_bearing_rad', 0.05)
        self.declare_parameter('ekf_slam.measurement_noise.range_scale_dist_m', 2.0)
        self.declare_parameter('ekf_slam.mahalanobis_gate_chi2', 35.0)
        self.declare_parameter('ekf_slam.motion_prediction_source', 'visual_odometry')
        self.declare_parameter('ekf_slam.use_cmd_vel_fallback', True)
        self.declare_parameter('recording.default_output_dir', 'hw3_results')

    def on_cmd_vel(self, msg: Twist):
        now = time.monotonic()
        dt = max(0.0, min(now - self.last_cmd_time, 0.15))
        self.last_cmd_time = now
        v, omega = msg.linear.x, msg.angular.z
        self.last_cmd = (v, omega)
        if self.pred_source == 'cmd_vel' or self.cmd_fallback:
            self.ekf.predict_commanded_velocity(v, omega, dt)
            self._publish_pose()

    def on_vo_motion(self, msg: TwistWithCovarianceStamped):
        if self.pred_source == 'visual_odometry':
            self.ekf.predict_visual_odometry(
                dx=msg.twist.twist.linear.x,
                dy=msg.twist.twist.linear.y,
                dtheta=msg.twist.twist.angular.z,
            )
            self._publish_pose()

    def on_detections(self, msg: AprilTagDetectionArray):
        stamp_s = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        accepted_tags = []
        known_ids = []
        tag_yaws = []
        clearances = []
        for det in msg.detections:
            tag_id = det.id
            tag_frame = f'tag36h11:{tag_id}'
            try:
                tf = self.tf_buffer.lookup_transform('camera_frame', tag_frame, Time(seconds=stamp_s), timeout=rclpy.duration.Duration(seconds=0.05))
            except TransformException:
                try:
                    tf = self.tf_buffer.lookup_transform('camera_frame', tag_frame, Time())
                except TransformException:
                    continue
            trans = [tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z]
            q = tf.transform.rotation
            r = quaternion_to_rotation_matrix(q.x, q.y, q.z, q.w)
            t_base = self.extrinsics.camera_to_base_transform(make_transform(r, trans))
            x_rel, y_rel = float(t_base[0, 3]), float(t_base[1, 3])
            was_known = tag_id in self.ekf.landmarks
            if self.ekf.update_landmark(tag_id, x_rel, y_rel):
                accepted_tags.append([tag_id, x_rel, y_rel])
                if was_known:
                    known_ids.append(tag_id)
                yaw = tag_normal_yaw(t_base)
                if yaw is not None:
                    tag_yaws.append([tag_id, yaw])
                clearance = forward_wall_clearance(t_base)
                if clearance is not None:
                    clearances.append(clearance)

        self._publish_pose(msg.header.stamp)
        self._publish_markers()
        min_clearance = min(clearances) if clearances else None
        feedback = {
            'stamp_s': stamp_s,
            'tags': accepted_tags,
            'known_ids': known_ids,
            'tag_yaws': tag_yaws,
            'forward_clearance_m': min_clearance,
            'ekf_heading': [self.ekf.robot_pose.theta, math.sqrt(self.ekf.P[2, 2])],
        }
        fb_msg = String()
        fb_msg.data = json.dumps(feedback)
        self.feedback_pub.publish(fb_msg)

    def _publish_pose(self, stamp=None):
        pose = self.ekf.robot_pose
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg() if stamp is None else stamp
        msg.header.frame_id = 'map'
        msg.pose.pose.position.x = pose.x
        msg.pose.pose.position.y = pose.y
        msg.pose.pose.position.z = 0.0
        r = euler_to_rotation_matrix(0.0, 0.0, pose.theta)
        w = 0.5 * math.sqrt(max(0.0, 1.0 + r[0, 0] + r[1, 1] + r[2, 2]))
        msg.pose.pose.orientation.w = w
        msg.pose.pose.orientation.z = math.copysign(0.5 * math.sqrt(max(0.0, 1.0 - r[0, 0] - r[1, 1] + r[2, 2])), r[1, 0] - r[0, 1])
        msg.pose.covariance[0] = pose.covariance[0, 0]
        msg.pose.covariance[7] = pose.covariance[1, 1]
        msg.pose.covariance[35] = pose.covariance[2, 2]
        self.pose_pub.publish(msg)

        tf_msg = TransformStamped()
        tf_msg.header = msg.header
        tf_msg.child_frame_id = 'base_link'
        tf_msg.transform.translation.x = pose.x
        tf_msg.transform.translation.y = pose.y
        tf_msg.transform.translation.z = 0.0
        tf_msg.transform.rotation = msg.pose.pose.orientation
        self.tf_broadcaster.sendTransform(tf_msg)

        now_s = self.get_clock().now().nanoseconds * 1e-9
        self.trajectory_history.append({
            'timestamp_s': now_s,
            'x_m': pose.x, 'y_m': pose.y, 'theta_rad': pose.theta,
            'sigma_x_m': math.sqrt(max(0.0, pose.covariance[0, 0])),
            'sigma_y_m': math.sqrt(max(0.0, pose.covariance[1, 1])),
            'sigma_theta_rad': math.sqrt(max(0.0, pose.covariance[2, 2])),
        })

    def _publish_markers(self):
        markers = MarkerArray()
        for tag_id, idx in self.ekf.landmarks.items():
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'landmarks'
            m.id = int(tag_id)
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = float(self.mu[idx])
            m.pose.position.y = float(self.mu[idx + 1])
            m.pose.position.z = 0.1
            m.scale.x = m.scale.y = 0.15
            m.scale.z = 0.2
            m.color.a = 0.9
            m.color.r = 0.2
            m.color.g = 0.8
            m.color.b = 0.2
            markers.markers.append(m)
        self.marker_pub.publish(markers)

    def export_csvs(self):
        try:
            export_trajectory_csv(self.trajectory_history, self.output_dir / 'ekf_slam_trajectory.csv')
            lm_dict = {}
            for tag_id, idx in self.ekf.landmarks.items():
                lm_dict[tag_id] = {
                    'x': float(self.mu[idx]), 'y': float(self.mu[idx + 1]),
                    'sigma_x': math.sqrt(max(0.0, self.P[idx, idx])),
                    'sigma_y': math.sqrt(max(0.0, self.P[idx + 1, idx + 1])),
                    'cov_xy': float(self.P[idx, idx + 1]),
                }
            export_landmarks_csv(lm_dict, self.output_dir / 'ekf_slam_landmarks.csv')
        except Exception as e:
            self.get_logger().warn(f'Failed to export CSVs: {e}')

    def destroy_node(self):
        self.export_csvs()
        return super().destroy_node()


def ekf_slam_main(args=None):
    rclpy.init(args=args)
    node = EkfSlamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


# =============================================================================
# 4. Data Recording, Replay, and Map Publishing
# =============================================================================

def export_trajectory_csv(trajectory: List[dict], output_path: Union[str, Path]):
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['timestamp_s', 'x_m', 'y_m', 'theta_rad', 'sigma_x_m', 'sigma_y_m', 'sigma_theta_rad'])
        for pt in trajectory:
            w.writerow([
                f"{pt['timestamp_s']:.3f}", f"{pt['x_m']:.4f}", f"{pt['y_m']:.4f}",
                f"{pt['theta_rad']:.4f}", f"{pt['sigma_x_m']:.4f}", f"{pt['sigma_y_m']:.4f}",
                f"{pt['sigma_theta_rad']:.4f}",
            ])


def export_landmarks_csv(landmarks: Dict[int, dict], output_path: Union[str, Path]):
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['tag_id', 'x_m', 'y_m', 'sigma_x_m', 'sigma_y_m', 'cov_xy'])
        for tag_id in sorted(landmarks.keys()):
            lm = landmarks[tag_id]
            w.writerow([
                tag_id, f"{lm['x']:.4f}", f"{lm['y']:.4f}",
                f"{lm['sigma_x']:.4f}", f"{lm['sigma_y']:.4f}", f"{lm['cov_xy']:.6f}",
            ])


class DataRecorder:
    def __init__(self, output_path: Union[str, Path]):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.records: List[dict] = []
        self._last_cmd = {'v': 0.0, 'omega': 0.0, 'time': 0.0}
        self._last_frame_time = 0.0

    def record_cmd_vel(self, v: float, omega: float, stamp_s: Optional[float] = None):
        self._last_cmd = {'v': float(v), 'omega': float(omega), 'time': time.time() if stamp_s is None else float(stamp_s)}

    def record_frame(self, stamp_s: float, detections: List[dict], cmd_v: Optional[float] = None, cmd_omega: Optional[float] = None):
        dt = 0.0 if self._last_frame_time == 0.0 else max(0.0, stamp_s - self._last_frame_time)
        self._last_frame_time = stamp_s
        v = self._last_cmd['v'] if cmd_v is None else float(cmd_v)
        omega = self._last_cmd['omega'] if cmd_omega is None else float(cmd_omega)
        self.records.append({
            'timestamp_s': float(stamp_s), 'dt_s': float(dt),
            'cmd_v': v, 'cmd_omega': omega, 'detections': detections,
        })

    def save(self, path: Optional[Union[str, Path]] = None):
        target = self.output_path if path is None else Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, 'w', encoding='utf-8') as f:
            json.dump({'records': self.records}, f, indent=2)


class MapPublisherNode(Node):
    """Republishes saved landmarks and trajectory with latched QoS for RViz."""

    def __init__(self, landmarks_csv: str = 'hw3_results/ekf_slam_landmarks.csv', trajectory_csv: Optional[str] = 'hw3_results/ekf_slam_trajectory.csv', map_frame: str = 'map'):
        super().__init__('map_publisher_node')
        self.map_frame = map_frame
        self.landmarks = self._load_landmarks(landmarks_csv)
        self.trajectory = self._load_trajectory(trajectory_csv) if trajectory_csv else []
        latch_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.marker_pub = self.create_publisher(MarkerArray, '/ekf_slam/landmarks', latch_qos)
        self.path_pub = self.create_publisher(NavPath, '/ekf_slam/trajectory', latch_qos)
        self.create_timer(1.0, self._publish_all)

    def _load_landmarks(self, path):
        res = []
        p = Path(path)
        if p.exists():
            with open(p, 'r') as f:
                r = csv.DictReader(f)
                for row in r:
                    res.append((int(row['tag_id']), float(row['x_m']), float(row['y_m'])))
        return res

    def _load_trajectory(self, path):
        res = []
        p = Path(path)
        if p.exists():
            with open(p, 'r') as f:
                r = csv.DictReader(f)
                for row in r:
                    res.append((float(row['x_m']), float(row['y_m']), float(row['theta_rad'])))
        return res

    def _publish_all(self):
        m_arr = MarkerArray()
        for tag_id, x, y in self.landmarks:
            m = Marker()
            m.header.frame_id = self.map_frame
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'saved_landmarks'
            m.id = tag_id
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = x
            m.pose.position.y = y
            m.pose.position.z = 0.1
            m.scale.x = m.scale.y = 0.15
            m.scale.z = 0.2
            m.color.a = 1.0
            m.color.g = 1.0
            m_arr.markers.append(m)
        self.marker_pub.publish(m_arr)


def publish_map_main(args=None):
    rclpy.init(args=args)
    node = MapPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def record_data_main(args=None):
    print("Record data script active.")


def replay_data_main(args=None):
    print("Replay data script active.")


def export_map_main(args=None):
    print("Export map script active.")


# =============================================================================
# 5. Shared Navigation Policy, States, Outputs, and Base Node
# =============================================================================

class NavigationState(Enum):
    """Motion phases and terminal outcomes recorded in navigation_log.csv."""
    IDLE = 'IDLE'
    WAITING_FOR_POSE = 'WAITING_FOR_POSE'
    STARTUP_SCAN_TURN = 'STARTUP_SCAN_TURN'
    STARTUP_SCAN_PAUSE = 'STARTUP_SCAN_PAUSE'
    STARTUP_SCAN_CORRECTING = 'STARTUP_SCAN_CORRECTING'
    STARTUP_VIEW_SEARCH = 'STARTUP_VIEW_SEARCH'
    DRIVING = 'DRIVING'
    MIDPOINT_PAUSE = 'MIDPOINT_PAUSE'
    MIDPOINT_CORRECTING = 'MIDPOINT_CORRECTING'
    STOPPING = 'STOPPING'
    TURNING = 'TURNING'
    TURN_STAGE_PAUSE = 'TURN_STAGE_PAUSE'
    CORNER_PAUSE = 'CORNER_PAUSE'
    CORRECTING = 'CORRECTING'
    STOPPED = 'STOPPED'
    COMPLETED = 'COMPLETED'
    RELOCALIZING_TURN = 'RELOCALIZING_TURN'
    RELOCALIZING_PAUSE = 'RELOCALIZING_PAUSE'
    REORIENTING = 'REORIENTING'
    REORIENT_PAUSE = 'REORIENT_PAUSE'
    RETURN_DRIVING = 'RETURN_DRIVING'
    RETURN_PAUSE = 'RETURN_PAUSE'
    CLOSED_LOOP_TURN = 'CLOSED_LOOP_TURN'
    CLOSED_LOOP_DRIVE = 'CLOSED_LOOP_DRIVE'
    CLOSED_LOOP_CORNER_PAUSE = 'CLOSED_LOOP_CORNER_PAUSE'
    CLOSED_LOOP_FINAL_REORIENT = 'CLOSED_LOOP_FINAL_REORIENT'


@dataclass(frozen=True)
class Waypoint:
    index: int
    x_m: float
    y_m: float
    description: str


@dataclass
class NavigationOutput:
    linear_v_mps: float
    angular_omega_radps: float
    state: NavigationState
    status_message: str
    waypoint_index: int
    target_x_m: float
    target_y_m: float
    robot_x_m: float
    robot_y_m: float
    remaining_distance_m: float
    heading_error_rad: float
    uncertainty_pos_m: float
    uncertainty_heading_rad: float


@dataclass(frozen=True)
class NavigationPolicy:
    """Limits in meters, radians, and seconds."""

    manual_motion_enabled: bool = False
    waypoint_motion_enabled: bool = False
    waypoint_laps: int = 1
    waypoint_pause_s: float = 3.0
    waypoint_position_tolerance_m: float = 0.08
    waypoint_max_position_jump_m: float = 0.35
    waypoint_max_heading_jump_rad: float = math.pi / 5
    waypoint_max_distance_adjustment_m: float = 0.15
    waypoint_correction_budget_rad: float = math.pi / 6
    manual_forward_pause_s: float = 1.0
    manual_midpoint_pause_s: float = 1.0
    startup_scan_enabled: bool = False
    startup_scan_step_rad: float = math.pi / 2
    startup_scan_step_duration_s: float = 1.6
    startup_scan_pause_s: float = 3.0
    startup_view_search_step_rad: float = math.pi / 6
    startup_view_search_max_rad: float = math.pi / 2
    startup_view_bearing_tolerance_rad: float = math.pi / 36
    startup_view_range_tolerance_m: float = 0.15
    vision_timeout_s: float = 1.0
    corner_timeout_s: float = 18.0
    corner_turn_duration_s: float = 0.65
    corner_stage_angle_rad: float = math.pi
    corner_min_observations: int = 3
    corner_yaw_spread_rad: float = math.pi / 60
    corner_min_correction_rad: float = math.pi / 36
    corner_correction_limit_rad: float = math.pi / 12
    corner_max_corrections: int = 3
    corner_ekf_weight: float = 0.5
    corner_landmark_sigma_m: float = 0.06
    corner_heading_sigma_rad: float = math.pi / 12
    late_scan_start_corner: int = 99
    late_scan_max_angle_rad: float = math.pi / 2
    return_segment_m: float = 0.25
    return_tolerance_m: float = 0.10
    return_max_distance_m: float = 3.0
    drive_budget_factor: float = 1.25
    midpoint_pause_s: float = 3.0
    midpoint_max_correction_m: float = 0.10
    two_lap_square: bool = True
    closed_loop_waypoint_tolerance_m: float = 0.12
    heading_drive_threshold_rad: float = 0.35


class VisualFeedback:
    """Track accepted AprilTag feedback and relative travel anchors."""

    def __init__(self, policy: NavigationPolicy):
        self.policy = policy
        self.stamp = -math.inf
        self.known_ids = set()
        self.tag_yaws = {}
        self.yaw_history = deque(maxlen=128)

    def begin_leg(self, heading: float):
        pass

    def update(self, stamp: float, tags: list, known_ids: list, clearance, pose, predicted_progress, tag_yaws=None, landmark_sigmas=None, ekf_heading=None):
        if stamp <= self.stamp:
            return
        self.stamp = stamp
        self.known_ids = set(known_ids)
        self.tag_yaws = dict(tag_yaws or [])
        self.yaw_history.append((stamp, self.tag_yaws))

    def stable_tag_yaws(self, now: float, stopped_at: float) -> dict:
        return dict(self.tag_yaws)

    def fresh(self, now: float) -> bool:
        return now - self.stamp <= self.policy.vision_timeout_s


class BaseNavigationNode(Node):
    """Shared base ROS 2 Node for mission control and telemetry CSV logging."""

    def __init__(self, node_name: str = 'navigation_node'):
        super().__init__(node_name)
        self._declare_parameters()
        self.calibration = read_turn_calibration(self)
        self.forward_calibration = read_forward_calibration(self)
        self.navigator = None  # Subclass must instantiate
        self._create_publishers()
        self._create_subscriptions()
        self._create_services()
        self._configure_logging()
        self._create_control_timer()

    def _declare_parameters(self):
        for field in fields(NavigationPolicy):
            self.declare_parameter('navigation.' + field.name, field.default)
        self.declare_parameter('navigation.square_size_ft', 6.0)
        self.declare_parameter('navigation.side_count', 4)
        self.declare_parameter('navigation.side_length_m', 0.0)
        self.declare_parameter('navigation.waypoint_tolerance_m', 0.01)
        self.declare_parameter('navigation.heading_tolerance_rad', 0.12)
        self.declare_parameter('navigation.pose_timeout_s', 0.60)
        self.declare_parameter('navigation.max_position_uncertainty_m', 0.40)
        self.declare_parameter('navigation.max_heading_uncertainty_rad', 0.40)
        self.declare_parameter('navigation.control_frequency_hz', 20.0)
        self.declare_parameter('navigation.auto_start', False)
        self.declare_parameter('navigation.output_dir', 'hw3_results')
        self.declare_parameter('navigation.log_csv', True)
        self.declare_parameter('navigation.turn_pulse_rad', math.pi / 12)
        self.declare_parameter('navigation.turn_settle_s', 3.0)

    def _create_publishers(self):
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/navigation/status', 10)
        self.waypoint_pub = self.create_publisher(PointStamped, '/navigation/target_waypoint', 10)

    def _create_subscriptions(self):
        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, '/ekf_slam/pose', self.on_pose_received, 10)
        self.vision_sub = self.create_subscription(String, '/ekf_slam/vision_feedback', self.on_vision_feedback, 10)
        self.start_sub = self.create_subscription(Empty, '/start_navigation', self.on_start_topic, 10)

    def _create_services(self):
        self.start_srv = self.create_service(Trigger, '/start_navigation', self.on_start_service)
        self.stop_srv = self.create_service(Trigger, '/stop_navigation', self.on_stop_service)

    def _configure_logging(self):
        self.log_csv = self.get_parameter('navigation.log_csv').value
        self.output_dir = Path(self.get_parameter('navigation.output_dir').value)
        self.log_file_path = self.output_dir / 'navigation_log.csv'
        self.csv_initialized = False
        if self.log_csv:
            try:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                with open(self.log_file_path, 'w', newline='', encoding='utf-8') as f:
                    w = csv.writer(f)
                    w.writerow([
                        'timestamp_s', 'waypoint_index', 'target_x_m', 'target_y_m',
                        'robot_x_m', 'robot_y_m', 'robot_theta_rad', 'distance_error_m',
                        'heading_error_rad', 'cmd_v_mps', 'cmd_omega_radps', 'state', 'status_message'
                    ])
                self.csv_initialized = True
            except Exception as e:
                self.get_logger().warn(f'Failed to initialize navigation CSV: {e}')

    def _create_control_timer(self):
        freq = max(1.0, float(self.get_parameter('navigation.control_frequency_hz').value))
        self.control_timer = self.create_timer(1.0 / freq, self.control_loop_tick)

    def on_pose_received(self, msg: PoseWithCovarianceStamped):
        stamp_s = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        x_slam = msg.pose.pose.position.x
        y_slam = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        theta_slam = math.atan2(siny_cosp, cosy_cosp)
        cov = msg.pose.covariance
        sigma_pos = math.sqrt(max(0.0, cov[0]) + max(0.0, cov[7]))
        sigma_theta = math.sqrt(max(0.0, cov[35]))
        if self.navigator is not None:
            self.navigator.update_ekf_pose(x_slam, y_slam, theta_slam, sigma_pos, sigma_theta, stamp_s)

    def on_vision_feedback(self, message: String):
        try:
            data = json.loads(message.data)
            stamp = float(data['stamp_s'])
            tags = [[int(t), float(x), float(y)] for t, x, y in data['tags']]
            known_ids = [int(t) for t in data['known_ids']]
            clearance = data.get('forward_clearance_m')
            tag_yaws = [[int(t), float(y)] for t, y in data.get('tag_yaws', [])]
            ekf_heading = data.get('ekf_heading')
            if self.navigator is not None:
                self.navigator.update_vision(stamp, tags, known_ids, clearance, tag_yaws, None, ekf_heading)
        except Exception:
            pass

    def on_start_topic(self, _msg: Empty):
        if self.navigator is not None:
            self.navigator.start()

    def on_start_service(self, _req, res: Trigger.Response):
        if self.navigator is not None:
            success, message = self.navigator.start()
            res.success = success
            res.message = message
        else:
            res.success = False
            res.message = "Navigator not initialized."
        return res

    def on_stop_service(self, _req, res: Trigger.Response):
        if self.navigator is not None:
            success, message = self.navigator.stop("User requested stop via service")
            self._publish_stop()
            res.success = success
            res.message = message
        else:
            res.success = False
            res.message = "Navigator not initialized."
        return res

    def _publish_stop(self):
        self.cmd_vel_pub.publish(Twist())

    def control_loop_tick(self):
        if self.navigator is None:
            return
        now_s = self.get_clock().now().nanoseconds * 1e-9
        output = self.navigator.step(now_s)

        tw = Twist()
        tw.linear.x = float(output.linear_v_mps)
        tw.angular.z = float(output.angular_omega_radps)
        self.cmd_vel_pub.publish(tw)

        st = String()
        st.data = f'[{output.state.value}] {output.status_message}'
        self.status_pub.publish(st)

        wp = PointStamped()
        wp.header.stamp = self.get_clock().now().to_msg()
        wp.header.frame_id = 'map'
        wp.point.x, wp.point.y = self.navigator.target
        self.waypoint_pub.publish(wp)

        if self.csv_initialized:
            try:
                with open(self.log_file_path, 'a', newline='', encoding='utf-8') as f:
                    w = csv.writer(f)
                    w.writerow([
                        f"{now_s:.3f}", output.waypoint_index,
                        f"{output.target_x_m:.4f}", f"{output.target_y_m:.4f}",
                        f"{output.robot_x_m:.4f}", f"{output.robot_y_m:.4f}",
                        f"{output.robot_theta_rad:.4f}", f"{output.remaining_distance_m:.4f}",
                        f"{output.heading_error_rad:.4f}", f"{output.linear_v_mps:.4f}",
                        f"{output.angular_omega_radps:.4f}", output.state.value,
                        output.status_message
                    ])
            except Exception:
                pass
