"""Extended Kalman Filter SLAM (EKF-SLAM) Implementation (Step 3).

Maintains joint estimation of:
- Robot planar pose [x_robot, y_robot, theta_robot]^T
- Dynamically discovered landmarks [x_1, y_1, x_2, y_2, ..., x_N, y_N]^T
using AprilTag relative observations.
"""

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from .coordinates import wrap_angle


@dataclass(frozen=True)
class RobotPoseEstimate:
    """Robot state estimate with uncertainty."""
    x: float
    y: float
    theta: float
    covariance: np.ndarray  # 3x3 [x, y, theta]


@dataclass(frozen=True)
class LandmarkEstimate:
    """Landmark state estimate with uncertainty."""
    tag_id: int
    x: float
    y: float
    covariance: np.ndarray  # 2x2 [x, y]


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
        """Initialize EKF-SLAM filter.

        Args:
            initial_robot_pose: (x, y, theta) initial coordinate frame anchor.
            initial_robot_sigma: (sigma_x, sigma_y, sigma_theta) initial robot uncertainty.
            vo_process_noise: (sigma_x, sigma_y, sigma_theta) process noise per VO step.
            cmd_process_noise: One-second noise std [forward_m, lateral_m, yaw_rad];
                variance grows linearly with elapsed time while commanding motion.
            base_measurement_noise: (sigma_range_m, sigma_bearing_rad) at 1 meter.
            range_scale_dist_m: Reference distance for scaling range uncertainty.
            mahalanobis_gate: Chi-square threshold for 2-DOF outlier rejection (99% = 9.21).

        Returns:
            None. State changes and publications are performed in place.
        """
        self.x = np.array([
            float(initial_robot_pose[0]),
            float(initial_robot_pose[1]),
            wrap_angle(float(initial_robot_pose[2])),
        ], dtype=float)

        self.P = np.diag([
            float(initial_robot_sigma[0]) ** 2,
            float(initial_robot_sigma[1]) ** 2,
            float(initial_robot_sigma[2]) ** 2,
        ]).astype(float)

        self.vo_process_noise = np.array(vo_process_noise, dtype=float)
        self.cmd_process_noise = np.array(cmd_process_noise, dtype=float)
        self.base_meas_noise = np.array(base_measurement_noise, dtype=float)
        self.range_scale_dist_m = float(range_scale_dist_m)
        self.mahalanobis_gate = float(mahalanobis_gate)

        # Map from tag_id to state index in landmark block
        # Index k corresponds to state indices 3 + 2*k and 3 + 2*k + 1
        self.tag_id_to_idx: Dict[int, int] = {}
        self.idx_to_tag_id: List[int] = []

    @property
    def num_landmarks(self) -> int:
        return len(self.idx_to_tag_id)

    @property
    def state_dim(self) -> int:
        return 3 + 2 * self.num_landmarks

    def predict_visual_odometry(
        self,
        dx: float,
        dy: float,
        dtheta: float,
        vo_covariance: Optional[np.ndarray] = None,
    ):
        """EKF Prediction step using visual relative motion.

        Robot motion model in robot base frame:
            x_k+1 = x_k + dx*cos(theta_k) - dy*sin(theta_k)
            y_k+1 = y_k + dx*sin(theta_k) + dy*cos(theta_k)
            theta_k+1 = theta_k + dtheta

        Args:
            dx: Relative displacement along robot X-axis (meters).
            dy: Relative displacement along robot Y-axis (meters).
            dtheta: Relative rotation (radians).
            vo_covariance: Optional 3x3 covariance matrix from visual odometry.

        Returns:
            None. State changes and publications are performed in place.
        """
        x_r = self.x[0]
        y_r = self.x[1]
        theta_r = self.x[2]

        c = math.cos(theta_r)
        s = math.sin(theta_r)

        # State propagation
        self.x[0] = x_r + dx * c - dy * s
        self.x[1] = y_r + dx * s + dy * c
        self.x[2] = wrap_angle(theta_r + dtheta)

        # Motion Jacobians
        # F_R = d(f_robot) / d(x_robot)
        f_r = np.array([
            [1.0, 0.0, -dx * s - dy * c],
            [0.0, 1.0,  dx * c - dy * s],
            [0.0, 0.0,  1.0],
        ], dtype=float)

        # G_R = d(f_robot) / d(motion_input)
        g_r = np.array([
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=float)

        if vo_covariance is not None and vo_covariance.shape == (3, 3):
            q_input = np.asarray(vo_covariance, dtype=float)
        else:
            q_input = np.diag(self.vo_process_noise ** 2)

        # Covariance propagation
        p_rr = self.P[:3, :3]
        p_rr_new = f_r @ p_rr @ f_r.T + g_r @ q_input @ g_r.T

        if self.num_landmarks > 0:
            p_rm = self.P[:3, 3:]
            p_rm_new = f_r @ p_rm
            self.P[:3, :3] = p_rr_new
            self.P[:3, 3:] = p_rm_new
            self.P[3:, :3] = p_rm_new.T
        else:
            self.P[:3, :3] = p_rr_new

        self._ensure_covariance_symmetry()

    def predict_commanded_velocity(
        self,
        v: float,
        omega: float,
        dt: float,
    ):
        """EKF Prediction step using commanded linear and angular velocity (fallback).

        Planar differential drive motion model:
            x_k+1 = x_k + v*dt*cos(theta_k)
            y_k+1 = y_k + v*dt*sin(theta_k)
            theta_k+1 = theta_k + omega*dt

        Args:
            v: Linear velocity in m/s.
            omega: Angular velocity in rad/s.
            dt: Time step in seconds.

        Returns:
            None. State changes and publications are performed in place.
        """
        if dt <= 0.0:
            return

        dx = v * dt
        dy = 0.0
        dtheta = omega * dt

        # one-second velocity noise defines diffusion, independent of callback rate
        # stopped commands introduce no model motion uncertainty
        active = float(abs(v) > 0 or abs(omega) > 0)
        q_cmd = np.diag(self.cmd_process_noise ** 2) * dt * active

        x_r = self.x[0]
        y_r = self.x[1]
        theta_r = self.x[2]

        c = math.cos(theta_r)
        s = math.sin(theta_r)

        self.x[0] = x_r + dx * c
        self.x[1] = y_r + dx * s
        self.x[2] = wrap_angle(theta_r + dtheta)

        f_r = np.array([
            [1.0, 0.0, -dx * s],
            [0.0, 1.0,  dx * c],
            [0.0, 0.0,  1.0],
        ], dtype=float)

        g_r = np.array([
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=float)

        p_rr = self.P[:3, :3]
        p_rr_new = f_r @ p_rr @ f_r.T + g_r @ q_cmd @ g_r.T

        if self.num_landmarks > 0:
            p_rm = self.P[:3, 3:]
            p_rm_new = f_r @ p_rm
            self.P[:3, :3] = p_rr_new
            self.P[:3, 3:] = p_rm_new
            self.P[3:, :3] = p_rm_new.T
        else:
            self.P[:3, :3] = p_rr_new

        self._ensure_covariance_symmetry()

    def predict_untracked_drift(
        self,
        dt: float,
        pos_drift_rate_mps: float = 0.05,
        rot_drift_rate_radps: float = 0.35,
    ):
        """Propagate state covariance when robot motion is unobserved/untracked.

        In visual-only setups without wheel encoders or IMU, if VO loses tracking
        or landmarks are temporarily obscured while the robot moves or rotates,
        uncertainty must grow to reflect unobserved drift. This allows subsequent
        re-observations of known landmarks to be accepted rather than falsely
        rejected by the Mahalanobis outlier gate.

        Args:
            dt: Elapsed time without tracking (seconds).
            pos_drift_rate_mps: Position drift 1-sigma rate (m/s).
            rot_drift_rate_radps: Heading drift 1-sigma rate (rad/s).

        Returns:
            None. State changes and publications are performed in place.
        """
        if dt <= 0.0:
            return

        self.P[0, 0] += (pos_drift_rate_mps * dt) ** 2
        self.P[1, 1] += (pos_drift_rate_mps * dt) ** 2
        self.P[2, 2] += (rot_drift_rate_radps * dt) ** 2

        self._ensure_covariance_symmetry()

    def update_landmark(
        self,
        tag_id: int,
        x_rel: float,
        y_rel: float,
        measurement_cov: Optional[np.ndarray] = None,
    ) -> bool:
        """Process observation of an AprilTag (range & bearing in robot base frame).

        If the landmark is new, it is initialized into the map and state vector.
        If it already exists, standard EKF measurement update is performed.

        Args:
            tag_id: Unique AprilTag integer ID.
            x_rel: Relative X position in robot base frame (meters).
            y_rel: Relative Y position in robot base frame (meters).
            measurement_cov: Optional 2x2 measurement noise covariance in range & bearing.

        Returns:
            True if observation was successfully incorporated or initialized,
            False if rejected as an outlier.
        """
        r_meas = math.hypot(x_rel, y_rel)
        beta_meas = math.atan2(y_rel, x_rel)

        if r_meas < 1e-4:
            return False

        if measurement_cov is not None and measurement_cov.shape == (2, 2):
            r_mat = np.asarray(measurement_cov, dtype=float)
        else:
            # Range uncertainty scales quadratically with distance
            scale = 1.0 + (r_meas / self.range_scale_dist_m) ** 2
            r_mat = np.diag([
                (self.base_meas_noise[0] * scale) ** 2,
                (self.base_meas_noise[1] * scale) ** 2,
            ])

        if tag_id not in self.tag_id_to_idx:
            self._initialize_landmark(tag_id, r_meas, beta_meas, r_mat)
            return True

        # Existing landmark: Perform measurement update
        idx = self.tag_id_to_idx[tag_id]
        m_idx = 3 + 2 * idx

        x_r = self.x[0]
        y_r = self.x[1]
        theta_r = self.x[2]

        m_x = self.x[m_idx]
        m_y = self.x[m_idx + 1]

        dx = m_x - x_r
        dy = m_y - y_r
        q = dx * dx + dy * dy
        r_pred = math.sqrt(q)

        if r_pred < 1e-6:
            return False

        beta_pred = wrap_angle(math.atan2(dy, dx) - theta_r)

        # Innovation (residual)
        nu = np.array([
            r_meas - r_pred,
            wrap_angle(beta_meas - beta_pred),
        ], dtype=float)

        # Measurement Jacobian H (2 x state_dim)
        h = np.zeros((2, self.state_dim), dtype=float)

        # Robot state derivatives
        h[0, 0] = -dx / r_pred
        h[0, 1] = -dy / r_pred
        h[0, 2] = 0.0

        h[1, 0] = dy / q
        h[1, 1] = -dx / q
        h[1, 2] = -1.0

        # Landmark state derivatives
        h[0, m_idx] = dx / r_pred
        h[0, m_idx + 1] = dy / r_pred

        h[1, m_idx] = -dy / q
        h[1, m_idx + 1] = dx / q

        # Innovation covariance S = H * P * H^T + R
        s = h @ self.P @ h.T + r_mat

        # Mahalanobis distance outlier check
        try:
            s_inv = np.linalg.inv(s)
            d2 = float(nu.T @ s_inv @ nu)
            if d2 > self.mahalanobis_gate:
                # Reject outlier measurement
                return False
        except np.linalg.LinAlgError:
            return False

        # Kalman Gain K = P * H^T * S^-1
        k = self.P @ h.T @ s_inv

        # State update
        self.x += k @ nu
        self.x[2] = wrap_angle(self.x[2])

        # Joseph form covariance update for numerical stability:
        # P = (I - K*H) * P * (I - K*H)^T + K * R * K^T
        eye = np.eye(self.state_dim, dtype=float)
        i_kh = eye - k @ h
        self.P = i_kh @ self.P @ i_kh.T + k @ r_mat @ k.T

        self._ensure_covariance_symmetry()
        return True

    def _initialize_landmark(
        self,
        tag_id: int,
        r_meas: float,
        beta_meas: float,
        r_mat: np.ndarray,
    ):
        """Initialize a new landmark in state and covariance using Jacobian augmentation.

        Global landmark position:
            m_x = x_r + r * cos(theta_r + beta)
            m_y = y_r + r * sin(theta_r + beta)

        Args:
            tag_id: AprilTag integer ID used to associate the same stationary landmark.
            r_meas: Measured planar landmark range from the robot base, in meters.
            beta_meas: Measured landmark bearing in the robot base frame, in radians.
            r_mat: 2x2 range/bearing measurement covariance ordered [meters, radians].

        Returns:
            None. State changes and publications are performed in place.
        """
        x_r = self.x[0]
        y_r = self.x[1]
        theta_r = self.x[2]

        alpha = theta_r + beta_meas
        c_a = math.cos(alpha)
        s_a = math.sin(alpha)

        m_x = x_r + r_meas * c_a
        m_y = y_r + r_meas * s_a

        # Jacobian w.r.t robot pose x_r [x, y, theta]
        # J_xr = d(g) / d(x_r)
        j_xr = np.array([
            [1.0, 0.0, -r_meas * s_a],
            [0.0, 1.0,  r_meas * c_a],
        ], dtype=float)

        # Jacobian w.r.t measurement [r, beta]
        # J_z = d(g) / d(z)
        j_z = np.array([
            [c_a, -r_meas * s_a],
            [s_a,  r_meas * c_a],
        ], dtype=float)

        # Auto-covariance of the new landmark
        p_rr = self.P[:3, :3]
        p_mm = j_xr @ p_rr @ j_xr.T + j_z @ r_mat @ j_z.T

        # Cross-covariance between existing state and new landmark
        # Cov(x_existing, m_new) = P_existing * [J_xr^T; 0]
        p_existing_m = self.P[:, :3] @ j_xr.T  # (old_dim x 2)

        old_dim = self.state_dim
        new_dim = old_dim + 2

        new_p = np.zeros((new_dim, new_dim), dtype=float)
        new_p[:old_dim, :old_dim] = self.P
        new_p[:old_dim, old_dim:] = p_existing_m
        new_p[old_dim:, :old_dim] = p_existing_m.T
        new_p[old_dim:, old_dim:] = p_mm

        new_x = np.zeros(new_dim, dtype=float)
        new_x[:old_dim] = self.x
        new_x[old_dim] = m_x
        new_x[old_dim + 1] = m_y

        self.x = new_x
        self.P = new_p

        new_idx = len(self.idx_to_tag_id)
        self.tag_id_to_idx[tag_id] = new_idx
        self.idx_to_tag_id.append(tag_id)

        self._ensure_covariance_symmetry()

    def _ensure_covariance_symmetry(self):
        """Enforce covariance symmetry and positive semi-definiteness."""
        self.P = 0.5 * (self.P + self.P.T)
        # Small regularization on diagonal if needed for numerical safety
        diag = np.diagonal(self.P)
        if np.any(diag < 0.0):
            np.fill_diagonal(self.P, np.maximum(diag, 1e-9))

    def get_robot_pose(self) -> RobotPoseEstimate:
        """Return current estimated robot pose and 3x3 covariance."""
        return RobotPoseEstimate(
            x=float(self.x[0]),
            y=float(self.x[1]),
            theta=float(self.x[2]),
            covariance=self.P[:3, :3].copy(),
        )

    def get_landmark_pose(self, tag_id: int) -> Optional[LandmarkEstimate]:
        """Return estimated global pose and 2x2 covariance for a given tag ID.

        Args:
            tag_id: AprilTag integer ID used to associate the same stationary landmark.

        Returns:
            LandmarkEstimate with xy coordinates and 2x2 covariance, or None for an unknown ID.
        """
        if tag_id not in self.tag_id_to_idx:
            return None
        idx = self.tag_id_to_idx[tag_id]
        m_idx = 3 + 2 * idx
        return LandmarkEstimate(
            tag_id=tag_id,
            x=float(self.x[m_idx]),
            y=float(self.x[m_idx + 1]),
            covariance=self.P[m_idx:m_idx + 2, m_idx:m_idx + 2].copy(),
        )

    def get_relative_landmark_covariance(self, tag_id):
        """Propagate joint pose/map uncertainty to the tag position in robot coordinates.

        Args:
            tag_id: Mapped AprilTag ID.

        Returns:
            2x2 covariance in m² for [forward, left], or None for an unknown tag.
            Uses the full joint covariance so shared global-pose uncertainty cancels.
        """
        if tag_id not in self.tag_id_to_idx:
            return None
        index = 3 + 2 * self.tag_id_to_idx[tag_id]
        c, s = math.cos(self.x[2]), math.sin(self.x[2])
        rotation = np.array([[c, s], [-s, c]])
        relative = rotation @ (self.x[index:index + 2] - self.x[:2])
        jacobian = np.zeros((2, self.state_dim))
        jacobian[:, :2], jacobian[:, index:index + 2] = -rotation, rotation
        jacobian[:, 2] = [relative[1], -relative[0]]
        covariance = jacobian @ self.P @ jacobian.T
        return (covariance + covariance.T) * 0.5

    def get_all_landmarks(self) -> Dict[int, LandmarkEstimate]:
        """Return map of all currently discovered landmarks."""
        result = {}
        for tag_id in self.idx_to_tag_id:
            est = self.get_landmark_pose(tag_id)
            if est is not None:
                result[tag_id] = est
        return result
