"""Closed-loop square waypoint navigation controller using EKF-SLAM pose feedback.

Directs the robot counterclockwise through a 7 ft x 7 ft square (or configurable size)
using continuous pose feedback from EKF-SLAM:
    WP0: (0, 0) ft     [0.0000, 0.0000] m - Start / Origin
    WP1: (7, 0) ft     [2.1336, 0.0000] m - Bottom-right (+X)
    WP2: (7, 7) ft     [2.1336, 2.1336] m - Top-right (+X, +Y)
    WP3: (0, 7) ft     [0.0000, 2.1336] m - Top-left (+Y)
    WP4: (0, 0) ft     [0.0000, 0.0000] m - Return to start (Origin)

Coordinates are maintained in the local navigation frame anchored at the initial EKF
pose upon start:
    x_nav =  (x_slam - x_0) * cos(theta_0) + (y_slam - y_0) * sin(theta_0)
    y_nav = -(x_slam - x_0) * sin(theta_0) + (y_slam - y_0) * cos(theta_0)
    theta_nav = wrap_angle(theta_slam - theta_0)
"""

from dataclasses import dataclass
from enum import Enum
import math
from typing import Dict, List, Optional, Tuple

from .coordinates import wrap_angle


class NavigationState(Enum):
    """Navigation controller state machine states."""
    IDLE = "IDLE"
    WAITING_FOR_POSE = "WAITING_FOR_POSE"
    TURNING = "TURNING"
    DRIVING = "DRIVING"
    PAUSED_STALE_POSE = "PAUSED_STALE_POSE"
    PAUSED_HIGH_UNCERTAINTY = "PAUSED_HIGH_UNCERTAINTY"
    STOPPED = "STOPPED"
    COMPLETED = "COMPLETED"


@dataclass(frozen=True)
class Waypoint:
    """Target waypoint in meters."""
    index: int
    x_m: float
    y_m: float
    description: str


@dataclass
class NavigationOutput:
    """Command output and telemetry from one controller cycle."""
    linear_v_mps: float
    angular_omega_radps: float
    state: NavigationState
    status_message: str
    waypoint_index: int
    target_x_m: float
    target_y_m: float
    robot_x_m: float
    robot_y_m: float
    robot_theta_rad: float
    distance_error_m: float
    heading_error_rad: float
    sigma_pos_m: float
    sigma_theta_rad: float


class SquareNavigator:
    """Manages closed-loop waypoint following for the square trajectory."""

    FEET_TO_METERS = 0.3048

    def __init__(
        self,
        square_size_ft: float = 7.0,
        max_linear_v_mps: float = 0.20,
        min_linear_v_mps: float = 0.05,
        max_angular_omega_radps: float = 0.80,
        min_angular_omega_radps: float = 0.20,
        linear_kp: float = 0.60,
        angular_kp: float = 1.20,
        waypoint_tolerance_m: float = 0.10,
        heading_tolerance_rad: float = 0.12,  # ~7 degrees
        heading_drive_threshold_rad: float = 0.40,  # ~23 degrees
        pose_timeout_s: float = 0.60,
        max_pos_uncertainty_m: float = 0.40,
        max_heading_uncertainty_rad: float = 0.40,
    ):
        self.square_size_ft = float(square_size_ft)
        self.square_size_m = self.square_size_ft * self.FEET_TO_METERS

        self.max_linear_v = float(max_linear_v_mps)
        self.min_linear_v = float(min_linear_v_mps)
        self.max_angular_omega = float(max_angular_omega_radps)
        self.min_angular_omega = float(min_angular_omega_radps)

        self.linear_kp = float(linear_kp)
        self.angular_kp = float(angular_kp)

        self.waypoint_tolerance_m = float(waypoint_tolerance_m)
        self.heading_tolerance_rad = float(heading_tolerance_rad)
        self.heading_drive_threshold_rad = float(heading_drive_threshold_rad)

        self.pose_timeout_s = float(pose_timeout_s)
        self.max_pos_uncertainty_m = float(max_pos_uncertainty_m)
        self.max_heading_uncertainty_rad = float(max_heading_uncertainty_rad)

        # Build waypoints: counterclockwise square starting at (0, 0)
        s = self.square_size_m
        self.waypoints: List[Waypoint] = [
            Waypoint(0, 0.0, 0.0, "WP0: Bottom-left (Start)"),
            Waypoint(1, s, 0.0, "WP1: Bottom-right (+X)"),
            Waypoint(2, s, s, "WP2: Top-right (+X, +Y)"),
            Waypoint(3, 0.0, s, "WP3: Top-left (+Y)"),
            Waypoint(4, 0.0, 0.0, "WP4: Bottom-left (Return to origin)"),
        ]

        self.current_wp_idx = 1  # First destination is WP1
        self.state = NavigationState.IDLE
        self.is_active = False

        # Anchor origin (latched upon start)
        self.origin_latched = False
        self.x0_slam = 0.0
        self.y0_slam = 0.0
        self.theta0_slam = 0.0

        # Latest received EKF pose
        self.latest_slam_pose: Optional[Tuple[float, float, float]] = None
        self.latest_uncertainty: Tuple[float, float] = (0.0, 0.0)  # (sigma_pos, sigma_theta)
        self.latest_pose_stamp_s: float = 0.0

    def start(self) -> Tuple[bool, str]:
        """Start or resume the square navigation mission."""
        if self.state == NavigationState.COMPLETED:
            return False, "Mission already completed. Reset navigator to run again."

        self.is_active = True
        if not self.origin_latched:
            self.state = NavigationState.WAITING_FOR_POSE
            return True, "Navigation started. Waiting for initial EKF pose to anchor origin."
        else:
            self.state = NavigationState.TURNING
            return True, f"Navigation resumed towards Waypoint {self.current_wp_idx}."

    def stop(self, reason: str = "Manual stop requested") -> Tuple[bool, str]:
        """Stop the robot immediately and halt navigation."""
        self.is_active = False
        self.state = NavigationState.STOPPED
        return True, f"Navigation stopped: {reason}"

    def reset(self):
        """Reset navigator state to initial unstarted condition."""
        self.is_active = False
        self.state = NavigationState.IDLE
        self.origin_latched = False
        self.current_wp_idx = 1
        self.latest_slam_pose = None
        self.latest_pose_stamp_s = 0.0

    def update_ekf_pose(
        self,
        x_slam: float,
        y_slam: float,
        theta_slam: float,
        sigma_pos_m: float,
        sigma_theta_rad: float,
        stamp_s: float,
    ):
        """Update latest robot pose and uncertainty estimate from EKF-SLAM."""
        self.latest_slam_pose = (float(x_slam), float(y_slam), wrap_angle(float(theta_slam)))
        self.latest_uncertainty = (float(sigma_pos_m), float(sigma_theta_rad))
        self.latest_pose_stamp_s = float(stamp_s)

        # Latch initial pose as origin if waiting for it
        if self.is_active and not self.origin_latched:
            if (sigma_pos_m <= self.max_pos_uncertainty_m and
                    sigma_theta_rad <= self.max_heading_uncertainty_rad):
                self.x0_slam = float(x_slam)
                self.y0_slam = float(y_slam)
                self.theta0_slam = wrap_angle(float(theta_slam))
                self.origin_latched = True
                self.current_wp_idx = 1
                self.state = NavigationState.TURNING

    def slam_to_nav_frame(
        self,
        x_slam: float,
        y_slam: float,
        theta_slam: float,
    ) -> Tuple[float, float, float]:
        """Transform coordinates from global EKF SLAM frame to local navigation frame."""
        if not self.origin_latched:
            return 0.0, 0.0, 0.0

        dx = x_slam - self.x0_slam
        dy = y_slam - self.y0_slam
        c0 = math.cos(self.theta0_slam)
        s0 = math.sin(self.theta0_slam)

        x_nav = dx * c0 + dy * s0
        y_nav = -dx * s0 + dy * c0
        theta_nav = wrap_angle(theta_slam - self.theta0_slam)
        return x_nav, y_nav, theta_nav

    def step(self, current_time_s: float) -> NavigationOutput:
        """Compute closed-loop motor commands for the current control cycle."""
        # 1. Check if inactive or completed
        if not self.is_active:
            target_wp = self.waypoints[min(self.current_wp_idx, len(self.waypoints) - 1)]
            return NavigationOutput(
                linear_v_mps=0.0,
                angular_omega_radps=0.0,
                state=self.state,
                status_message="Navigation inactive" if self.state == NavigationState.IDLE else "Navigation stopped",
                waypoint_index=self.current_wp_idx,
                target_x_m=target_wp.x_m,
                target_y_m=target_wp.y_m,
                robot_x_m=0.0,
                robot_y_m=0.0,
                robot_theta_rad=0.0,
                distance_error_m=0.0,
                heading_error_rad=0.0,
                sigma_pos_m=self.latest_uncertainty[0],
                sigma_theta_rad=self.latest_uncertainty[1],
            )

        if self.state == NavigationState.COMPLETED:
            target_wp = self.waypoints[-1]
            return NavigationOutput(
                linear_v_mps=0.0,
                angular_omega_radps=0.0,
                state=NavigationState.COMPLETED,
                status_message="Square trajectory completed successfully",
                waypoint_index=len(self.waypoints) - 1,
                target_x_m=target_wp.x_m,
                target_y_m=target_wp.y_m,
                robot_x_m=0.0,
                robot_y_m=0.0,
                robot_theta_rad=0.0,
                distance_error_m=0.0,
                heading_error_rad=0.0,
                sigma_pos_m=self.latest_uncertainty[0],
                sigma_theta_rad=self.latest_uncertainty[1],
            )

        # 2. Check for initial pose availability
        if not self.origin_latched or self.latest_slam_pose is None:
            self.state = NavigationState.WAITING_FOR_POSE
            target_wp = self.waypoints[self.current_wp_idx]
            return NavigationOutput(
                linear_v_mps=0.0,
                angular_omega_radps=0.0,
                state=NavigationState.WAITING_FOR_POSE,
                status_message="Waiting for initial EKF-SLAM pose to latch local origin",
                waypoint_index=self.current_wp_idx,
                target_x_m=target_wp.x_m,
                target_y_m=target_wp.y_m,
                robot_x_m=0.0,
                robot_y_m=0.0,
                robot_theta_rad=0.0,
                distance_error_m=0.0,
                heading_error_rad=0.0,
                sigma_pos_m=self.latest_uncertainty[0],
                sigma_theta_rad=self.latest_uncertainty[1],
            )

        # 3. Check for stale pose
        pose_age = current_time_s - self.latest_pose_stamp_s
        if pose_age > self.pose_timeout_s:
            self.state = NavigationState.PAUSED_STALE_POSE
            target_wp = self.waypoints[self.current_wp_idx]
            return NavigationOutput(
                linear_v_mps=0.0,
                angular_omega_radps=0.0,
                state=NavigationState.PAUSED_STALE_POSE,
                status_message=f"Halted: pose estimate is stale (age {pose_age:.2f}s > {self.pose_timeout_s:.2f}s)",
                waypoint_index=self.current_wp_idx,
                target_x_m=target_wp.x_m,
                target_y_m=target_wp.y_m,
                robot_x_m=0.0,
                robot_y_m=0.0,
                robot_theta_rad=0.0,
                distance_error_m=0.0,
                heading_error_rad=0.0,
                sigma_pos_m=self.latest_uncertainty[0],
                sigma_theta_rad=self.latest_uncertainty[1],
            )

        # 4. Check for high localization uncertainty
        sigma_pos, sigma_theta = self.latest_uncertainty
        if (sigma_pos > self.max_pos_uncertainty_m or
                sigma_theta > self.max_heading_uncertainty_rad):
            self.state = NavigationState.PAUSED_HIGH_UNCERTAINTY
            target_wp = self.waypoints[self.current_wp_idx]
            return NavigationOutput(
                linear_v_mps=0.0,
                angular_omega_radps=0.0,
                state=NavigationState.PAUSED_HIGH_UNCERTAINTY,
                status_message=(
                    f"Halted: localization uncertainty high "
                    f"(pos: {sigma_pos:.2f}m > {self.max_pos_uncertainty_m:.2f}m, "
                    f"heading: {math.degrees(sigma_theta):.1f}° > {math.degrees(self.max_heading_uncertainty_rad):.1f}°)"
                ),
                waypoint_index=self.current_wp_idx,
                target_x_m=target_wp.x_m,
                target_y_m=target_wp.y_m,
                robot_x_m=0.0,
                robot_y_m=0.0,
                robot_theta_rad=0.0,
                distance_error_m=0.0,
                heading_error_rad=0.0,
                sigma_pos_m=sigma_pos,
                sigma_theta_rad=sigma_theta,
            )

        # 5. Transform current pose to navigation frame
        rx, ry, rtheta = self.slam_to_nav_frame(*self.latest_slam_pose)
        target_wp = self.waypoints[self.current_wp_idx]

        # 6. Compute distance and heading errors to current waypoint
        dx = target_wp.x_m - rx
        dy = target_wp.y_m - ry
        dist = math.hypot(dx, dy)
        theta_goal = math.atan2(dy, dx)
        alpha = wrap_angle(theta_goal - rtheta)

        # 7. Check if current waypoint is reached
        if dist <= self.waypoint_tolerance_m:
            if self.current_wp_idx >= len(self.waypoints) - 1:
                # Reached final waypoint (WP4: return to start)
                self.state = NavigationState.COMPLETED
                self.is_active = False
                return NavigationOutput(
                    linear_v_mps=0.0,
                    angular_omega_radps=0.0,
                    state=NavigationState.COMPLETED,
                    status_message=f"Waypoint {self.current_wp_idx} reached! Square completed.",
                    waypoint_index=self.current_wp_idx,
                    target_x_m=target_wp.x_m,
                    target_y_m=target_wp.y_m,
                    robot_x_m=rx,
                    robot_y_m=ry,
                    robot_theta_rad=rtheta,
                    distance_error_m=dist,
                    heading_error_rad=alpha,
                    sigma_pos_m=sigma_pos,
                    sigma_theta_rad=sigma_theta,
                )
            else:
                # Advance to next corner waypoint and pivot
                self.current_wp_idx += 1
                target_wp = self.waypoints[self.current_wp_idx]
                dx = target_wp.x_m - rx
                dy = target_wp.y_m - ry
                dist = math.hypot(dx, dy)
                theta_goal = math.atan2(dy, dx)
                alpha = wrap_angle(theta_goal - rtheta)
                self.state = NavigationState.TURNING

        # 8. Controller state machine: TURNING vs DRIVING
        if self.state in (NavigationState.TURNING, NavigationState.PAUSED_STALE_POSE, NavigationState.PAUSED_HIGH_UNCERTAINTY):
            # Check if aligned with target heading
            if abs(alpha) <= self.heading_tolerance_rad:
                self.state = NavigationState.DRIVING
            else:
                self.state = NavigationState.TURNING

        # If driving but heading error becomes excessive, revert to turn-in-place
        if self.state == NavigationState.DRIVING and abs(alpha) > self.heading_drive_threshold_rad:
            self.state = NavigationState.TURNING

        # 9. Compute motor commands based on active state
        if self.state == NavigationState.TURNING:
            cmd_v = 0.0
            # Proportional rotation with deadzone clamping and direction
            raw_omega = self.angular_kp * abs(alpha)
            clamped_omega = max(self.min_angular_omega, min(self.max_angular_omega, raw_omega))
            cmd_omega = math.copysign(clamped_omega, alpha)
            msg = f"Turning in place towards WP{self.current_wp_idx} (heading err: {math.degrees(alpha):.1f}°)"

        elif self.state == NavigationState.DRIVING:
            # Proportional linear speed scaled by cosine of heading error
            raw_v = self.linear_kp * dist
            clamped_v = max(self.min_linear_v, min(self.max_linear_v, raw_v))
            heading_factor = max(0.0, math.cos(alpha))
            cmd_v = clamped_v * heading_factor

            # Proportional angular steering correction
            cmd_omega = max(-self.max_angular_omega, min(self.max_angular_omega, self.angular_kp * alpha))
            msg = (
                f"Driving to WP{self.current_wp_idx} "
                f"(dist: {dist:.2f}m, heading err: {math.degrees(alpha):.1f}°, v: {cmd_v:.2f}m/s)"
            )
        else:
            cmd_v = 0.0
            cmd_omega = 0.0
            msg = f"Controller in state {self.state.value}"

        return NavigationOutput(
            linear_v_mps=cmd_v,
            angular_omega_radps=cmd_omega,
            state=self.state,
            status_message=msg,
            waypoint_index=self.current_wp_idx,
            target_x_m=target_wp.x_m,
            target_y_m=target_wp.y_m,
            robot_x_m=rx,
            robot_y_m=ry,
            robot_theta_rad=rtheta,
            distance_error_m=dist,
            heading_error_rad=alpha,
            sigma_pos_m=sigma_pos,
            sigma_theta_rad=sigma_theta,
        )
