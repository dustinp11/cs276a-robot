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
    SETTLING = "SETTLING"
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
        angular_velocity_ccw_radps: float = 1.2566,  # Calibrated: 360 deg in 5.0s (motor [-0.35, 0.35])
        angular_velocity_cw_radps: float = 1.3090,   # Calibrated: 360 deg in 4.8s (motor [0.35, -0.35])
        max_angular_omega_radps: float = 1.31,
        min_angular_omega_radps: float = 0.20,
        linear_kp: float = 0.60,
        angular_kp: float = 1.20,
        waypoint_tolerance_m: float = 0.15,
        heading_tolerance_rad: float = 0.10,  # ~5.7 degrees
        heading_drive_threshold_rad: float = 0.45,  # ~26 degrees
        turn_step_deg: float = 90.0,  # Option A: Single continuous 90 deg turn at corners
        turn_settle_time_s: float = 1.5,  # Pause duration (s) after corner turn for stable camera frames & tag detection
        pose_timeout_s: float = 0.60,
        max_pos_uncertainty_m: float = 0.40,
        max_heading_uncertainty_rad: float = 0.40,
    ):
        self.square_size_ft = float(square_size_ft)
        self.square_size_m = self.square_size_ft * self.FEET_TO_METERS

        self.max_linear_v = float(max_linear_v_mps)
        self.min_linear_v = float(min_linear_v_mps)
        self.angular_velocity_ccw = float(angular_velocity_ccw_radps)
        self.angular_velocity_cw = float(angular_velocity_cw_radps)
        self.max_angular_omega = float(max_angular_omega_radps)
        self.min_angular_omega = float(min_angular_omega_radps)

        self.linear_kp = float(linear_kp)
        self.angular_kp = float(angular_kp)

        self.waypoint_tolerance_m = float(waypoint_tolerance_m)
        self.heading_tolerance_rad = float(heading_tolerance_rad)
        self.heading_drive_threshold_rad = float(heading_drive_threshold_rad)

        self.turn_step_deg = float(turn_step_deg)
        self.turn_settle_time_s = float(turn_settle_time_s)
        self.num_subturns = max(1, int(round(90.0 / self.turn_step_deg)))

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

        # Nominal segment headings for legs 1..4
        # Leg 1 (WP0 -> WP1): 0 rad (+X)
        # Leg 2 (WP1 -> WP2): +pi/2 rad (+Y)
        # Leg 3 (WP2 -> WP3): pi rad (-X)
        # Leg 4 (WP3 -> WP4): -pi/2 rad (-Y)
        self.nominal_leg_headings = [0.0, math.pi / 2.0, math.pi, -math.pi / 2.0]

        self.current_wp_idx = 1  # First destination is WP1
        self.state = NavigationState.IDLE
        self.is_active = False

        # Corner sub-turn state tracking (30 deg x 3 at corners with settling)
        self.in_corner_turn = False
        self.corner_subturn_idx = 0
        self.corner_incoming_heading = 0.0
        self.corner_step_target_heading = 0.0
        self.settle_start_time = 0.0

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
        self.in_corner_turn = False
        self.corner_subturn_idx = 0
        self.corner_incoming_heading = 0.0
        self.corner_step_target_heading = 0.0
        self.settle_start_time = 0.0

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

        # 6. Compute distance to current waypoint
        dx = target_wp.x_m - rx
        dy = target_wp.y_m - ry
        dist = math.hypot(dx, dy)

        # 7. Check if current waypoint is reached (if not already executing a corner turn)
        if not self.in_corner_turn:
            reached = False
            if dist <= self.waypoint_tolerance_m:
                reached = True
            elif self.current_wp_idx == 1 and rx >= target_wp.x_m:
                reached = True
            elif self.current_wp_idx == 2 and ry >= target_wp.y_m:
                reached = True
            elif self.current_wp_idx == 3 and rx <= target_wp.x_m:
                reached = True
            elif self.current_wp_idx == 4 and ry <= target_wp.y_m:
                reached = True

            if reached:
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
                        heading_error_rad=0.0,
                        sigma_pos_m=sigma_pos,
                        sigma_theta_rad=sigma_theta,
                    )
                else:
                    # Corner 1, 2, or 3 reached: start step-wise corner turn (30 deg x 3)
                    self.in_corner_turn = True
                    self.corner_subturn_idx = 0
                    self.corner_incoming_heading = self.nominal_leg_headings[self.current_wp_idx - 1]
                    step_rad = math.radians(self.turn_step_deg)
                    self.corner_step_target_heading = wrap_angle(self.corner_incoming_heading + step_rad)
                    self.state = NavigationState.TURNING

        # 8. Controller execution: Corner turn sequence vs Leg navigation
        if self.in_corner_turn:
            target_heading = self.corner_step_target_heading
            alpha = wrap_angle(target_heading - rtheta)

            if self.state in (NavigationState.TURNING, NavigationState.PAUSED_STALE_POSE, NavigationState.PAUSED_HIGH_UNCERTAINTY):
                if abs(alpha) <= self.heading_tolerance_rad:
                    # Reached the turn angle! Begin camera settle period
                    self.state = NavigationState.SETTLING
                    self.settle_start_time = current_time_s
                    cmd_v = 0.0
                    cmd_omega = 0.0
                    if self.num_subturns == 1:
                        msg = (
                            f"Corner WP{self.current_wp_idx} 90° turn reached ({math.degrees(rtheta):.1f}°). "
                            f"Settling camera feed facing new wall for {self.turn_settle_time_s:.1f}s..."
                        )
                    else:
                        msg = (
                            f"Corner WP{self.current_wp_idx} sub-turn {self.corner_subturn_idx + 1}/{self.num_subturns} "
                            f"reached ({math.degrees(rtheta):.1f}°). Settling camera feed..."
                        )
                else:
                    cmd_v = 0.0
                    cmd_omega = self.angular_velocity_ccw if alpha > 0 else -self.angular_velocity_cw
                    if self.num_subturns == 1:
                        msg = (
                            f"Corner WP{self.current_wp_idx}: turning 90° to {math.degrees(target_heading):.1f}° "
                            f"(err: {math.degrees(alpha):.1f}°)"
                        )
                    else:
                        msg = (
                            f"Corner WP{self.current_wp_idx} sub-turn {self.corner_subturn_idx + 1}/{self.num_subturns}: "
                            f"turning to {math.degrees(target_heading):.1f}° (err: {math.degrees(alpha):.1f}°)"
                        )

            elif self.state == NavigationState.SETTLING:
                cmd_v = 0.0
                cmd_omega = 0.0
                elapsed = current_time_s - self.settle_start_time
                if elapsed >= self.turn_settle_time_s:
                    # Settle time complete! Advance to next sub-turn or finish corner
                    self.corner_subturn_idx += 1
                    if self.corner_subturn_idx < self.num_subturns:
                        step_rad = math.radians((self.corner_subturn_idx + 1) * self.turn_step_deg)
                        self.corner_step_target_heading = wrap_angle(self.corner_incoming_heading + step_rad)
                        self.state = NavigationState.TURNING
                        msg = (
                            f"Corner WP{self.current_wp_idx} starting sub-turn {self.corner_subturn_idx + 1}/{self.num_subturns} "
                            f"(target: {math.degrees(self.corner_step_target_heading):.1f}°)"
                        )
                    else:
                        # Completed corner turn (full 90 deg corner turn finished)
                        self.in_corner_turn = False
                        self.current_wp_idx += 1
                        target_wp = self.waypoints[self.current_wp_idx]
                        self.state = NavigationState.DRIVING
                        msg = f"Corner WP{self.current_wp_idx - 1} 90° turn completed. Driving to WP{self.current_wp_idx}."
                else:
                    if self.num_subturns == 1:
                        msg = (
                            f"Corner WP{self.current_wp_idx} settling camera on new wall "
                            f"({elapsed:.1f}s / {self.turn_settle_time_s:.1f}s)"
                        )
                    else:
                        msg = (
                            f"Corner WP{self.current_wp_idx} sub-turn {self.corner_subturn_idx + 1}/{self.num_subturns} "
                            f"settling ({elapsed:.1f}s / {self.turn_settle_time_s:.1f}s)"
                        )
            else:
                cmd_v = 0.0
                cmd_omega = 0.0
                msg = f"Controller in state {self.state.value}"

        else:
            # Driving towards target_wp or initial alignment
            theta_goal = math.atan2(dy, dx)
            alpha = wrap_angle(theta_goal - rtheta)

            if self.state in (NavigationState.TURNING, NavigationState.PAUSED_STALE_POSE, NavigationState.PAUSED_HIGH_UNCERTAINTY):
                if abs(alpha) <= self.heading_tolerance_rad:
                    self.state = NavigationState.DRIVING
                else:
                    self.state = NavigationState.TURNING

            # If driving but heading drifts significantly, turn in place to realign
            if self.state == NavigationState.DRIVING and abs(alpha) > self.heading_drive_threshold_rad:
                self.state = NavigationState.TURNING

            if self.state == NavigationState.TURNING:
                cmd_v = 0.0
                cmd_omega = self.angular_velocity_ccw if alpha > 0 else -self.angular_velocity_cw
                msg = f"Aligning heading towards WP{self.current_wp_idx} (heading err: {math.degrees(alpha):.1f}°)"

            elif self.state == NavigationState.DRIVING:
                raw_v = self.linear_kp * dist
                clamped_v = max(self.min_linear_v, min(self.max_linear_v, raw_v))
                heading_factor = max(0.0, math.cos(alpha))
                cmd_v = clamped_v * heading_factor
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
