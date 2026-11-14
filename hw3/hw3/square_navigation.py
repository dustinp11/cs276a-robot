"""Square navigation controller and ROS 2 node.

Implements the dedicated 2-lap square traversal:
- Starts directly driving +X without pre-turning
- Lap 1: Open-loop dead-reckoning (4 sides of square_size_ft with 1-second midpoint and post-forward camera pauses; 90° CCW turns)
- Transition: Return to estimated (0, 0) using AprilTag EKF-SLAM pose feedback and reorient to 0.0 rad
- Lap 2: Closed-loop square waypoint traversal using EKF-SLAM and visual odometry prediction with corner landmark updates
- Universal navigation_node entry point dispatching between Square and Octagon based on side_count
"""

from collections import deque
from dataclasses import fields
import math

import rclpy
from rclpy.node import Node

from .motor_control import ForwardCalibration, TurnCalibration
from .slam_vision import (
    BaseNavigationNode,
    NavigationOutput,
    NavigationPolicy,
    NavigationState,
    VisualFeedback,
    Waypoint,
    wrap_angle,
)


class SquareNavigator:
    """Pure controller for two-lap square navigation (open-loop lap 1, relocalize to origin, closed-loop lap 2)."""

    def __init__(
        self,
        policy: NavigationPolicy = None,
        side_count: int = 4,
        side_length_m: float = 0.0,
        square_size_ft: float = 6.0,
        turn_calibration: TurnCalibration = None,
        forward_calibration: ForwardCalibration = None,
        waypoint_tolerance_m: float = 0.05,
        heading_tolerance_rad: float = 0.12,
        pose_timeout_s: float = 0.60,
        max_pos_uncertainty_m: float = 0.40,
        max_heading_uncertainty_rad: float = 0.40,
        **_ignored,
    ):
        """Initialize two-lap square controller parameters, waypoints, and state."""
        self.policy = policy or NavigationPolicy(manual_motion_enabled=True, two_lap_square=True)
        self.side_count = 4
        self.square_size_m = side_length_m if side_length_m > 0 else square_size_ft * 0.3048
        self.turn_calibration = turn_calibration
        self.forward = forward_calibration or ForwardCalibration()
        self.waypoint_tolerance_m = waypoint_tolerance_m
        self.heading_tolerance_rad = heading_tolerance_rad
        self.pose_timeout_s = pose_timeout_s
        self.max_pos_uncertainty_m = max_pos_uncertainty_m
        self.max_heading_uncertainty_rad = max_heading_uncertainty_rad

        self.corner_angle_rad = math.pi / 2.0  # 90 degrees
        self.state = NavigationState.IDLE
        self.is_active = False
        self.stop_reason = 'Not started'

        self.lap = 1
        self.two_lap_square = getattr(self.policy, 'two_lap_square', True)
        self.closed_loop_tolerance_m = max(self.waypoint_tolerance_m, getattr(self.policy, 'closed_loop_waypoint_tolerance_m', 0.12))
        self.heading_drive_threshold_rad = getattr(self.policy, 'heading_drive_threshold_rad', 0.35)

        self.side_index = 0
        self.current_wp_idx = 1
        self.midpoint_done = False
        self.manual_final_turn_done = False
        self.drive_used = 0.0
        self.corner_used = 0.0

        self.origin_latched = False
        self.x0_slam = self.y0_slam = self.theta0_slam = 0.0
        self.latest_slam_pose = None
        self.latest_pose_stamp_s = -math.inf
        self.latest_uncertainty = (math.inf, math.inf)

        self.navigation_x = self.navigation_y = self.navigation_yaw = 0.0
        self.heading_time = -math.inf
        self.course_heading = 0.0
        self.target = (self.square_size_m, 0.0)
        self.last_command = (0.0, 0.0)
        self.last_tick = None
        self.phase_started = None

        self.waypoints = self._build_waypoints()
        self.vision = VisualFeedback(self.policy)

    def _build_waypoints(self) -> list[Waypoint]:
        """Construct local square corner waypoints in meters from side length."""
        s = self.square_size_m
        return [
            Waypoint(0, 0.0, 0.0, "WP0: Origin"),
            Waypoint(1, s, 0.0, "WP1: Corner 1 (+X)"),
            Waypoint(2, s, s, "WP2: Corner 2 (+X, +Y)"),
            Waypoint(3, 0.0, s, "WP3: Corner 3 (+Y)"),
            Waypoint(4, 0.0, 0.0, "WP4: Corner 4 (Origin)"),
        ]

    def _nav_to_slam_point(self, x_nav: float, y_nav: float) -> tuple[float, float]:
        """Transform navigation frame (x, y) point into SLAM coordinate frame."""
        c, s = math.cos(self.theta0_slam), math.sin(self.theta0_slam)
        return (self.x0_slam + c * x_nav - s * y_nav, self.y0_slam + s * x_nav + c * y_nav)

    def slam_to_nav_frame(self, x_slam: float, y_slam: float, theta_slam: float) -> tuple[float, float, float]:
        """Transform SLAM pose into navigation frame relative to latched origin."""
        dx, dy = x_slam - self.x0_slam, y_slam - self.y0_slam
        c, s = math.cos(-self.theta0_slam), math.sin(-self.theta0_slam)
        return (c * dx - s * dy, s * dx + c * dy, wrap_angle(theta_slam - self.theta0_slam))

    def start(self) -> tuple[bool, str]:
        """Activate square navigation mission and transition to initial wait state."""
        if self.turn_calibration is None:
            return False, 'Turn calibration is required.'
        if self.state != NavigationState.IDLE:
            return False, 'Mission already started; reset/relaunch for a new run.'
        self.is_active = True
        self.state = NavigationState.WAITING_FOR_POSE
        return True, 'Starting two-lap square mission'

    def stop(self, reason: str = 'Manual stop requested') -> tuple[bool, str]:
        """Halt navigation mission with a given reason."""
        self.stop_reason = reason
        self.is_active = False
        self.state = NavigationState.STOPPED
        return True, reason

    def update_ekf_pose(self, x_slam: float, y_slam: float, theta_slam: float, sigma_pos_m: float, sigma_theta_rad: float, stamp_s: float):
        """Update controller with latest EKF-SLAM pose and covariance estimates."""
        self.latest_slam_pose = (x_slam, y_slam, wrap_angle(theta_slam))
        self.latest_uncertainty = (sigma_pos_m, sigma_theta_rad)
        self.latest_pose_stamp_s = stamp_s
        if self.is_active and not self.origin_latched:
            self.x0_slam, self.y0_slam, self.theta0_slam = self.latest_slam_pose
            self.origin_latched = True
            self.navigation_yaw = self.theta0_slam
            self.heading_time = stamp_s
            self.navigation_x, self.navigation_y = x_slam, y_slam

    def update_vision(self, stamp: float, tags: list, known_ids: list, clearance, tag_yaws, sigmas, ekf_heading):
        """Update visual feedback state with latest AprilTag detection results."""
        self.vision.update(stamp, tags, known_ids, clearance, self.latest_slam_pose, self.drive_used, tag_yaws, sigmas, ekf_heading)

    def _best_pose(self) -> tuple[float, float, float]:
        """Return latest valid SLAM pose or dead-reckoned odometry if timed out."""
        now = self.last_tick
        if self.latest_slam_pose is not None and now is not None:
            if -0.05 <= now - self.latest_pose_stamp_s <= self.pose_timeout_s:
                return self.latest_slam_pose
        return self.navigation_x, self.navigation_y, self.navigation_yaw

    def _transition(self, state: NavigationState, now: float):
        """Switch navigation state and record the phase transition timestamp."""
        self.state = state
        self.phase_started = now

    def _heading_error(self) -> float:
        """Compute wrapped heading error toward current target or course heading."""
        if self.state in (NavigationState.RELOCALIZING_TURN, NavigationState.RETURN_DRIVING):
            rx, ry, rtheta = self.slam_to_nav_frame(*self._best_pose())
            target_heading = math.atan2(0.0 - ry, 0.0 - rx)
            return wrap_angle(target_heading - rtheta)
        if self.state in (NavigationState.REORIENTING, NavigationState.REORIENT_PAUSE):
            _, _, rtheta = self.slam_to_nav_frame(*self._best_pose())
            return wrap_angle(0.0 - rtheta)
        if self.two_lap_square and self.lap == 2:
            if self.current_wp_idx < len(self.waypoints):
                wp = self.waypoints[self.current_wp_idx]
                rx, ry, rtheta = self.slam_to_nav_frame(*self._best_pose())
                target_heading = math.atan2(wp.y_m - ry, wp.x_m - rx)
                return wrap_angle(target_heading - rtheta)
        return wrap_angle(self.course_heading - self.navigation_yaw)

    def _remaining_distance(self) -> float:
        """Compute remaining Euclidean distance to target waypoint or leg end."""
        if self.state in (NavigationState.RELOCALIZING_TURN, NavigationState.RETURN_DRIVING, NavigationState.RETURN_PAUSE):
            rx, ry, _ = self.slam_to_nav_frame(*self._best_pose())
            return math.hypot(rx, ry)
        if self.two_lap_square and self.lap == 2:
            if self.current_wp_idx < len(self.waypoints):
                wp = self.waypoints[self.current_wp_idx]
                rx, ry, _ = self.slam_to_nav_frame(*self._best_pose())
                return math.hypot(wp.x_m - rx, wp.y_m - ry)
        return max(0.0, self.square_size_m - self.drive_used)

    def _output(self, message: str, linear: float = 0.0, angular: float = 0.0) -> NavigationOutput:
        """Construct a navigation telemetry output container for the current tick."""
        self.last_command = (linear, angular)
        rx, ry, rtheta = self.slam_to_nav_frame(*self._best_pose())
        return NavigationOutput(
            linear_v_mps=linear,
            angular_omega_radps=angular,
            state=self.state,
            status_message=message,
            waypoint_index=self.current_wp_idx,
            target_x_m=self.target[0],
            target_y_m=self.target[1],
            robot_x_m=rx,
            robot_y_m=ry,
            remaining_distance_m=self._remaining_distance(),
            heading_error_rad=self._heading_error(),
            uncertainty_pos_m=self.latest_uncertainty[0],
            uncertainty_heading_rad=self.latest_uncertainty[1],
        )

    def step(self, current_time_s: float) -> NavigationOutput:
        """Advance navigation state machine by one discrete time step."""
        now = float(current_time_s)
        dt = 0.0 if self.last_tick is None else max(0.0, min(now - self.last_tick, 0.15))
        self.navigation_x += self.last_command[0] * dt * math.cos(self.navigation_yaw)
        self.navigation_y += self.last_command[0] * dt * math.sin(self.navigation_yaw)
        self.navigation_yaw = wrap_angle(self.navigation_yaw + self.last_command[1] * dt)
        self.drive_used += abs(self.last_command[0]) * dt
        if self.state == NavigationState.TURNING:
            self.corner_used += abs(self.last_command[1]) * dt
        self.last_tick = now

        if not self.is_active:
            return self._output(self.stop_reason)
        if self.phase_started is None:
            self.phase_started = now

        if self.two_lap_square:
            if self.lap == 1:
                if self.state in (NavigationState.RELOCALIZING_TURN, NavigationState.RETURN_DRIVING,
                                  NavigationState.RETURN_PAUSE, NavigationState.REORIENTING,
                                  NavigationState.REORIENT_PAUSE):
                    return self._return_step(now)
                return self._manual_step(now)
            elif self.lap == 2:
                return self._closed_loop_step(now)
        return self._manual_step(now)

    # -------------------------------------------------------------------------
    # Lap 1: Open-Loop Dead Reckoning
    # -------------------------------------------------------------------------

    def _manual_step(self, now: float) -> NavigationOutput:
        """Dispatch step handler for open-loop dead-reckoning state in Lap 1."""
        handlers = {
            NavigationState.WAITING_FOR_POSE: self._wait_for_start,
            NavigationState.TURNING: self._manual_turn,
            NavigationState.DRIVING: self._manual_drive,
            NavigationState.MIDPOINT_PAUSE: self._manual_midpoint_pause,
            NavigationState.STOPPING: self._manual_pause,
        }
        return handlers.get(self.state, lambda t: self._output(f'Unhandled: {self.state}'))(now)

    def _wait_for_start(self, now: float) -> NavigationOutput:
        """Latch origin pose and initiate first forward driving leg."""
        if not self.origin_latched:
            self.x0_slam, self.y0_slam, self.theta0_slam = self._best_pose()
            self.navigation_yaw = self.theta0_slam
            self.origin_latched = True
        self.side_index = 0
        self.current_wp_idx = 1
        self.course_heading = self.theta0_slam
        self.target = self._nav_to_slam_point(self.square_size_m, 0.0)
        self._begin_leg(now)
        return self._manual_drive(now)

    def _begin_leg(self, now: float):
        """Reset leg progress distance and enter forward driving state."""
        self.drive_used = 0.0
        self.midpoint_done = False
        self._transition(NavigationState.DRIVING, now)

    def _manual_drive(self, now: float) -> NavigationOutput:
        """Execute constant forward velocity until reaching midpoint or corner."""
        half_distance = self.square_size_m / 2.0
        goal_distance = self.square_size_m if self.midpoint_done else half_distance
        if self.drive_used + 1e-9 >= goal_distance:
            return self._manual_checkpoint(now)
        return self._output(f'[Lap 1] Driving side {self.side_index + 1}', linear=self.forward.speed_mps)

    def _manual_checkpoint(self, now: float) -> NavigationOutput:
        """Transition to midpoint pause or post-leg corner pause."""
        if not self.midpoint_done:
            self._transition(NavigationState.MIDPOINT_PAUSE, now)
            return self._output(f'[Lap 1] Side {self.side_index + 1} midpoint pause')
        self.side_index += 1
        self._transition(NavigationState.STOPPING, now)
        return self._output(f'[Lap 1] Side {self.side_index} finished; pause before turn')

    def _manual_midpoint_pause(self, now: float) -> NavigationOutput:
        """Pause robot at leg midpoint before resuming forward drive."""
        if now - self.phase_started + 1e-9 < self.policy.manual_midpoint_pause_s:
            return self._output(f'[Lap 1] Side {self.side_index + 1} midpoint pause')
        self.midpoint_done = True
        self._transition(NavigationState.DRIVING, now)
        return self._manual_drive(now)

    def _manual_pause(self, now: float) -> NavigationOutput:
        """Pause robot after driving a leg before initiating next corner turn."""
        if now - self.phase_started + 1e-9 < self.policy.manual_forward_pause_s:
            return self._output(f'[Lap 1] Side {self.side_index} post-forward pause')
        if self.side_index >= self.side_count:
            if not self.manual_final_turn_done:
                self.manual_final_turn_done = True
                self.course_heading = self.theta0_slam + 4 * self.corner_angle_rad
                self.corner_used = 0.0
                self.target = (self.x0_slam, self.y0_slam)
                self._transition(NavigationState.TURNING, now)
                return self._output('[Lap 1] Final turn 4 reorienting to 0.0 rad', angular=self.turn_calibration.rate(1))
            if self.two_lap_square and self.lap == 1:
                return self._begin_return_to_origin(now)
            self.state, self.is_active = NavigationState.COMPLETED, False
            return self._output('Square complete (1 lap)')
        return self._manual_begin_turn(now)

    def _manual_begin_turn(self, now: float) -> NavigationOutput:
        """Initialize corner turn angle and transition to turning state."""
        self.current_wp_idx = self.side_index + 1
        self.course_heading = self.theta0_slam + (self.side_index) * self.corner_angle_rad
        self.corner_used = 0.0
        if self.current_wp_idx < len(self.waypoints):
            wp = self.waypoints[self.current_wp_idx]
            self.target = self._nav_to_slam_point(wp.x_m, wp.y_m)
        self._transition(NavigationState.TURNING, now)
        return self._output(f'[Lap 1] Turning corner {self.side_index}', angular=self.turn_calibration.rate(1))

    def _manual_turn(self, now: float) -> NavigationOutput:
        """Command open-loop calibrated corner turn until timed duration elapses."""
        duration = getattr(self.policy, 'corner_turn_duration_s', 0.65)
        if now - self.phase_started + 1e-9 >= duration:
            if self.side_index >= self.side_count:
                self._transition(NavigationState.STOPPING, now)
                return self._output('[Lap 1] Completed 4th corner turn; pause before return phase')
            self._begin_leg(now)
            return self._manual_drive(now)
        return self._output(f'[Lap 1] Turning corner {self.side_index}', angular=self.turn_calibration.rate(1))

    # -------------------------------------------------------------------------
    # Transition: Return to Origin via AprilTag SLAM Pose
    # -------------------------------------------------------------------------

    def _begin_return_to_origin(self, now: float) -> NavigationOutput:
        """Initiate return-to-origin transit using AprilTag SLAM pose."""
        self.current_wp_idx = 0
        self.target = (self.x0_slam, self.y0_slam)
        rx, ry, _ = self.slam_to_nav_frame(*self._best_pose())
        dist = math.hypot(rx, ry)
        if dist <= self.closed_loop_tolerance_m:
            self._transition(NavigationState.RETURN_PAUSE, now)
            return self._output('[Return to Origin] Start already within tolerance; pausing before reorienting')
        self._transition(NavigationState.RELOCALIZING_TURN, now)
        return self._return_align(now)

    def _return_step(self, now: float) -> NavigationOutput:
        """Dispatch step handler for return-to-origin relocation phases."""
        handlers = {
            NavigationState.RELOCALIZING_TURN: self._return_align,
            NavigationState.RETURN_DRIVING: self._return_drive,
            NavigationState.RETURN_PAUSE: self._return_pause,
            NavigationState.REORIENTING: self._return_reorient,
            NavigationState.REORIENT_PAUSE: self._return_reorient_pause,
        }
        return handlers.get(self.state, lambda t: self._output(f'Unhandled return state: {self.state}'))(now)

    def _return_align(self, now: float) -> NavigationOutput:
        """Pivot towards estimated (0, 0) origin until heading error is within tolerance."""
        rx, ry, rtheta = self.slam_to_nav_frame(*self._best_pose())
        dist = math.hypot(rx, ry)
        if dist <= self.closed_loop_tolerance_m:
            self._transition(NavigationState.RETURN_PAUSE, now)
            return self._output(f'[Return to Origin] Origin reached (dist: {dist:.3f}m); pausing')
        target_heading = math.atan2(0.0 - ry, 0.0 - rx)
        alpha = wrap_angle(target_heading - rtheta)
        if abs(alpha) <= self.heading_tolerance_rad:
            self._transition(NavigationState.RETURN_DRIVING, now)
            return self._output(f'[Return to Origin] Aligned (err: {math.degrees(alpha):.1f}°); driving to origin', linear=self.forward.speed_mps)
        return self._output(f'[Return to Origin] Aligning to (0,0) (dist: {dist:.2f}m, err: {math.degrees(alpha):.1f}°)', angular=self.turn_calibration.rate(alpha))

    def _return_drive(self, now: float) -> NavigationOutput:
        """Drive towards estimated (0, 0) origin, realigning heading if drifted."""
        rx, ry, rtheta = self.slam_to_nav_frame(*self._best_pose())
        dist = math.hypot(rx, ry)
        if dist <= self.closed_loop_tolerance_m:
            self._transition(NavigationState.RETURN_PAUSE, now)
            return self._output(f'[Return to Origin] Arrived at origin (dist: {dist:.3f}m); settling')
        target_heading = math.atan2(0.0 - ry, 0.0 - rx)
        alpha = wrap_angle(target_heading - rtheta)
        if abs(alpha) > self.heading_drive_threshold_rad:
            self._transition(NavigationState.RELOCALIZING_TURN, now)
            return self._output(f'[Return to Origin] Heading drifted ({math.degrees(alpha):.1f}°); realigning', angular=self.turn_calibration.rate(alpha))
        return self._output(f'[Return to Origin] Driving to origin (dist: {dist:.2f}m, err: {math.degrees(alpha):.1f}°)', linear=self.forward.speed_mps)

    def _return_pause(self, now: float) -> NavigationOutput:
        """Settle at origin to allow EKF-SLAM to update landmark estimates."""
        if now - self.phase_started + 1e-9 < self.policy.manual_forward_pause_s:
            return self._output('[Return to Origin] Settling at origin to update landmark estimates')
        self._transition(NavigationState.REORIENTING, now)
        return self._return_reorient(now)

    def _return_reorient(self, now: float) -> NavigationOutput:
        """Pivot at origin to reorient heading to 0 radians before Lap 2."""
        _, _, rtheta = self.slam_to_nav_frame(*self._best_pose())
        alpha = wrap_angle(0.0 - rtheta)
        if abs(alpha) <= self.heading_tolerance_rad:
            self._transition(NavigationState.REORIENT_PAUSE, now)
            return self._output(f'[Return to Origin] Reoriented to heading 0.0 rad (err: {math.degrees(alpha):.1f}°); settle pause')
        return self._output(f'[Return to Origin] Reorienting to heading 0.0 rad (err: {math.degrees(alpha):.1f}°)', angular=self.turn_calibration.rate(alpha))

    def _return_reorient_pause(self, now: float) -> NavigationOutput:
        """Pause briefly after reorienting before launching closed-loop Lap 2."""
        if now - self.phase_started + 1e-9 < self.policy.manual_forward_pause_s:
            return self._output('[Return to Origin] Settle pause at origin before Lap 2')
        self.lap = 2
        self.current_wp_idx = 1
        self.side_index = 0
        wp = self.waypoints[self.current_wp_idx]
        self.target = self._nav_to_slam_point(wp.x_m, wp.y_m)
        self._transition(NavigationState.CLOSED_LOOP_TURN, now)
        return self._closed_loop_turn(now)

    # -------------------------------------------------------------------------
    # Lap 2: Closed-Loop Waypoint Traversal with Visual Odometry
    # -------------------------------------------------------------------------

    def _closed_loop_step(self, now: float) -> NavigationOutput:
        """Dispatch step handler for closed-loop visual odometry Lap 2."""
        handlers = {
            NavigationState.CLOSED_LOOP_TURN: self._closed_loop_turn,
            NavigationState.CLOSED_LOOP_DRIVE: self._closed_loop_drive,
            NavigationState.CLOSED_LOOP_CORNER_PAUSE: self._closed_loop_corner_pause,
            NavigationState.CLOSED_LOOP_FINAL_REORIENT: self._closed_loop_final_reorient,
            NavigationState.COMPLETED: lambda t: self._output('Square complete (2 laps)'),
        }
        return handlers.get(self.state, lambda t: self._output(f'Unhandled Lap 2 state: {self.state}'))(now)

    def _closed_loop_turn(self, now: float) -> NavigationOutput:
        """Turn towards current waypoint until aligned within heading tolerance."""
        wp = self.waypoints[self.current_wp_idx]
        self.target = self._nav_to_slam_point(wp.x_m, wp.y_m)
        rx, ry, rtheta = self.slam_to_nav_frame(*self._best_pose())
        dx, dy = wp.x_m - rx, wp.y_m - ry
        dist = math.hypot(dx, dy)
        if dist <= self.closed_loop_tolerance_m:
            self._transition(NavigationState.CLOSED_LOOP_CORNER_PAUSE, now)
            return self._output(f'[Lap 2] WP{self.current_wp_idx} already reached; corner pause')
        target_heading = math.atan2(dy, dx)
        alpha = wrap_angle(target_heading - rtheta)
        if abs(alpha) <= self.heading_tolerance_rad:
            self._transition(NavigationState.CLOSED_LOOP_DRIVE, now)
            return self._output(f'[Lap 2] Aligned to WP{self.current_wp_idx} (err: {math.degrees(alpha):.1f}°); driving', linear=self.forward.speed_mps)
        return self._output(f'[Lap 2] Aligning to WP{self.current_wp_idx} (dist: {dist:.2f}m, err: {math.degrees(alpha):.1f}°)', angular=self.turn_calibration.rate(alpha))

    def _closed_loop_drive(self, now: float) -> NavigationOutput:
        """Drive closed-loop towards current waypoint using visual odometry pose."""
        wp = self.waypoints[self.current_wp_idx]
        self.target = self._nav_to_slam_point(wp.x_m, wp.y_m)
        rx, ry, rtheta = self.slam_to_nav_frame(*self._best_pose())
        dx, dy = wp.x_m - rx, wp.y_m - ry
        dist = math.hypot(dx, dy)
        if dist <= self.closed_loop_tolerance_m:
            self._transition(NavigationState.CLOSED_LOOP_CORNER_PAUSE, now)
            return self._output(f'[Lap 2] WP{self.current_wp_idx} reached (dist: {dist:.3f}m); corner pause')
        target_heading = math.atan2(dy, dx)
        alpha = wrap_angle(target_heading - rtheta)
        if abs(alpha) > self.heading_drive_threshold_rad:
            self._transition(NavigationState.CLOSED_LOOP_TURN, now)
            return self._output(f'[Lap 2] Heading drifted ({math.degrees(alpha):.1f}°); realigning to WP{self.current_wp_idx}', angular=self.turn_calibration.rate(alpha))
        return self._output(f'[Lap 2] Driving to WP{self.current_wp_idx} (dist: {dist:.2f}m, err: {math.degrees(alpha):.1f}°)', linear=self.forward.speed_mps)

    def _closed_loop_corner_pause(self, now: float) -> NavigationOutput:
        """Pause at corner waypoint to observe landmarks before advancing to next leg."""
        if now - self.phase_started + 1e-9 < self.policy.manual_forward_pause_s:
            return self._output(f'[Lap 2] WP{self.current_wp_idx} corner pause; updating landmarks')
        if self.current_wp_idx >= len(self.waypoints) - 1:
            self._transition(NavigationState.CLOSED_LOOP_FINAL_REORIENT, now)
            return self._closed_loop_final_reorient(now)
        self.current_wp_idx += 1
        self.side_index += 1
        wp = self.waypoints[self.current_wp_idx]
        self.target = self._nav_to_slam_point(wp.x_m, wp.y_m)
        self._transition(NavigationState.CLOSED_LOOP_TURN, now)
        return self._closed_loop_turn(now)

    def _closed_loop_final_reorient(self, now: float) -> NavigationOutput:
        """Reorient robot to 0 radians at origin upon completing Lap 2."""
        _, _, rtheta = self.slam_to_nav_frame(*self._best_pose())
        alpha = wrap_angle(0.0 - rtheta)
        if abs(alpha) <= self.heading_tolerance_rad:
            self.state, self.is_active = NavigationState.COMPLETED, False
            return self._output('Square traversal complete: 2 laps (dead-reckoning + closed loop visual odometry)')
        return self._output(f'[Lap 2] Final reorienting to 0.0 rad (err: {math.degrees(alpha):.1f}°)', angular=self.turn_calibration.rate(alpha))


class SquareNavigationNode(BaseNavigationNode):
    """ROS 2 Node dedicated to running Square missions."""

    def __init__(self):
        """Initialize ROS 2 square navigation node, parameters, and controller."""
        super().__init__('square_navigation_node')
        policy = NavigationPolicy(
            manual_motion_enabled=True,
            two_lap_square=bool(self.get_parameter('navigation.two_lap_square').value) if self.has_parameter('navigation.two_lap_square') else True,
            corner_turn_duration_s=float(self.get_parameter('navigation.corner_turn_duration_s').value) if self.has_parameter('navigation.corner_turn_duration_s') else 0.65,
            manual_forward_pause_s=float(self.get_parameter('navigation.manual_forward_pause_s').value) if self.has_parameter('navigation.manual_forward_pause_s') else 1.0,
            manual_midpoint_pause_s=float(self.get_parameter('navigation.manual_midpoint_pause_s').value) if self.has_parameter('navigation.manual_midpoint_pause_s') else 1.0,
            closed_loop_waypoint_tolerance_m=float(self.get_parameter('navigation.closed_loop_waypoint_tolerance_m').value) if self.has_parameter('navigation.closed_loop_waypoint_tolerance_m') else 0.12,
            heading_drive_threshold_rad=float(self.get_parameter('navigation.heading_drive_threshold_rad').value) if self.has_parameter('navigation.heading_drive_threshold_rad') else 0.35,
        )
        self.navigator = SquareNavigator(
            policy=policy,
            side_count=4,
            side_length_m=float(self.get_parameter('navigation.side_length_m').value or 0.0),
            square_size_ft=float(self.get_parameter('navigation.square_size_ft').value or 6.0),
            turn_calibration=self.calibration,
            forward_calibration=self.forward_calibration,
            waypoint_tolerance_m=float(self.get_parameter('navigation.waypoint_tolerance_m').value),
            heading_tolerance_rad=float(self.get_parameter('navigation.heading_tolerance_rad').value),
        )
        if self.get_parameter('navigation.auto_start').value:
            self.get_logger().info('Square auto-start enabled: starting traversal.')
            self.navigator.start()
        else:
            self.get_logger().info('Square Navigation Node initialized in IDLE mode. Waiting for /start_navigation.')


def navigation_node_main(args=None):
    """Universal navigation entry point: runs OctagonNavigationNode if side_count==8, else SquareNavigationNode."""
    rclpy.init(args=args)
    # Temporary check parameter or run square navigation node
    node = SquareNavigationNode()
    if node.has_parameter('navigation.side_count') and int(node.get_parameter('navigation.side_count').value) == 8:
        # Switch to Octagon node
        node.destroy_node()
        from .octagon_navigation import OctagonNavigationNode
        node = OctagonNavigationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
