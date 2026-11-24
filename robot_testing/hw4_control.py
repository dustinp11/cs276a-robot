"""Closed-Loop Safe Voronoi Navigation Controller for HW4.

Features:
1. Voronoi-based safe path planning from Start (234.5, 63.0) cm to Goal (65.0, 229.5) cm.
2. Closed-loop visual waypoint following using known AprilTag coordinates (no SLAM).
3. Stop-measure-correct state machine:
   - Orient toward waypoint.
   - Drive forward bounded distance.
   - Settle camera and re-localize using AprilTags.
   - Recalculate distance and heading error.
   - Advance waypoint when within tolerance.
4. Robust localization failure handling:
   - Halts robot immediately if visual pose is lost or stale.
   - Executes search/reacquisition pulse if needed.
   - Resumes navigation only after confident visual pose is re-acquired.
5. Logs telemetry and writes trajectory to hw4_trajectory.txt / coordinates.txt.
"""

from enum import Enum
import json
import math
from pathlib import Path
import threading
import time
from typing import List, Optional, Tuple

try:
    import rclpy
    from geometry_msgs.msg import PoseStamped, Twist
    from rclpy.node import Node
    from tf2_ros import Buffer, TransformListener
except ImportError:
    rclpy = None
    PoseStamped = None
    Twist = None
    Node = object
    Buffer = None
    TransformListener = None

try:
    from apriltag_msgs.msg import AprilTagDetectionArray
except ImportError:
    AprilTagDetectionArray = None

try:
    from hw2_interfaces.srv import GetRobotPose
except ImportError:
    GetRobotPose = None

from robot_testing.voronoi_planner import (
    Environment,
    Obstacle,
    PlannedPath,
    RobotFootprint,
    SafePathPlanner,
    VoronoiGraph,
    Workspace,
)


class NavigationState(Enum):
    IDLE = 'IDLE'
    PLANNING = 'PLANNING'
    ACQUIRING_INITIAL_POSE = 'ACQUIRING_INITIAL_POSE'
    ORIENTING = 'ORIENTING'
    DRIVING = 'DRIVING'
    SETTLING = 'SETTLING'
    LOCALIZATION_RECOVERY = 'LOCALIZATION_RECOVERY'
    WAYPOINT_REACHED = 'WAYPOINT_REACHED'
    GOAL_REACHED = 'GOAL_REACHED'
    STOPPED = 'STOPPED'


def wrap_angle(angle_rad: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


class HW4Navigator:
    """Core closed-loop navigation logic separated from ROS 2 wrappers."""

    def __init__(
        self,
        start_cm: Tuple[float, float] = (234.5, 63.0),
        goal_cm: Tuple[float, float] = (65.0, 229.5),
        position_tolerance_m: float = 0.06,
        heading_tolerance_rad: float = 0.12,  # ~6.9 deg
        max_forward_step_m: float = 0.25,      # 25 cm bounded visual drive segments
        drive_speed_mps: float = 0.18,         # calibrated forward speed
        turn_rate_radps: float = 1.60,         # calibrated turn rate (~3.75s per 360 deg)
        settle_time_s: float = 1.0,            # camera stabilization pause
        max_pose_age_s: float = 0.8,           # maximum age of visual pose
    ):
        self.start_cm = start_cm
        self.goal_cm = goal_cm
        self.pos_tol = position_tolerance_m
        self.heading_tol = heading_tolerance_rad
        self.max_step = max_forward_step_m
        self.drive_speed = drive_speed_mps
        self.turn_rate = turn_rate_radps
        self.settle_time = settle_time_s
        self.max_pose_age = max_pose_age_s

        self.env = Environment()
        self.vg = VoronoiGraph(self.env, boundary_sample_step_cm=2.0)
        self.planner = SafePathPlanner(self.env, self.vg)

        self.path: Optional[PlannedPath] = None
        self.waypoints_m: List[Tuple[float, float]] = []
        self.current_wp_idx = 0
        self.state = NavigationState.IDLE

        # Telemetry
        self.total_traveled_m = 0.0
        self.current_x_m = start_cm[0] / 100.0
        self.current_y_m = start_cm[1] / 100.0
        self.current_yaw_rad = 0.0
        self.last_pose_time = 0.0
        self.recovery_attempts = 0

    def plan_path(self) -> PlannedPath:
        """Compute the safe Voronoi path."""
        self.state = NavigationState.PLANNING
        self.path = self.planner.plan(self.start_cm, self.goal_cm, smoothing_tolerance_cm=2.0)
        # Convert waypoints from cm to meters for navigation
        self.waypoints_m = [(x / 100.0, y / 100.0) for x, y in self.path.smoothed_waypoints]
        self.current_wp_idx = 0
        return self.path

    @property
    def current_waypoint(self) -> Optional[Tuple[float, float]]:
        if 0 <= self.current_wp_idx < len(self.waypoints_m):
            return self.waypoints_m[self.current_wp_idx]
        return None

    @property
    def is_finished(self) -> bool:
        return self.state == NavigationState.GOAL_REACHED

    def update_pose(self, x_m: float, y_m: float, yaw_rad: float, stamp_s: float):
        self.current_x_m = x_m
        self.current_y_m = y_m
        self.current_yaw_rad = yaw_rad
        self.last_pose_time = stamp_s

    def compute_motion_command(
        self,
        current_time_s: float,
    ) -> Tuple[float, float, str]:
        """Compute (linear_velocity_mps, angular_velocity_radps, status_msg)."""
        # 1. Check pose freshness
        pose_age = current_time_s - self.last_pose_time
        if self.last_pose_time == 0.0 or pose_age > self.max_pose_age:
            self.state = NavigationState.LOCALIZATION_RECOVERY
            return 0.0, 0.0, f"Localization stale (age={pose_age:.2f}s > {self.max_pose_age}s) - STOPPING"

        wp = self.current_waypoint
        if wp is None:
            self.state = NavigationState.GOAL_REACHED
            return 0.0, 0.0, "All waypoints reached successfully!"

        dx = wp[0] - self.current_x_m
        dy = wp[1] - self.current_y_m
        dist = math.hypot(dx, dy)

        # Check waypoint arrival
        target_tol = self.pos_tol if self.current_wp_idx < len(self.waypoints_m) - 1 else 0.05
        if dist <= target_tol:
            self.current_wp_idx += 1
            if self.current_wp_idx >= len(self.waypoints_m):
                self.state = NavigationState.GOAL_REACHED
                return 0.0, 0.0, f"Goal ({wp[0]:.2f}, {wp[1]:.2f}) reached! Distance={dist:.3f}m"
            self.state = NavigationState.WAYPOINT_REACHED
            next_wp = self.current_waypoint
            return 0.0, 0.0, f"WP {self.current_wp_idx - 1} reached. Advancing to WP {self.current_wp_idx}: {next_wp}"

        # Calculate desired heading
        desired_yaw = math.atan2(dy, dx)
        heading_err = wrap_angle(desired_yaw - self.current_yaw_rad)

        # 2. If heading error exceeds tolerance, rotate in place
        if abs(heading_err) > self.heading_tol:
            self.state = NavigationState.ORIENTING
            turn_dir = 1.0 if heading_err > 0 else -1.0
            return 0.0, turn_dir * self.turn_rate, f"Orienting: heading_err={math.degrees(heading_err):.1f}° toward WP {self.current_wp_idx}"

        # 3. Aligned: drive forward
        self.state = NavigationState.DRIVING
        step_speed = self.drive_speed if dist > 0.15 else max(0.10, self.drive_speed * 0.7)
        return step_speed, 0.0, f"Driving: dist={dist:.3f}m, heading_err={math.degrees(heading_err):.1f}° to WP {self.current_wp_idx}"


class HW4ControlNode(Node):
    """ROS 2 Node executing safe Voronoi path planning and closed-loop visual navigation."""

    def __init__(self):
        super().__init__('hw4_control')

        # Publishers & Subscribers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pose_sub = self.create_subscription(PoseStamped, '/robot_pose', self.on_pose, 10)

        # Trajectory logging paths
        self.coord_paths = [
            Path('hw4_trajectory.txt').resolve(),
            Path('coordinates.txt').resolve(),
        ]
        for p in self.coord_paths:
            try:
                p.write_text('', encoding='utf-8')
            except Exception:
                pass

        # Parameters
        self.declare_parameter('start_x_cm', 234.5)
        self.declare_parameter('start_y_cm', 63.0)
        self.declare_parameter('goal_x_cm', 65.0)
        self.declare_parameter('goal_y_cm', 229.5)
        self.declare_parameter('position_tolerance_m', 0.06)
        self.declare_parameter('heading_tolerance_deg', 7.0)
        self.declare_parameter('speed_mps', 0.18)
        self.declare_parameter('turn_time_360_s', 3.75)
        self.declare_parameter('settle_time_s', 1.0)

        start_x = self.get_parameter('start_x_cm').value
        start_y = self.get_parameter('start_y_cm').value
        goal_x = self.get_parameter('goal_x_cm').value
        goal_y = self.get_parameter('goal_y_cm').value

        self.navigator = HW4Navigator(
            start_cm=(start_x, start_y),
            goal_cm=(goal_x, goal_y),
            position_tolerance_m=self.get_parameter('position_tolerance_m').value,
            heading_tolerance_rad=math.radians(self.get_parameter('heading_tolerance_deg').value),
            drive_speed_mps=self.get_parameter('speed_mps').value,
            turn_rate_radps=2.0 * math.pi / self.get_parameter('turn_time_360_s').value,
            settle_time_s=self.get_parameter('settle_time_s').value,
        )

        self.get_logger().info('=' * 60)
        self.get_logger().info('HW4 SAFE VORONOI NAVIGATION CONTROLLER INITIALIZING')
        self.get_logger().info(f'Start: ({start_x}, {start_y}) cm | Goal: ({goal_x}, {goal_y}) cm')

        # Plan the safe Voronoi path immediately
        path = self.navigator.plan_path()
        self.get_logger().info(f'Voronoi Path planned: {path.waypoint_count} waypoints, length={path.total_length_cm:.1f} cm')
        self.get_logger().info(f'Bottleneck clearance: {path.bottleneck_clearance_cm:.2f} cm (Min: {path.min_clearance_cm:.2f} cm)')
        for i, (wx, wy) in enumerate(path.smoothed_waypoints):
            self.get_logger().info(f'  WP {i}: ({wx:.1f}, {wy:.1f}) cm')
        self.get_logger().info('=' * 60)

        # Control thread
        self.running = True
        self.lock = threading.Lock()
        self.mission_thread = threading.Thread(target=self.run_mission, daemon=True)
        self.mission_thread.start()

    def on_pose(self, msg: PoseStamped):
        """Update robot pose from visual AprilTag localization."""
        px = msg.pose.position.x
        py = msg.pose.position.y
        qz = msg.pose.orientation.z
        qw = msg.pose.orientation.w
        yaw = 2.0 * math.atan2(qz, qw)
        now_s = time.monotonic()

        with self.lock:
            self.navigator.update_pose(px, py, yaw, now_s)

        # Log coordinate to file
        coord_line = f"({px * 100.0:.2f}, {py * 100.0:.2f})\n"
        for p in self.coord_paths:
            try:
                with open(p, 'a', encoding='utf-8') as f:
                    f.write(coord_line)
            except Exception:
                pass

    def stop_robot(self):
        """Publish zero velocity to stop motors."""
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)

    def execute_turn(self, angle_rad: float):
        """Execute a calibrated rotation in place with settled re-localization."""
        if abs(angle_rad) < math.radians(3.0) or not self.running:
            return
        direction = 1.0 if angle_rad > 0 else -1.0
        duration = abs(angle_rad) / self.navigator.turn_rate

        self.get_logger().info(
            f"Rotating {math.degrees(angle_rad):.1f}° ({direction:+1.0f}) for {duration:.2f}s..."
        )
        end_time = time.monotonic() + duration
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = float(direction * self.navigator.turn_rate)

        while time.monotonic() < end_time and self.running and rclpy.ok():
            self.cmd_pub.publish(twist)
            time.sleep(0.04)

        self.stop_robot()
        time.sleep(self.navigator.settle_time)

    def execute_drive(self, distance_m: float):
        """Execute a forward segment bounded by max_step with settled re-localization."""
        if distance_m <= 0.01 or not self.running:
            return
        step = min(distance_m, self.navigator.max_step)
        duration = step / self.navigator.drive_speed

        self.get_logger().info(
            f"Driving forward {step * 100.0:.1f} cm (duration {duration:.2f}s)..."
        )
        end_time = time.monotonic() + duration
        twist = Twist()
        twist.linear.x = float(self.navigator.drive_speed)
        twist.angular.z = 0.0

        while time.monotonic() < end_time and self.running and rclpy.ok():
            self.cmd_pub.publish(twist)
            time.sleep(0.04)

        self.stop_robot()
        time.sleep(self.navigator.settle_time)

    def run_mission(self):
        """Main waypoint-following control loop."""
        time.sleep(1.0)
        self.get_logger().info("Mission thread started. Waiting for initial AprilTag pose...")

        # 1. Wait for visual pose
        while self.running and rclpy.ok():
            with self.lock:
                last_time = self.navigator.last_pose_time
            if last_time > 0 and (time.monotonic() - last_time) <= 1.0:
                self.get_logger().info(
                    f"Initial visual pose acquired: ({self.navigator.current_x_m * 100.0:.1f}, "
                    f"{self.navigator.current_y_m * 100.0:.1f}) cm, "
                    f"yaw={math.degrees(self.navigator.current_yaw_rad):.1f}°"
                )
                break
            time.sleep(0.2)

        # 2. Closed-loop waypoint traversal
        while self.running and rclpy.ok() and not self.navigator.is_finished:
            now_s = time.monotonic()
            with self.lock:
                lin_v, ang_v, status = self.navigator.compute_motion_command(now_s)
                curr_state = self.navigator.state
                curr_wp_idx = self.navigator.current_wp_idx
                wp = self.navigator.current_waypoint
                cx, cy = self.navigator.current_x_m, self.navigator.current_y_m
                cyaw = self.navigator.current_yaw_rad

            self.get_logger().info(f"[{curr_state.value}] {status}")

            if curr_state == NavigationState.GOAL_REACHED:
                self.stop_robot()
                self.get_logger().info("=" * 60)
                self.get_logger().info("GOAL REACHED! Safe Voronoi mission complete!")
                self.get_logger().info("=" * 60)
                break

            if curr_state == NavigationState.LOCALIZATION_RECOVERY:
                self.stop_robot()
                self.get_logger().warn("Visual localization lost! Pausing to reacquire AprilTags...")
                time.sleep(1.5)
                continue

            if curr_state == NavigationState.ORIENTING:
                # Calculate heading error and execute pulse
                dx = wp[0] - cx
                dy = wp[1] - cy
                desired_yaw = math.atan2(dy, dx)
                heading_err = wrap_angle(desired_yaw - cyaw)
                self.execute_turn(heading_err)
                continue

            if curr_state == NavigationState.DRIVING:
                dx = wp[0] - cx
                dy = wp[1] - cy
                dist = math.hypot(dx, dy)
                self.execute_drive(dist)
                continue

            time.sleep(0.1)

        self.stop_robot()


def main(args=None):
    rclpy.init(args=args)
    node = HW4ControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Navigation interrupted by user.')
    finally:
        node.running = False
        node.stop_robot()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
