
import math
import threading
import time
from collections import deque
from pathlib import Path

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray


# ── Calibrated motion parameters ─────────────────────────────────────────────
FORWARD_INPUT = 0.3
FORWARD_SPEED = 0.4          # m/s

# Kept for IMU-unavailable fallback only
ROTATION_INPUT = 0.5
ANGULAR_SPEED_CW  = math.pi / 3   # rad/s
ANGULAR_SPEED_CCW = math.pi / 3   # assumed symmetric

# ── Fix 1: Closed-loop turn controller ───────────────────────────────────────
# The robot rotates until the IMU-measured yaw change matches the planned angle.
# Motor effort scales with remaining error: effort = TURN_KP * error_radians,
# clamped to [-MAX_TURN_INPUT, MAX_TURN_INPUT].
TURN_KP = 0.4                # proportional gain (tune up if turns are sluggish)
MAX_TURN_INPUT = 0.5         # motor command cap for turns
TURN_TOLERANCE = 0.03        # rad (~1.7°) – declare turn complete within this band

# ── Fix 2: Heading-hold controller (forward motion) ──────────────────────────
# While driving forward, the target heading is locked at motion start.
# A small differential correction is added/subtracted to keep the robot straight.
# Correction = HEADING_KP * heading_error, clamped to ±MAX_HEADING_CORRECTION.
HEADING_KP = 0.3
MAX_HEADING_CORRECTION = 0.15


def get_waypoint_file():
    """Find waypoints in the source tree or installed ROS package share."""
    source_file = Path(__file__).resolve().parent / "waypoints.txt"
    if source_file.is_file():
        return source_file

    try:
        share_file = Path(get_package_share_directory("robot_testing")) / "waypoints.txt"
    except PackageNotFoundError as error:
        raise FileNotFoundError(
            "Could not find the robot_testing ROS package or its waypoints.txt"
        ) from error

    if not share_file.is_file():
        raise FileNotFoundError(f"Waypoint file does not exist: {share_file}")

    return share_file


def wrap(angle):
    """Normalize angle to [0, 2*pi)."""
    return angle % (2 * math.pi)


def wrap_signed(angle):
    """Normalize angle to (-pi, pi] – used for small delta comparisons."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def load_waypoints(path):
    """Read x,y,theta from waypoint file."""
    waypoints = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            try:
                values = [float(value.strip()) for value in line.split(",")]
            except ValueError as error:
                raise ValueError(
                    f"Invalid waypoint at {path}:{line_number}: {line!r}"
                ) from error

            if len(values) != 3:
                raise ValueError(
                    f"Expected x,y,theta at {path}:{line_number}, got {line!r}"
                )

            x, y, theta = values
            waypoints.append((x, y, theta))

    return waypoints


class RobotTestNode(Node):
    def __init__(self):
        super().__init__("robot_test")

        self.publisher = self.create_publisher(
            Float32MultiArray,
            "motor_commands",
            10
        )

        # IMU yaw subscription
        self._imu_lock = threading.Lock()
        self._imu_yaw: float | None = None           
        self._initial_imu_yaw: float | None = None   

        self._imu_sub = self.create_subscription(
            Imu,
            "imu_data",
            self._imu_callback,
            10,
        )

        self._rot_prev_yaw: float | None = None   # IMU yaw at the previous timer tick
        self._rot_accumulated: float = 0.0         # total rotation measured so far
        self._rot_target: float = 0.0              # desired total rotation (rad, ≥ 0)

        self._move_target_heading: float | None = None  # IMU yaw locked at move start
        self._imu_warned = False

        self.waypoints = load_waypoints(get_waypoint_file())
        if not self.waypoints:
            raise ValueError("No waypoints found")

        self.x, self.y, self.theta = self.waypoints[0]

        self.command_queue: deque[tuple[str, float]] = deque()
        self.motor_command = [0.0, 0.0]
        self.command_end_time: float | None = None   # used for "move" and IMU-fallback "rotate"
        self.active_movement: tuple[str, float] | None = None
        self.finished = False
        self.warned_no_subscriber = False

        self.plan_waypoints()

        self.timer = self.create_timer(
            0.05,
            self.publish_motor_command
        )

        self.get_logger().info(
            f"Loaded {len(self.waypoints)} waypoints"
        )
        self.get_logger().info(
            f"Closed-loop turns active  (TURN_KP={TURN_KP}, "
            f"tolerance={math.degrees(TURN_TOLERANCE):.1f}°)"
        )
        self.get_logger().info(
            f"Heading-hold active  (HEADING_KP={HEADING_KP}, "
            f"max_correction=±{MAX_HEADING_CORRECTION})"
        )

    def _imu_callback(self, msg: Imu) -> None:
        """IMU quarternion to yaw"""
        qw = msg.orientation.w
        qx = msg.orientation.x
        qy = msg.orientation.y
        qz = msg.orientation.z

        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        with self._imu_lock:
            self._imu_yaw = yaw
            if self._initial_imu_yaw is None:
                self._initial_imu_yaw = yaw
                self.get_logger().info(
                    f"IMU reference locked: yaw₀ = {math.degrees(yaw):.1f}°"
                )

    def _get_imu_yaw(self) -> float | None:
        """Read of the latest raw IMU yaw."""
        with self._imu_lock:
            return self._imu_yaw

    def move(self, dist):
        """Queue forward movement by dist meters."""
        if dist <= 1e-6:
            return
        self.command_queue.append(("move", dist))

    def rotate(self, angle):
        """Queue rotation. Always rotate counterclockwise."""
        # Convert any angle to [0, 2*pi)
        angle = angle % (2 * math.pi)

        if angle < 1e-6 or 2 * math.pi - angle < 1e-6:
            return

        self.command_queue.append(("rotate", angle))

    def plan_waypoints(self):
        """Generate commands to traverse every waypoint."""

        # Planning pose, separate from executed pose
        px, py, ptheta = self.waypoints[0]

        for i, (gx, gy, gtheta) in enumerate(
            self.waypoints[1:], start=2
        ):
            dx = gx - px
            dy = gy - py

            distance = math.hypot(dx, dy)

            # Step 1: Face destination
            if distance > 1e-6:
                target_angle = math.atan2(dy, dx)

                initial_rotation = wrap(
                    target_angle - ptheta
                )
                self.rotate(initial_rotation)

                # Step 2: Drive to destination
                self.move(distance)

                ptheta = target_angle

            # Step 3: Match waypoint's final angle
            final_rotation = wrap(gtheta - ptheta)
            self.rotate(final_rotation)

            # Update planning pose
            px, py, ptheta = gx, gy, wrap(gtheta)

            self.get_logger().info(
                f"Planned waypoint {i}: "
                f"({gx}, {gy}, {gtheta})"
            )

    def update_pose(self, movement_type, value):
        """Update estimated pose using kinematic equations."""
        with self._imu_lock:
            current_imu_yaw = self._imu_yaw
            initial_imu_yaw = self._initial_imu_yaw

        if movement_type == "move":
            # Use the live IMU heading for the position update when available;
            # it reflects any residual heading error after heading-hold.
            if current_imu_yaw is not None and initial_imu_yaw is not None:
                self.theta = wrap(current_imu_yaw - initial_imu_yaw)
            self.x += value * math.cos(self.theta)
            self.y += value * math.sin(self.theta)

        elif movement_type == "rotate":
            # value = actual IMU-accumulated rotation for this turn
            self.theta = wrap(self.theta + value)

        self.get_logger().info(
            f"Estimated pose: "
            f"x={self.x:.3f}, "
            f"y={self.y:.3f}, "
            f"theta={math.degrees(self.theta):.1f}°"
        )

    def publish_motor_command(self):
        now = time.monotonic()

        # Do not start without motor controller
        if self.publisher.get_subscription_count() == 0:
            if not self.warned_no_subscriber:
                self.get_logger().warning(
                    "Waiting for motor_control subscriber"
                )
                self.warned_no_subscriber = True

            # A running movement is no longer reliable
            if self.active_movement is not None:
                self.get_logger().error(
                    "Motor subscriber disconnected. Aborting waypoint traversal."
                )
                self.command_queue.clear()
                self.active_movement = None
                self.command_end_time = None
                self.finished = True

            self.motor_command = [0.0, 0.0]
            self.publish_command()
            return

        current_imu_yaw = self._get_imu_yaw()

        if self.active_movement is not None:
            movement_type, value = self.active_movement

            #updated IMU rotate
            if movement_type == "rotate":
                if current_imu_yaw is None:
                    # IMU not available, fall back to timed rotation
                    if not self._imu_warned:
                        self.get_logger().warning(
                            "IMU unavailable – using timed rotation fallback"
                        )
                        self._imu_warned = True
                    if self.command_end_time is not None and now >= self.command_end_time:
                        self.motor_command = [0.0, 0.0]
                        self.command_end_time = None
                        self.update_pose("rotate", value)
                        self.active_movement = None
                        self.publish_command()
                        return
                    self.motor_command = [-ROTATION_INPUT, ROTATION_INPUT]
                    self.publish_command()
                    return

                # accumulate the yaw change since the last timer tick.
                # wrap_signed handles ±π crossovers.
                if self._rot_prev_yaw is not None:
                    delta = wrap_signed(current_imu_yaw - self._rot_prev_yaw)
                    self._rot_accumulated += delta
                self._rot_prev_yaw = current_imu_yaw

                error = self._rot_target - self._rot_accumulated

                if abs(error) < TURN_TOLERANCE:
                    self.motor_command = [0.0, 0.0]
                    self.active_movement = None
                    self.update_pose("rotate", self._rot_accumulated)
                    self.publish_command()
                    return

                # P controller, positive error rotate more CCW
                effort = TURN_KP * error
                effort = max(-MAX_TURN_INPUT, min(MAX_TURN_INPUT, effort))
                self.motor_command = [-effort, effort]

            # slanted drift error adjustment
            elif movement_type == "move":
                if now >= self.command_end_time:
                    self.motor_command = [0.0, 0.0]
                    self.command_end_time = None
                    self.update_pose("move", value)
                    self.active_movement = None
                    self.publish_command()
                    return

                if (current_imu_yaw is not None and self._move_target_heading is not None):
                    # positive heading error, correct CW.
                    heading_error = wrap_signed(
                        self._move_target_heading - current_imu_yaw
                    )
                    correction = HEADING_KP * heading_error
                    correction = max(-MAX_HEADING_CORRECTION,
                                     min(MAX_HEADING_CORRECTION, correction))
                    L = max(-0.5, min(0.5, FORWARD_INPUT + correction))
                    R = max(-0.5, min(0.5, FORWARD_INPUT - correction))
                    self.motor_command = [L, R]
                else:
                    self.motor_command = [FORWARD_INPUT, FORWARD_INPUT]

        elif self.command_queue and not self.finished:
            movement_type, value = self.command_queue.popleft()
            self.active_movement = (movement_type, value)

            if movement_type == "rotate":
                self._rot_accumulated = 0.0
                self._rot_target = value
                self._rot_prev_yaw = current_imu_yaw

                if current_imu_yaw is None:
                    # Fallback: compute duration and turn at fixed speed
                    duration = value / ANGULAR_SPEED_CCW
                    self.command_end_time = now + duration
                    self.motor_command = [-ROTATION_INPUT, ROTATION_INPUT]
                    self.get_logger().warning(
                        f"Starting timed rotation: "
                        f"{math.degrees(value):.1f}°, "
                        f"duration={duration:.2f}s (no IMU)"
                    )
                else:
                    # Initial effort from P controller
                    effort = min(MAX_TURN_INPUT, TURN_KP * value)
                    self.motor_command = [-effort, effort]
                    self.get_logger().info(
                        f"Starting closed-loop rotation: "
                        f"target={math.degrees(value):.1f}°"
                    )

            elif movement_type == "move":
                duration = value / FORWARD_SPEED
                self.command_end_time = now + duration
                self._move_target_heading = current_imu_yaw  # lock heading now

                self.motor_command = [FORWARD_INPUT, FORWARD_INPUT]
                heading_str = (
                    f"{math.degrees(current_imu_yaw):.1f}°"
                    if current_imu_yaw is not None
                    else "none (no IMU)"
                )
                self.get_logger().info(
                    f"Starting move: {value:.3f} m, "
                    f"duration={duration:.3f}s, "
                    f"heading locked={heading_str}"
                )

        # All movements completed
        elif not self.command_queue and not self.finished:
            self.finished = True
            self.motor_command = [0.0, 0.0]

            self.get_logger().info(
                "All waypoints completed!"
            )

        self.publish_command()

    def publish_command(self):
        message = Float32MultiArray()
        message.data = self.motor_command
        self.publisher.publish(message)


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = RobotTestNode()
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        if node is not None:
            node.motor_command = [0.0, 0.0]
            for _ in range(3):
                node.publish_command()
                time.sleep(0.05)
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
