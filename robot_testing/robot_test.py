
import math
import time
from collections import deque
from pathlib import Path

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


# Calibrated parameters
FORWARD_INPUT = 0.3
FORWARD_SPEED = 0.4          # m/s

ROTATION_INPUT = 0.5
ANGULAR_SPEED_CW = math.pi / 3   # rad/s
ANGULAR_SPEED_CCW = math.pi / 3  # assumed symmetric


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

        # Load waypoints
        self.waypoints = load_waypoints(get_waypoint_file())

        if not self.waypoints:
            raise ValueError("No waypoints found")

        # Assume robot starts at first waypoint
        self.x, self.y, self.theta = self.waypoints[0]

        # Motion execution
        self.command_queue = deque()
        self.motor_command = [0.0, 0.0]
        self.command_end_time = None
        self.active_movement = None
        self.finished = False
        self.warned_no_subscriber = False

        # Generate movements for all waypoints
        self.plan_waypoints()

        # Execute commands using ROS2 timer
        self.timer = self.create_timer(
            0.05,
            self.publish_motor_command
        )

        self.get_logger().info(
            f"Loaded {len(self.waypoints)} waypoints"
        )

    def move(self, dist):
        """Queue forward movement by dist meters."""
        if dist <= 1e-6:
            return

        duration = dist / FORWARD_SPEED

        self.command_queue.append((
            [FORWARD_INPUT, FORWARD_INPUT],
            duration,
            "move",
            dist
        ))

    
    def rotate(self, angle):
        """Queue rotation. Always rotate counterclockwise."""

        # Convert any angle to [0, 2*pi)
        angle = angle % (2 * math.pi)

        if angle < 1e-6 or 2 * math.pi - angle < 1e-6:
            return

        duration = angle / ANGULAR_SPEED_CCW

        self.command_queue.append((
            [-ROTATION_INPUT, ROTATION_INPUT],
            duration,
            "rotate",
            angle
        ))


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
        if movement_type == "move":
            self.x += value * math.cos(self.theta)
            self.y += value * math.sin(self.theta)

        elif movement_type == "rotate":
            self.theta = wrap(self.theta + value)

        self.get_logger().info(
            f"Estimated pose: "
            f"x={self.x:.3f}, "
            f"y={self.y:.3f}, "
            f"theta={self.theta:.3f}"
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
                    "Motor subscriber disconnected. "
                    "Aborting waypoint traversal."
                )
                self.command_queue.clear()
                self.active_movement = None
                self.command_end_time = None
                self.finished = True

            self.motor_command = [0.0, 0.0]
            self.publish_command()
            return

        # Check whether movement is finished
        if self.command_end_time is not None:
            if now >= self.command_end_time:
                self.motor_command = [0.0, 0.0]
                self.command_end_time = None

                movement_type, value = self.active_movement
                self.update_pose(movement_type, value)
                self.active_movement = None

                # Publish stop before starting next motion
                self.publish_command()
                return

        # Start next movement
        elif self.command_queue and not self.finished:
            command, duration, movement_type, value = (
                self.command_queue.popleft()
            )

            self.motor_command = command
            self.command_end_time = now + duration
            self.active_movement = (movement_type, value)

            self.get_logger().info(
                f"Starting {movement_type}: "
                f"{value:.3f}, duration={duration:.3f}s"
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
