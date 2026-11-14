"""ROS adapter for the HW3 fixed-turn velocity mapper."""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray

from hw3.turn_calibration import read_forward_calibration, read_turn_calibration
from hw3.velocity_mapping import VelocityMapper


class VelocityToMotorNode(Node):
    """Publish motor inputs and model velocity from the same mapping decision."""

    def __init__(self):
        """Create the mapper and ROS interfaces without commanding the motors.

        Args:
            None.

        Returns:
            None.
        """
        super().__init__('velocity_mapping')
        calibration = read_turn_calibration(self)
        if calibration is None:
            raise ValueError('HW3 mapping requires enabled, measured turn calibration.')
        self.mapper = VelocityMapper(calibration, read_forward_calibration(self))
        self.motor_pub = self.create_publisher(Float32MultiArray, '/motor_commands', 10)
        self.applied_pub = self.create_publisher(Twist, '/cmd_vel_applied', 10)
        self.subscription = self.create_subscription(Twist, '/cmd_vel', self.on_command, 10)

    def on_command(self, message: Twist) -> None:
        """Map and publish one straight, pivot, or stopped command."""
        command = self.mapper.map_command(message.linear.x, message.angular.z)
        self.motor_pub.publish(Float32MultiArray(data=[command.left, command.right]))
        applied = Twist()
        applied.linear.x = command.linear_mps
        applied.angular.z = command.angular_radps
        self.applied_pub.publish(applied)


def main(args=None):
    """Run the mapping adapter until ROS shutdown.

    Args:
        args: Optional ROS command-line arguments.

    Returns:
        None.
    """
    rclpy.init(args=args)
    node = VelocityToMotorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
