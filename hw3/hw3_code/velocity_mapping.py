"""Map HW3 velocities to the existing motor model and report applied command estimates."""

import time

import rclpy
from geometry_msgs.msg import Twist, TwistStamped
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from std_msgs.msg import Float32MultiArray

from .motor_model import MOTOR_DEFAULTS, MotorModel


class VelocityToMotorNode(Node):
    """Use our wheel mapping with surface-specific turning input and command reporting."""

    def __init__(self):
        """Connect HW3-only topics and a 20 Hz command heartbeat."""
        super().__init__('hw3_velocity_mapping')
        self.values = {key: self.declare_parameter(key, value).value for key, value in MOTOR_DEFAULTS.items()}
        self.model = MotorModel(self.values)
        self.command, self.received = (0., 0.), None
        self.subscription = self.create_subscription(Twist, '/hw3/cmd_vel', self.on_command, 1)
        self.motor_pub = self.create_publisher(Float32MultiArray, '/hw3/motor_commands', 1)
        self.motion_pub = self.create_publisher(TwistStamped, '/hw3/applied_motion', 10)
        self.timer = self.create_timer(0.05, self.tick)

    def on_command(self, message):
        """Cache body requests; the motor model rejects invalid or mixed drive/turn inputs."""
        linear, angular = message.linear.x, message.angular.z
        self.command, self.received = (linear, angular), time.monotonic()

    def tick(self):
        """Publish motor commands and the corresponding model velocities at the same stamp."""
        stale = (self.received is None or time.monotonic()-self.received > self.values['command_timeout_s']
                 or self.motor_pub.get_subscription_count() == 0)
        linear, angular = (0., 0.) if stale else self.command
        left, right, applied_linear, applied_angular = self.model.map_command(linear, angular)
        motors = Float32MultiArray()
        motors.data = [left, right]
        self.motor_pub.publish(motors)
        motion = TwistStamped()
        motion.header.stamp = self.get_clock().now().to_msg()
        motion.header.frame_id = 'base_link'
        motion.twist.linear.x, motion.twist.angular.z = applied_linear, applied_angular
        self.motion_pub.publish(motion)

    def destroy_node(self):
        """Send a final stop before tearing down the mapper."""
        if rclpy.ok():
            stop = Float32MultiArray()
            stop.data = [0., 0.]
            self.motor_pub.publish(stop)
        super().destroy_node()


def main(args=None):
    """Run only the HW3 motor mapping node until shutdown."""
    rclpy.init(args=args)
    node = None
    try:
        node = VelocityToMotorNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
