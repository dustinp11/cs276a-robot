"""Run left/right motor inputs of -0.3/0.3 for 10 seconds, then stop."""

import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class RobotTestNode(Node):
    """Publish raw [left, right] motor inputs at 20 Hz."""

    def __init__(self):
        super().__init__('robot_test')
        self.publisher = self.create_publisher(
            Float32MultiArray, 'motor_commands', 10)
        self.start_time = None
        self.finished = False
        self.timer = self.create_timer(0.05, self.publish_motor_command)
        self.get_logger().info('Waiting for motor_control before starting')

    def publish_command(self, left=0.0, right=0.0):
        """Publish motor inputs in [left, right] order."""
        message = Float32MultiArray()
        message.data = [left, right]
        self.publisher.publish(message)

    def publish_motor_command(self):
        """Run once for 10 wall-clock seconds after a subscriber connects."""
        now = time.monotonic()
        if self.publisher.get_subscription_count() == 0:
            if self.start_time is not None:
                self.finished = True
            self.publish_command()
            return
        if self.start_time is None:
            self.start_time = now
            self.get_logger().info('Running L=-0.3, R=0.3 for 10 seconds')
        self.publish_timed_command(now)

    def publish_timed_command(self, now):
        """Keep publishing zero after completion or subscriber loss."""
        if not self.finished and now - self.start_time >= 10.0:
            self.finished = True
            self.get_logger().info('10 seconds complete; motors stopped')
        if self.finished:
            self.publish_command()
        else:
            self.publish_command(-0.3, 0.3)


def main(args=None):
    """Run the motor test and send stop commands on exit."""
    rclpy.init(args=args)
    node = RobotTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop_node(node)


def stop_node(node):
    """Send repeated stops before releasing the ROS node."""
    if rclpy.ok():
        for _ in range(3):
            node.publish_command()
            time.sleep(0.05)
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
