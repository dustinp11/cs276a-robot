import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class RobotTestNode(Node):
    command_sequence = (
        [0.3, -0.3],
        [0.3, -0.3], 
        [0.4, 0.4]
    )

    def __init__(self):
        super().__init__('robot_test')
        self.publisher = self.create_publisher(Float32MultiArray, 'motor_commands', 10)
        self.started_at = time.monotonic()
        self.warned_no_subscriber = False
        self.timer = self.create_timer(0.05, self.publish_motor_command)
        self.get_logger().info('Running the movement sequence with a 1-second stop after each move')

    def publish_motor_command(self):
        now = time.monotonic()
        if self.publisher.get_subscription_count() == 0 and not self.warned_no_subscriber:
            self.get_logger().warning(
                'No motor_control subscriber on motor_commands; commands are being published '
                'but will not move the robot until motor_control is running'
            )
            self.warned_no_subscriber = True

        elapsed = now - self.started_at
        phase = int(elapsed)
        if phase < len(self.command_sequence):
            command = self.command_sequence[phase]
        else:
            command = [0.0, 0.0]

        message = Float32MultiArray()
        message.data = command
        self.publisher.publish(message)


def main(args=None):
    rclpy.init(args=args)
    node = RobotTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
