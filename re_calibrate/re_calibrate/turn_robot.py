import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

TURN_DURATION_SECONDS = 3.85 / 4 


class TurnRobotNode(Node):
    def __init__(self):
        super().__init__('re_calibrate')
        self.publisher = self.create_publisher(
            Float32MultiArray, 'motor_commands', 10
        )
        self.started_at = None
        self.finished = False
        self.stopped = False
        self.timer = None

    def wait_for_motor_driver(self, timeout=10.0):
        self.get_logger().info(
            'Waiting for a motor driver on /motor_commands...'
        )
        deadline = time.monotonic() + timeout
        while rclpy.ok() and self.count_subscribers('motor_commands') == 0:
            if time.monotonic() >= deadline:
                self.get_logger().error(
                    'No subscriber appeared on /motor_commands. Start the '
                    'robot_testing motor_control node, then run this command again.'
                )
                return False
            rclpy.spin_once(self, timeout_sec=0.1)

        if not rclpy.ok():
            return False

        self.get_logger().info(
            f'Motor driver connected; turning for {TURN_DURATION_SECONDS} seconds.'
        )
        self.started_at = time.monotonic()
        self._publish_turn_command()
        self.timer = self.create_timer(0.05, self._publish_for_duration)
        return True

    def _publish_turn_command(self):
        command = Float32MultiArray()
        command.data = [-0.5, 0.5]
        self.publisher.publish(command)

    def _publish_stop_command(self):
        command = Float32MultiArray()
        command.data = [0.0, 0.0]
        self.publisher.publish(command)
        self.stopped = True

    def _publish_for_duration(self):
        if time.monotonic() - self.started_at >= TURN_DURATION_SECONDS:
            self._publish_stop_command()
            self.finished = True
            self.get_logger().info('Turn complete; stop command published.')
            self.timer.cancel()
            return

        self._publish_turn_command()

    def destroy_node(self):
        if not self.stopped:
            self._publish_stop_command()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TurnRobotNode()
    completed = False
    try:
        if node.wait_for_motor_driver():
            while rclpy.ok() and not node.finished:
                rclpy.spin_once(node, timeout_sec=0.1)
            completed = node.finished
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0 if completed else 1


if __name__ == '__main__':
    main()
