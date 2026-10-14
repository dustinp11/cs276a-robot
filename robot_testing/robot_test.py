import math
import time
# ##### ADDED: orientation log file path #####
from pathlib import Path
# ##### END ADDED CODE #####

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Quaternion


class RobotTestNode(Node):
    def __init__(self):
        super().__init__('robot_test')
        self.publisher = self.create_publisher(Float32MultiArray, 'motor_commands', 10)
        # ##### ADDED: subscribe to IMU orientation #####
        self.orientation_subscription = self.create_subscription(
            Quaternion,
            'imu_orientation',
            self.orientation_callback,
            10
        )
        self.latest_orientation = None
        # ##### END ADDED CODE #####

        # ##### ADDED: open an append-only file for IMU orientation #####
        self.orientation_log_path = Path.cwd() / 'imu_yaw.txt'
        self.orientation_log = self.orientation_log_path.open('a', encoding='utf-8')
        self.get_logger().info(f'Writing IMU orientation to {self.orientation_log_path}')
        # ##### END ADDED CODE #####

        self.started_at = time.monotonic()
        self.warned_no_subscriber = False
        self.timer = self.create_timer(0.05, self.publish_motor_command)
        self.get_logger().info('Moving forward at [0.3, 0.3] for 1 second, then stopping')

    def publish_motor_command(self):
        now = time.monotonic()
        if self.publisher.get_subscription_count() == 0 and not self.warned_no_subscriber:
            self.get_logger().warning(
                'No motor_control subscriber on motor_commands; commands are being published '
                'but will not move the robot until motor_control is running'
            )
            self.warned_no_subscriber = True

        elapsed = now - self.started_at
        command = [0.5, -0.5] if elapsed < 1 else [0.0, 0.0]

        message = Float32MultiArray()
        message.data = command
        self.publisher.publish(message)

    # ##### ADDED: receive IMU orientation #####
    def orientation_callback(self, msg):
        self.latest_orientation = msg
        # ##### ADDED: write each received quaternion to the text file #####
        stamp = self.get_clock().now().to_msg()
        try:
            yaw = math.degrees(math.atan2(
                2.0 * (msg.w * msg.z + msg.x * msg.y),
                1.0 - 2.0 * (msg.y * msg.y + msg.z * msg.z)
            ))
            self.orientation_log.write(
                f'{stamp.sec}.{stamp.nanosec:09d} '
                f'yaw={yaw:.2f} degrees\n'
            )
            self.orientation_log.flush()
        except OSError as error:
            self.get_logger().error(f'Could not write IMU orientation log: {error}')
        # ##### END ADDED CODE #####
    # ##### END ADDED CODE #####

    # ##### ADDED: close the orientation file on node shutdown #####
    def destroy_node(self):
        self.orientation_log.close()
        super().destroy_node()
    # ##### END ADDED CODE #####


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
