
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu


class YawTestNode(Node):
    def __init__(self):
        super().__init__("yaw_test")

        self._initial_yaw: float | None = None

        self._imu_sub = self.create_subscription(
            Imu,
            "imu_data",
            self._imu_callback,
            10,
        )

        self.get_logger().info(
            "Listening for IMU data. Rotate the robot and watch the heading."
        )

    def _imu_callback(self, msg: Imu) -> None:
        # Same quaternion → yaw formula as robot_test.py
        qw = msg.orientation.w
        qx = msg.orientation.x
        qy = msg.orientation.y
        qz = msg.orientation.z

        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)   # raw yaw in (-π, π]

        # Lock the reference on the first reading
        if self._initial_yaw is None:
            self._initial_yaw = yaw
            self.get_logger().info(
                f"Reference heading locked: {math.degrees(yaw):.2f}°  "
                f"(all future readings are relative to this)"
            )

        # Relative angle, wrapped to [0, 360)
        relative_deg = math.degrees((yaw - self._initial_yaw) % (2 * math.pi))

        print(f"Heading: {relative_deg:6.1f}°", flush=True)


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = YawTestNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
