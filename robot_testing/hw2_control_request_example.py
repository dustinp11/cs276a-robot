"""Service-based control example; hw2_control.py is left unchanged."""

import time

import rclpy
from geometry_msgs.msg import Twist
try:
    from hw2_interfaces.srv import GetRobotPose
except ImportError:
    GetRobotPose = None

from robot_testing.hw2_control import ControlNode


class RequestControlNode(ControlNode):
    def __init__(self):
        super().__init__()
        self.destroy_subscription(self.pose_sub)
        self.pose_client = self.create_client(GetRobotPose, '/get_robot_pose')
        self.declare_parameter('pose_request_timeout_s', 1.5)
        self.request_timeout_s = float(self.get_parameter('pose_request_timeout_s').value)
        if not 0 < self.request_timeout_s < float('inf'):
            raise ValueError('pose_request_timeout_s must be finite and positive')
        self.pose_future = None
        self.request_started = 0.0

    def control_tick(self):
        self.request_pose()
        super().control_tick()

    def request_pose(self):
        if self.pose_future is not None:
            if time.monotonic() - self.request_started > self.request_timeout_s:
                future = self.pose_future
                self.pose_future = None
                self.pose_client.remove_pending_request(future)
                future.cancel()
                self.latest_pose = None
            return
        if not self.pose_client.service_is_ready():
            self.latest_pose = None
            return
        self.request_started = time.monotonic()
        self.pose_future = self.pose_client.call_async(GetRobotPose.Request())
        self.pose_future.add_done_callback(self.on_pose_response)

    def on_pose_response(self, future):
        if future is not self.pose_future:
            return
        self.pose_future = None
        if future.cancelled():
            self.latest_pose = None
            return
        try:
            response = future.result()
        except Exception as exception:
            self.latest_pose = None
            self.get_logger().error(f'Pose request failed: {exception}')
            return
        if response.success:
            self.on_pose(response.pose)
        else:
            self.latest_pose = None
            self.get_logger().debug(response.message)


def main(args=None):
    rclpy.init(args=args)
    node = RequestControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.cmd_pub.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
