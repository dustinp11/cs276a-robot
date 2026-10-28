"""Partner's file: map-frame robot pose + waypoints -> velocity commands.

Contract and run commands: ../HW2_TEAM_INTERFACE.md.
This skeleton intentionally commands zero velocity until compute_command works.
"""

from math import atan2, isfinite, pi

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node


WAYPOINTS = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
             (1.0, 2.0, pi), (0.0, 0.0, 0.0))


def yaw_from_pose(pose):
    """Read planar yaw in radians from a PoseStamped unit quaternion."""
    q = pose.pose.orientation
    return atan2(2.0 * (q.w * q.z + q.x * q.y),
                 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class ControlNode(Node):
    def __init__(self):
        super().__init__('hw2_control')
        self.declare_parameter('max_pose_age_s', 0.5)
        self.max_pose_age_s = float(self.get_parameter('max_pose_age_s').value)
        if not isfinite(self.max_pose_age_s) or self.max_pose_age_s <= 0:
            raise ValueError('max_pose_age_s must be finite and positive')
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 1)
        self.pose_sub = self.create_subscription(
            PoseStamped, '/robot_pose', self.on_pose, 1
        )
        self.latest_pose = None
        self.waypoint_index = 0
        self.last_tick = self.get_clock().now()
        self.timer = self.create_timer(0.1, self.control_tick)
        # TODO: Add stage, gains, tolerances, dwell and timeout state here.

    def on_pose(self, pose):
        p, q = pose.pose.position, pose.pose.orientation
        values = (p.x, p.y, p.z, q.x, q.y, q.z, q.w)
        norm_squared = q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w
        if (pose.header.frame_id != 'map'
                or not all(isfinite(value) for value in values)
                or abs(norm_squared - 1.0) > 0.01):
            self.latest_pose = None
            return
        stamp = pose.header.stamp
        age = (self.get_clock().now().nanoseconds / 1e9
               - (stamp.sec + stamp.nanosec / 1e9))
        if not 0.0 <= age <= self.max_pose_age_s:
            self.latest_pose = None
            return
        # Ignore reordered observations; do not let one replace a newer pose.
        if self.latest_pose is not None:
            old = self.latest_pose.header.stamp
            new = pose.header.stamp
            if (new.sec, new.nanosec) <= (old.sec, old.nanosec):
                return
        self.latest_pose = pose

    def control_tick(self):
        now = self.get_clock().now()
        dt = (now - self.last_tick).nanoseconds / 1e9
        self.last_tick = now
        command = Twist()  # Default to a stop for every missing/invalid case.
        pose = self.latest_pose
        if pose is not None and dt > 0 and self.waypoint_index < len(WAYPOINTS):
            stamp = pose.header.stamp
            age = now.nanoseconds / 1e9 - (stamp.sec + stamp.nanosec / 1e9)
            if 0.0 <= age <= self.max_pose_age_s:
                goal = WAYPOINTS[self.waypoint_index]
                command, reached = self.compute_command(pose, goal, dt)
                if reached:
                    self.waypoint_index += 1
                    command = Twist()  # Stop between goals and after the last.
        self.cmd_pub.publish(command)

    def compute_command(self, pose, goal, dt):
        """Return (Twist, reached) for one control update.

        INPUT: fresh map-frame PoseStamped, goal=(x_m, y_m, yaw_rad), dt_seconds.
        OUTPUT: Twist.linear.x in m/s; Twist.angular.z in rad/s; all other
        components zero. reached=True only after position AND yaw settle.

        TODO: Implement ALIGN -> DRIVE -> FINAL_YAW with dwell and timeouts.
        Use pose.pose.position.x/y and yaw_from_pose(pose); wrap angle errors.
        Correct heading during DRIVE, limit velocities/acceleration, and handle
        already-at-goal positions before computing a bearing with atan2.
        Reset dwell/integral state after stale data; do not count blind time
        toward waypoint acceptance. This skeleton does not predict across gaps.
        """
        return Twist(), False


def main(args=None):
    rclpy.init(args=args)
    node = ControlNode()
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
