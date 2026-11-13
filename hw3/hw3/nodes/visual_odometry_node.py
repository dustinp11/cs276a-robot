"""ROS 2 Node for Visual Motion Estimation using AprilTags.

Subscribes to:
- /detections (apriltag_msgs/msg/AprilTagDetectionArray)
- TF2 transforms between camera_frame and detected tags

Publishes:
- /visual_odometry/relative_motion (geometry_msgs/msg/TwistWithCovarianceStamped)
- /visual_odometry/odom (nav_msgs/msg/Odometry)
- /visual_odometry/status (std_msgs/msg/String)
- /visual_odometry/active (std_msgs/msg/Bool)
"""

import math
from typing import Optional
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.time import Time
    from geometry_msgs.msg import TwistWithCovarianceStamped
    from nav_msgs.msg import Odometry
    from std_msgs.msg import Bool, String
    from apriltag_msgs.msg import AprilTagDetectionArray
    from tf2_ros import Buffer, TransformException, TransformListener
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False
    Node = object

from hw3.coordinates import (
    CameraExtrinsics,
    make_transform,
    quaternion_to_rotation_matrix,
)
from hw3.visual_odometry import VisualMotionEstimator


class VisualOdometryNode(Node):
    """ROS 2 Node estimating inter-frame robot motion from AprilTag observations."""

    def __init__(self):
        if not HAS_ROS2:
            raise RuntimeError("ROS 2 Python packages (rclpy, apriltag_msgs, tf2_ros) are required.")
        super().__init__('visual_odometry_node')

        # Declare parameters
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('camera_frame', 'camera_frame')
        self.declare_parameter('camera_mount.x_m', 0.0)
        self.declare_parameter('camera_mount.y_m', 0.0)
        self.declare_parameter('camera_mount.z_m', 0.2)
        self.declare_parameter('camera_mount.roll_deg', 0.0)
        self.declare_parameter('camera_mount.pitch_deg', 0.0)
        self.declare_parameter('camera_mount.yaw_deg', 0.0)

        self.declare_parameter('visual_odometry.min_common_tags', 1)
        self.declare_parameter('visual_odometry.max_dt_s', 0.75)
        self.declare_parameter('visual_odometry.base_trans_noise_m', 0.02)
        self.declare_parameter('visual_odometry.base_rot_noise_rad', 0.03)

        self.declare_parameter('quality.min_decision_margin', 20.0)
        self.declare_parameter('quality.max_hamming', 1)
        self.declare_parameter('quality.max_tag_distance_m', 5.0)
        self.declare_parameter('quality.min_tag_distance_m', 0.1)

        # Build CameraExtrinsics
        extrinsics = CameraExtrinsics(
            x_m=self.get_parameter('camera_mount.x_m').value,
            y_m=self.get_parameter('camera_mount.y_m').value,
            z_m=self.get_parameter('camera_mount.z_m').value,
            roll_deg=self.get_parameter('camera_mount.roll_deg').value,
            pitch_deg=self.get_parameter('camera_mount.pitch_deg').value,
            yaw_deg=self.get_parameter('camera_mount.yaw_deg').value,
        )

        # Initialize Visual Motion Estimator
        self.estimator = VisualMotionEstimator(
            extrinsics=extrinsics,
            min_common_tags=self.get_parameter('visual_odometry.min_common_tags').value,
            max_dt_s=self.get_parameter('visual_odometry.max_dt_s').value,
            min_decision_margin=self.get_parameter('quality.min_decision_margin').value,
            max_hamming=self.get_parameter('quality.max_hamming').value,
            max_tag_distance_m=self.get_parameter('quality.max_tag_distance_m').value,
            min_tag_distance_m=self.get_parameter('quality.min_tag_distance_m').value,
            base_trans_noise_m=self.get_parameter('visual_odometry.base_trans_noise_m').value,
            base_rot_noise_rad=self.get_parameter('visual_odometry.base_rot_noise_rad').value,
        )

        self.robot_frame = self.get_parameter('robot_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value

        # TF Buffer and Listener
        self.tf_buffer = Buffer(node=self)
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscribers and Publishers
        self.detection_sub = self.create_subscription(
            AprilTagDetectionArray,
            '/detections',
            self.on_detections,
            10
        )

        self.twist_pub = self.create_publisher(
            TwistWithCovarianceStamped,
            '/visual_odometry/relative_motion',
            10
        )
        self.odom_pub = self.create_publisher(
            Odometry,
            '/visual_odometry/odom',
            10
        )
        self.status_pub = self.create_publisher(
            String,
            '/visual_odometry/status',
            10
        )
        self.active_pub = self.create_publisher(
            Bool,
            '/visual_odometry/active',
            10
        )

        # Accumulated VO pose for odom publishing
        self.accum_x = 0.0
        self.accum_y = 0.0
        self.accum_theta = 0.0

        self.get_logger().info(
            f"Visual Odometry Node initialized. Listening on /detections (camera frame: {self.camera_frame})"
        )

    def on_detections(self, msg: AprilTagDetectionArray):
        """Callback for incoming AprilTag detections."""
        stamp = msg.header.stamp
        stamp_s = stamp.sec + stamp.nanosec * 1e-9
        stamp_ns = stamp.sec * 10**9 + stamp.nanosec
        cam_frame = msg.header.frame_id if msg.header.frame_id else self.camera_frame

        ros_time = Time(seconds=stamp.sec, nanoseconds=stamp.nanosec, clock_type=self.get_clock().clock_type)

        processed_detections = []
        for det in msg.detections:
            tag_id = det.id
            family = det.family

            # Look up tag pose in TF buffer
            t_cam_tag = self._lookup_tag_transform(cam_frame, tag_id, family, ros_time)
            if t_cam_tag is None:
                continue

            processed_detections.append({
                'id': tag_id,
                'family': family,
                't_cam_tag': t_cam_tag,
                'margin': det.decision_margin,
                'hamming': det.hamming,
            })

        # Process frame through estimator
        vo_est = self.estimator.process_frame(
            stamp_s=stamp_s,
            detections=processed_detections,
            stamp_ns=stamp_ns,
        )

        # Publish status
        status_msg = String()
        status_msg.data = vo_est.status_message
        self.status_pub.publish(status_msg)

        active_msg = Bool()
        active_msg.data = vo_est.valid
        self.active_pub.publish(active_msg)

        if not vo_est.valid or vo_est.dt_s <= 0.0:
            return

        # Relative velocities
        vx = vo_est.dx / vo_est.dt_s
        vy = vo_est.dy / vo_est.dt_s
        omega = vo_est.dtheta / vo_est.dt_s

        # Publish TwistWithCovarianceStamped
        twist_msg = TwistWithCovarianceStamped()
        twist_msg.header.stamp = stamp
        twist_msg.header.frame_id = self.robot_frame

        twist_msg.twist.twist.linear.x = float(vx)
        twist_msg.twist.twist.linear.y = float(vy)
        twist_msg.twist.twist.linear.z = 0.0
        twist_msg.twist.twist.angular.x = 0.0
        twist_msg.twist.twist.angular.y = 0.0
        twist_msg.twist.twist.angular.z = float(omega)

        # 6x6 Covariance mapping [x, y, z, roll, pitch, yaw]
        cov_6x6 = np.zeros((6, 6), dtype=float)
        # Scaled to velocity rates
        dt2 = vo_est.dt_s ** 2
        cov_6x6[0, 0] = vo_est.covariance[0, 0] / dt2
        cov_6x6[1, 1] = vo_est.covariance[1, 1] / dt2
        cov_6x6[5, 5] = vo_est.covariance[2, 2] / dt2
        cov_6x6[0, 1] = vo_est.covariance[0, 1] / dt2
        cov_6x6[1, 0] = vo_est.covariance[1, 0] / dt2
        twist_msg.twist.covariance = cov_6x6.flatten().tolist()

        self.twist_pub.publish(twist_msg)

        # Update accumulated VO pose
        c = math.cos(self.accum_theta)
        s = math.sin(self.accum_theta)
        self.accum_x += vo_est.dx * c - vo_est.dy * s
        self.accum_y += vo_est.dx * s + vo_est.dy * c
        self.accum_theta = (self.accum_theta + vo_est.dtheta + math.pi) % (2.0 * math.pi) - math.pi

        # Publish Odometry
        odom_msg = Odometry()
        odom_msg.header.stamp = stamp
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = self.robot_frame

        odom_msg.pose.pose.position.x = float(self.accum_x)
        odom_msg.pose.pose.position.y = float(self.accum_y)
        odom_msg.pose.pose.position.z = 0.0
        odom_msg.pose.pose.orientation.z = math.sin(self.accum_theta * 0.5)
        odom_msg.pose.pose.orientation.w = math.cos(self.accum_theta * 0.5)

        odom_msg.twist = twist_msg.twist
        self.odom_pub.publish(odom_msg)

    def _lookup_tag_transform(
        self,
        camera_frame: str,
        tag_id: int,
        family: str,
        stamp: Time,
    ) -> Optional[np.ndarray]:
        """Look up 4x4 transform T_cam_tag from TF."""
        # Try configured frame name 'tag_{id}', then fallback to '{family}:{id}'
        candidate_frames = [f"tag_{tag_id}", f"{family}:{tag_id}"]
        for tag_frame in candidate_frames:
            try:
                tf_res = self.tf_buffer.lookup_transform(camera_frame, tag_frame, stamp)
                t = tf_res.transform.translation
                q = tf_res.transform.rotation
                rot = quaternion_to_rotation_matrix(q.x, q.y, q.z, q.w)
                pos = [t.x, t.y, t.z]
                return make_transform(rot, pos)
            except TransformException:
                continue

        # If exact timestamp failed, try latest available transform
        for tag_frame in candidate_frames:
            try:
                tf_res = self.tf_buffer.lookup_transform(camera_frame, tag_frame, Time())
                t = tf_res.transform.translation
                q = tf_res.transform.rotation
                rot = quaternion_to_rotation_matrix(q.x, q.y, q.z, q.w)
                pos = [t.x, t.y, t.z]
                return make_transform(rot, pos)
            except TransformException:
                continue

        return None


def main(args=None):
    if not HAS_ROS2:
        print("ROS 2 packages not available.")
        return
    rclpy.init(args=args)
    node = VisualOdometryNode()
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
