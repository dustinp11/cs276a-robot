"""ROS 2 Node for Extended Kalman Filter SLAM (EKF-SLAM).

Subscribes to:
- /visual_odometry/relative_motion (geometry_msgs/msg/TwistWithCovarianceStamped)
- /cmd_vel (geometry_msgs/msg/Twist) - fallback prediction
- /detections (apriltag_msgs/msg/AprilTagDetectionArray) - landmark measurements

Publishes:
- /ekf_slam/pose (geometry_msgs/msg/PoseWithCovarianceStamped)
- /ekf_slam/trajectory (nav_msgs/msg/Path)
- /ekf_slam/landmarks (visualization_msgs/msg/MarkerArray)
- /ekf_slam/status (std_msgs/msg/String)
- TF broadcast: map -> base_link
- Optional service /get_robot_pose (hw2_interfaces/srv/GetRobotPose)
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.time import Time
    from geometry_msgs.msg import Point, PoseStamped, PoseWithCovarianceStamped, Twist, TwistWithCovarianceStamped
    from nav_msgs.msg import Path as NavPath
    from std_msgs.msg import ColorRGBA, Header, String
    from visualization_msgs.msg import Marker, MarkerArray
    from apriltag_msgs.msg import AprilTagDetectionArray
    from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener
    from geometry_msgs.msg import TransformStamped
    try:
        from hw2_interfaces.srv import GetRobotPose
    except ImportError:
        GetRobotPose = None
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False
    Node = object

from hw3.coordinates import (
    CameraExtrinsics,
    make_transform,
    quaternion_to_rotation_matrix,
    wrap_angle,
)
from hw3.data_recorder import export_landmarks_csv, export_trajectory_csv
from hw3.ekf_slam import EkfSlam
from hw3.visual_odometry import VisualMotionEstimator
from hw3.vision_navigation import forward_wall_clearance, tag_normal_yaw


class EkfSlamNode(Node):
    """ROS 2 Node executing 2D EKF-SLAM and publishing visualization markers."""

    def __init__(self):
        """Initialize EKF state, ROS adapters, and publication/recording timers.

        Args:
            None.

        Returns:
            None. State changes and publications are performed in place.
        """
        if not HAS_ROS2:
            raise RuntimeError("ROS 2 packages required.")
        super().__init__('ekf_slam_node')
        self._declare_parameters()
        self._load_settings()
        self._create_extrinsics()
        self._create_estimator()
        self._create_vo_estimator()
        self._create_tf()
        self._create_subscriptions()
        self._create_publishers()
        self._create_pose_service()
        self._initialize_history()
        self._create_timers()

    def _declare_parameters(self):
        """Declare estimator, frame, covariance, and recording parameters.

        Args:
            None.

        Returns:
            None. State changes and publications are performed in place.
        """
        # Declare parameters
        self.declare_parameter('quality.min_decision_margin', 20.0)
        self.declare_parameter('quality.max_hamming', 1)
        self.declare_parameter('quality.min_tag_distance_m', 0.1)
        self.declare_parameter('quality.max_tag_distance_m', 5.0)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('camera_frame', 'camera_frame')

        self.declare_parameter('camera_mount.x_m', 0.0)
        self.declare_parameter('camera_mount.y_m', 0.0)
        self.declare_parameter('camera_mount.z_m', 0.2)
        self.declare_parameter('camera_mount.roll_deg', 0.0)
        self.declare_parameter('camera_mount.pitch_deg', 0.0)
        self.declare_parameter('camera_mount.yaw_deg', 0.0)

        self.declare_parameter('ekf_slam.motion_prediction_source', 'visual_odometry')
        self.declare_parameter('ekf_slam.use_cmd_vel_fallback', True)
        self.declare_parameter('ekf_slam.cmd_vel_timeout_s', 0.5)
        self.declare_parameter('ekf_slam.command_topic', '/cmd_vel')
        self.declare_parameter('ekf_slam.command_hold_s', 0.15)

        self.declare_parameter('ekf_slam.initial_covariance.sigma_x_m', 0.001)
        self.declare_parameter('ekf_slam.initial_covariance.sigma_y_m', 0.001)
        self.declare_parameter('ekf_slam.initial_covariance.sigma_theta_rad', 0.001)

        self.declare_parameter('ekf_slam.process_noise.vo_sigma_x_m', 0.03)
        self.declare_parameter('ekf_slam.process_noise.vo_sigma_y_m', 0.03)
        self.declare_parameter('ekf_slam.process_noise.vo_sigma_theta_rad', 0.04)
        self.declare_parameter('ekf_slam.process_noise.cmd_sigma_v_mps', 0.15)
        self.declare_parameter('ekf_slam.process_noise.cmd_sigma_lat_mps', 0.05)
        self.declare_parameter('ekf_slam.process_noise.cmd_sigma_omega_radps', 0.20)

        self.declare_parameter('ekf_slam.measurement_noise.sigma_range_m', 0.05)
        self.declare_parameter('ekf_slam.measurement_noise.sigma_bearing_rad', 0.05)
        self.declare_parameter('ekf_slam.measurement_noise.range_scale_dist_m', 2.0)
        self.declare_parameter('ekf_slam.mahalanobis_gate', 9.21)

        self.declare_parameter('visualization.publish_tf', True)
        self.declare_parameter('visualization.trajectory_history_len', 2000)
        self.declare_parameter('visualization.ellipse_sigma_scale', 2.0)

        self.declare_parameter('recording.default_output_dir', 'hw3_results')
        self.declare_parameter('recording.auto_export_csv', True)

    def _load_settings(self):
        """Load frame names and prediction-source settings.

        Args:
            None.

        Returns:
            None. State changes and publications are performed in place.
        """
        # Retrieve parameters
        self.map_frame = self.get_parameter('map_frame').value
        self.robot_frame = self.get_parameter('robot_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.publish_tf = self.get_parameter('visualization.publish_tf').value
        self.traj_history_len = self.get_parameter('visualization.trajectory_history_len').value
        self.ellipse_scale = self.get_parameter('visualization.ellipse_sigma_scale').value
        self.use_cmd_fallback = self.get_parameter('ekf_slam.use_cmd_vel_fallback').value
        self.cmd_vel_timeout_s = self.get_parameter('ekf_slam.cmd_vel_timeout_s').value

        self.command_prediction = self.get_parameter('ekf_slam.motion_prediction_source').value == 'cmd_vel'
        self.command_hold_s = self.get_parameter('ekf_slam.command_hold_s').value
        self.last_prediction_time = None

    def _create_extrinsics(self):
        """Construct the camera optical-to-base mount transform.

        Args:
            None.

        Returns:
            None. State changes and publications are performed in place.
        """
        # Build CameraExtrinsics
        self.extrinsics = CameraExtrinsics(
            x_m=self.get_parameter('camera_mount.x_m').value,
            y_m=self.get_parameter('camera_mount.y_m').value,
            z_m=self.get_parameter('camera_mount.z_m').value,
            roll_deg=self.get_parameter('camera_mount.roll_deg').value,
            pitch_deg=self.get_parameter('camera_mount.pitch_deg').value,
            yaw_deg=self.get_parameter('camera_mount.yaw_deg').value,
        )

    def _create_estimator(self):
        """Initialize EKF state and noise models from parameters.

        Args:
            None.

        Returns:
            None. State changes and publications are performed in place.
        """
        # Initialize EKF
        init_sigma = (
            self.get_parameter('ekf_slam.initial_covariance.sigma_x_m').value,
            self.get_parameter('ekf_slam.initial_covariance.sigma_y_m').value,
            self.get_parameter('ekf_slam.initial_covariance.sigma_theta_rad').value,
        )
        vo_noise = (
            self.get_parameter('ekf_slam.process_noise.vo_sigma_x_m').value,
            self.get_parameter('ekf_slam.process_noise.vo_sigma_y_m').value,
            self.get_parameter('ekf_slam.process_noise.vo_sigma_theta_rad').value,
        )
        cmd_noise = (
            self.get_parameter('ekf_slam.process_noise.cmd_sigma_v_mps').value,
            self.get_parameter('ekf_slam.process_noise.cmd_sigma_lat_mps').value,
            self.get_parameter('ekf_slam.process_noise.cmd_sigma_omega_radps').value,
        )
        meas_noise = (
            self.get_parameter('ekf_slam.measurement_noise.sigma_range_m').value,
            self.get_parameter('ekf_slam.measurement_noise.sigma_bearing_rad').value,
        )

        self.ekf = EkfSlam(
            initial_robot_pose=(0.0, 0.0, 0.0),
            initial_robot_sigma=init_sigma,
            vo_process_noise=vo_noise,
            cmd_process_noise=cmd_noise,
            base_measurement_noise=meas_noise,
            range_scale_dist_m=self.get_parameter('ekf_slam.measurement_noise.range_scale_dist_m').value,
            mahalanobis_gate=self.get_parameter('ekf_slam.mahalanobis_gate').value,
        )

    def _create_vo_estimator(self):
        """Create the internal frame-to-frame visual motion estimator.

        Args:
            None.

        Returns:
            None. State changes and publications are performed in place.
        """
        # Synchronous internal Visual Motion Estimator (ensures prediction precedes update)
        self.vo_estimator = VisualMotionEstimator(
            extrinsics=self.extrinsics,
            min_common_tags=1,
            max_dt_s=self.get_parameter('visual_odometry.max_dt_s').value if self.has_parameter('visual_odometry.max_dt_s') else 1.5,
            min_decision_margin=20.0,
            max_hamming=1,
            max_tag_distance_m=5.0,
            min_tag_distance_m=0.1,
            base_trans_noise_m=self.get_parameter('ekf_slam.process_noise.vo_sigma_x_m').value,
            base_rot_noise_rad=self.get_parameter('ekf_slam.process_noise.vo_sigma_theta_rad').value,
        )

    def _create_tf(self):
        """Create TF lookup and map-to-base publication interfaces."""
        # TF & Buffers
        self.tf_buffer = Buffer(node=self)
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

    def _create_subscriptions(self):
        """Subscribe to relative motion, commands, and tag observations.

        Args:
            None.

        Returns:
            None. State changes and publications are performed in place.
        """
        # Subscriptions
        self.vo_sub = self.create_subscription(
            TwistWithCovarianceStamped,
            '/visual_odometry/relative_motion',
            self.on_vo_motion,
            10
        )
        self.cmd_sub = self.create_subscription(
            Twist,
            self.get_parameter('ekf_slam.command_topic').value,
            self.on_cmd_vel,
            10
        )
        self.det_sub = self.create_subscription(
            AprilTagDetectionArray,
            '/detections',
            self.on_detections,
            10
        )

    def _create_publishers(self):
        """Create pose, trajectory, landmark, and status publishers.

        Args:
            None.

        Returns:
            None. State changes and publications are performed in place.
        """
        self.vision_pub = self.create_publisher(String, '/ekf_slam/vision_feedback', 10)
        # publishers
        self.pose_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            '/ekf_slam/pose',
            10
        )
        self.path_pub = self.create_publisher(
            NavPath,
            '/ekf_slam/trajectory',
            10
        )
        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/ekf_slam/landmarks',
            10
        )
        self.status_pub = self.create_publisher(
            String,
            '/ekf_slam/status',
            10
        )

    def _create_pose_service(self):
        """Expose the optional pose service when its interface is available.

        Args:
            None.

        Returns:
            None. State changes and publications are performed in place.
        """
        # Service /get_robot_pose compatible with HW2
        if GetRobotPose is not None:
            self.get_pose_srv = self.create_service(
                GetRobotPose,
                '/get_robot_pose',
                self.on_get_robot_pose_request
            )
            self.get_logger().info("Compatible /get_robot_pose service registered.")
        else:
            self.get_pose_srv = None

    def _initialize_history(self):
        """Initialize command times and empty trajectory histories."""
        # State tracking for fallback & trajectory history
        self.last_detection_stamp_s = -math.inf
        self.last_vo_time = 0.0
        self.last_cmd_time = 0.0
        self.last_cmd = (0.0, 0.0)
        self.trajectory_poses: List[PoseStamped] = []
        self.trajectory_records: List[dict] = []

    def _create_timers(self):
        """Create pose publication and optional CSV export timers."""
        # Periodic publish timer (10 Hz)
        self.pub_timer = self.create_timer(0.1, self.on_publish_timer)

        # Periodic auto-export timer (every 10 seconds)
        if self.get_parameter('recording.auto_export_csv').value:
            self.export_timer = self.create_timer(10.0, self.export_to_csv)

        self.get_logger().info("EKF-SLAM Node started successfully.")

    def on_vo_motion(self, msg: TwistWithCovarianceStamped):
        """Optional subscriber for external VO if internal vo_estimator is not active."""
        # When internal vo_estimator is used in on_detections, external on_vo_motion is ignored
        # to guarantee strictly synchronous prediction-before-update ordering.
        pass

    def on_cmd_vel(self, msg: Twist):
        """Integrate the previous held command before storing its replacement."""
        now_s = self.get_clock().now().nanoseconds * 1e-9
        self._predict_command_until(now_s)
        self.last_cmd = (msg.linear.x, msg.angular.z)
        self.last_cmd_time = now_s
        self.last_prediction_time = now_s

    def _predict_command_until(self, now_s):
        """Predict each held-command interval once, ending at the motor timeout.

        Args:
            now_s: Current ROS time in seconds; bounds the interval being integrated.

        Returns:
            None. State changes and publications are performed in place.
        """
        if self.last_prediction_time is None:
            return
        start = max(self.last_prediction_time, self.last_cmd_time)
        end = min(now_s, self.last_cmd_time + self.command_hold_s)
        self.last_prediction_time = now_s
        if not self.command_prediction:
            if not self.use_cmd_fallback:
                return
            start = max(start, self.last_vo_time + self.cmd_vel_timeout_s)
        if end > start:
            self.ekf.predict_commanded_velocity(*self.last_cmd, end - start)

    def on_detections(self, msg: AprilTagDetectionArray):
        """Apply one fresh image once, then report actual accepted visual constraints.

        Args:
            msg: Tag36h11 detections stamped in the camera optical frame.

        Returns:
            None. Updates EKF and publishes timestamped relative tag feedback.
        """
        stamp = msg.header.stamp
        stamp_s = stamp.sec + stamp.nanosec * 1e-9
        now_s = self.get_clock().now().nanoseconds * 1e-9
        if stamp_s <= self.last_detection_stamp_s or not -0.05 <= now_s - stamp_s <= 0.5:
            return
        self.last_detection_stamp_s = stamp_s
        detections = self._accepted_detections(msg)
        self._predict_observation(stamp_s, detections)
        known_ids = self._update_observed_landmarks(detections)
        self._publish_vision_feedback(stamp_s, detections, known_ids)

    def _accepted_detections(self, msg):
        """Filter quality and obtain only transforms at the original image time.

        Args:
            msg: Original stamped AprilTagDetectionArray.

        Returns:
            Detection dictionaries with finite optical-to-tag transforms.
        """
        stamp = msg.header.stamp
        time = Time(seconds=stamp.sec, nanoseconds=stamp.nanosec,
                    clock_type=self.get_clock().clock_type)
        frame = msg.header.frame_id or self.camera_frame
        records = []
        for det in msg.detections:
            if not self._detection_quality(det):
                continue
            transform = self._lookup_tag_transform(frame, det.id, det.family, time)
            if transform is not None and self._valid_tag_range(transform):
                records.append({'id': det.id, 'family': det.family, 't_cam_tag': transform,
                                'margin': det.decision_margin, 'hamming': det.hamming})
        return records

    def _detection_quality(self, detection):
        """Accept only the requested family and finite margin/hamming quality."""
        margin = float(detection.decision_margin)
        return (str(detection.family).removeprefix('tag') == '36h11'
                and math.isfinite(margin)
                and margin >= self.get_parameter('quality.min_decision_margin').value
                and 0 <= detection.hamming <= self.get_parameter('quality.max_hamming').value)

    def _valid_tag_range(self, transform):
        """Reject nonfinite, behind-camera, or out-of-range optical tag poses."""
        distance = np.linalg.norm(transform[:3, 3])
        return (np.all(np.isfinite(transform)) and transform[2, 3] > 0
                and self.get_parameter('quality.min_tag_distance_m').value <= distance
                <= self.get_parameter('quality.max_tag_distance_m').value)

    def _predict_observation(self, stamp_s, detections):
        """Use one prediction source per image; navigation retains a weak command model.

        Args:
            stamp_s: Original ROS image time in seconds.
            detections: Quality-filtered relative tag observations.

        Returns:
            None. Command prediction and tag-derived VO are never both integrated.
        """
        vo = self.vo_estimator.process_frame(stamp_s=stamp_s, detections=detections,
                                             stamp_ns=round(stamp_s * 1e9))
        if self.command_prediction:
            self._predict_command_until(self.get_clock().now().nanoseconds * 1e-9)
        elif vo.valid:
            self.ekf.predict_visual_odometry(vo.dx, vo.dy, vo.dtheta, vo.covariance)
            self.last_vo_time = stamp_s
        else:
            self._predict_command_until(self.get_clock().now().nanoseconds * 1e-9)
            if self.last_vo_time > 0:
                self.ekf.predict_untracked_drift(min(2.0, max(0.0, stamp_s - self.last_vo_time)))
            self.last_vo_time = stamp_s

    def _landmark_sigma(self, tag_id):
        """Return the largest robot-relative positional standard deviation in meters."""
        covariance = self.ekf.get_relative_landmark_covariance(tag_id)
        return math.sqrt(max(0.0, float(np.linalg.eigvalsh(covariance).max())))

    def _update_observed_landmarks(self, detections):
        """Distinguish map initialization from accepted reobservations of known tags.

        Args:
            detections: Filtered optical-frame tag observations.

        Returns:
            IDs successfully corrected that existed before this image's updates.
            Caches their pre-image robot-relative major-axis standard deviations in meters.
        """
        known_before = set(self.ekf.tag_id_to_idx)
        accepted = []
        self.frame_landmark_sigmas = {tag: self._landmark_sigma(tag) for tag in known_before}
        for det in detections:
            x, y, _ = self.extrinsics.landmark_in_base_2d(det['t_cam_tag'])
            if self.ekf.update_landmark(det['id'], x, y) and det['id'] in known_before:
                accepted.append(det['id'])
        return accepted

    def _publish_vision_feedback(self, stamp_s, detections, known_ids):
        """Report base-frame geometry independently of the pose publication timer.

        Args:
            stamp_s: Original accepted image time in ROS seconds.
            detections: Finite quality-filtered optical tag transforms.
            known_ids: Successfully reobserved mapped tag IDs.

        Returns:
            None. Publishes a JSON String; clearance is from robot center in meters.
        """
        tags, clearances, tag_yaws = [], [], []
        for det in detections:
            base_tag = self.extrinsics.camera_to_base_transform(det['t_cam_tag'])
            tags.append([int(det['id']), float(base_tag[0, 3]), float(base_tag[1, 3])])
            yaw = tag_normal_yaw(base_tag)
            if yaw is not None:
                tag_yaws.append([int(det['id']), yaw])
            clearance = forward_wall_clearance(base_tag)
            if clearance is not None:
                clearances.append(clearance)
        pose = self.ekf.get_robot_pose()
        sigmas = [[tag, self.frame_landmark_sigmas[tag]] for tag in known_ids
                  if tag in self.frame_landmark_sigmas]
        payload = {'landmark_sigmas': sigmas,
                   'ekf_heading': [pose.theta, math.sqrt(max(0.0, float(pose.covariance[2, 2])))],
                   'stamp_s': stamp_s, 'tags': tags, 'known_ids': known_ids, 'tag_yaws': tag_yaws,
                   'forward_clearance_m': min(clearances) if clearances else None}
        self.vision_pub.publish(String(data=json.dumps(payload, allow_nan=False)))

    def on_publish_timer(self):
        """Periodic publisher for pose, trajectory, landmarks, and TF.

        Args:
            None.

        Returns:
            None. State changes and publications are performed in place.
        """
        now = self.get_clock().now()
        self._predict_command_until(now.nanoseconds * 1e-9)
        pose_est = self.ekf.get_robot_pose()

        # 1. Publish PoseWithCovarianceStamped
        p_msg = PoseWithCovarianceStamped()
        p_msg.header.stamp = now.to_msg()
        p_msg.header.frame_id = self.map_frame

        p_msg.pose.pose.position.x = pose_est.x
        p_msg.pose.pose.position.y = pose_est.y
        p_msg.pose.pose.position.z = 0.0

        p_msg.pose.pose.orientation.z = math.sin(pose_est.theta * 0.5)
        p_msg.pose.pose.orientation.w = math.cos(pose_est.theta * 0.5)

        # Fill 6x6 covariance [x, y, z, roll, pitch, yaw]
        cov_6x6 = np.zeros((6, 6), dtype=float)
        cov_6x6[0:2, 0:2] = pose_est.covariance[0:2, 0:2]
        cov_6x6[5, 5] = pose_est.covariance[2, 2]
        cov_6x6[0:2, 5] = pose_est.covariance[0:2, 2]
        cov_6x6[5, 0:2] = pose_est.covariance[2, 0:2]
        p_msg.pose.covariance = cov_6x6.flatten().tolist()

        self.pose_pub.publish(p_msg)

        # 2. Append and publish trajectory Path
        stamped_pose = PoseStamped()
        stamped_pose.header = p_msg.header
        stamped_pose.pose = p_msg.pose.pose
        self.trajectory_poses.append(stamped_pose)
        if len(self.trajectory_poses) > self.traj_history_len:
            self.trajectory_poses.pop(0)

        path_msg = NavPath()
        path_msg.header = p_msg.header
        path_msg.poses = self.trajectory_poses
        self.path_pub.publish(path_msg)

        # Record for CSV export
        t_sec = now.nanoseconds * 1e-9
        self.trajectory_records.append({
            'timestamp_s': t_sec,
            'x': pose_est.x,
            'y': pose_est.y,
            'theta': pose_est.theta,
            'sigma_x': math.sqrt(max(0.0, pose_est.covariance[0, 0])),
            'sigma_y': math.sqrt(max(0.0, pose_est.covariance[1, 1])),
            'sigma_theta': math.sqrt(max(0.0, pose_est.covariance[2, 2])),
        })

        # 3. Publish TF (map -> base_link)
        if self.publish_tf:
            tf_msg = TransformStamped()
            tf_msg.header.stamp = now.to_msg()
            tf_msg.header.frame_id = self.map_frame
            tf_msg.child_frame_id = self.robot_frame
            tf_msg.transform.translation.x = pose_est.x
            tf_msg.transform.translation.y = pose_est.y
            tf_msg.transform.translation.z = 0.0
            tf_msg.transform.rotation = p_msg.pose.pose.orientation
            self.tf_broadcaster.sendTransform(tf_msg)

        # 4. Publish RViz Landmark Markers & Covariance Ellipses
        self._publish_landmark_markers(now.to_msg())

    def _publish_landmark_markers(self, stamp):
        """Create and publish MarkerArray for landmark positions, IDs, and ellipses.

        Args:
            stamp: ROS observation timestamp used for lookup or publication.

        Returns:
            None. State changes and publications are performed in place.
        """
        markers = []
        landmarks = self.ekf.get_all_landmarks()

        # Delete all previous markers
        del_marker = Marker()
        del_marker.action = Marker.DELETEALL
        markers.append(del_marker)

        marker_id = 0
        for tag_id, lm in landmarks.items():
            # 1. Sphere marker for landmark position
            m_pos = Marker()
            m_pos.header.stamp = stamp
            m_pos.header.frame_id = self.map_frame
            m_pos.ns = "landmark_positions"
            m_pos.id = marker_id
            marker_id += 1
            m_pos.type = Marker.SPHERE
            m_pos.action = Marker.ADD
            m_pos.pose.position.x = lm.x
            m_pos.pose.position.y = lm.y
            m_pos.pose.position.z = 0.15  # Tag height
            m_pos.pose.orientation.w = 1.0
            m_pos.scale.x = 0.12
            m_pos.scale.y = 0.12
            m_pos.scale.z = 0.12
            m_pos.color = ColorRGBA(r=0.1, g=0.8, b=0.2, a=0.9)  # Green
            markers.append(m_pos)

            # 2. Text marker for AprilTag ID
            m_text = Marker()
            m_text.header.stamp = stamp
            m_text.header.frame_id = self.map_frame
            m_text.ns = "landmark_labels"
            m_text.id = marker_id
            marker_id += 1
            m_text.type = Marker.TEXT_VIEW_FACING
            m_text.action = Marker.ADD
            m_text.pose.position.x = lm.x
            m_text.pose.position.y = lm.y
            m_text.pose.position.z = 0.30
            m_text.scale.z = 0.15
            m_text.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)  # White text
            m_text.text = f"Tag {tag_id}"
            markers.append(m_text)

            # 3. Covariance Ellipse (Line Strip)
            m_ellipse = self._create_covariance_ellipse_marker(
                stamp, marker_id, f"ellipse_{tag_id}", lm.x, lm.y, lm.covariance
            )
            marker_id += 1
            markers.append(m_ellipse)

        # 4. Robot pose covariance ellipse
        pose_est = self.ekf.get_robot_pose()
        m_robot_ellipse = self._create_covariance_ellipse_marker(
            stamp, marker_id, "robot_pose_ellipse",
            pose_est.x, pose_est.y, pose_est.covariance[:2, :2],
            color=ColorRGBA(r=1.0, g=0.2, b=0.2, a=0.8)  # Red
        )
        markers.append(m_robot_ellipse)

        marker_array = MarkerArray()
        marker_array.markers = markers
        self.marker_pub.publish(marker_array)

    def _create_covariance_ellipse_marker(
        self,
        stamp,
        marker_id: int,
        ns: str,
        cx: float,
        cy: float,
        cov_2x2: np.ndarray,
        color: Optional[ColorRGBA] = None,
    ) -> Marker:
        """Create a 2D confidence ellipse marker from a 2x2 covariance matrix.

        Args:
            stamp: ROS observation timestamp used for lookup or publication.
            marker_id: Integer marker identifier within its namespace.
            ns: Marker namespace used to identify this ellipse.
            cx: Ellipse center x coordinate in the map frame, in meters.
            cy: Ellipse center y coordinate in the map frame, in meters.
            cov_2x2: 2x2 position covariance ordered [x, y], in square meters.
            color: Optional ROS RGBA marker color; None uses the default landmark color.

        Returns:
            ROS Marker line strip in the configured map frame.
        """
        m = Marker()
        m.header.stamp = stamp
        m.header.frame_id = self.map_frame
        m.ns = ns
        m.id = marker_id
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.02  # Line thickness
        m.color = color if color is not None else ColorRGBA(r=0.2, g=0.6, b=1.0, a=0.8)  # Cyan/blue

        # Eigendecomposition of 2x2 covariance
        eigenvalues, eigenvectors = np.linalg.eigh(cov_2x2)
        # Sort descending
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        # Semi-major and semi-minor axes scaled by confidence factor (e.g. 2.0 = 95%)
        a = self.ellipse_scale * math.sqrt(max(1e-6, eigenvalues[0]))
        b = self.ellipse_scale * math.sqrt(max(1e-6, eigenvalues[1]))
        angle = math.atan2(eigenvectors[1, 0], eigenvectors[0, 0])

        # Generate ellipse points
        num_points = 32
        points = []
        for i in range(num_points + 1):
            theta = 2.0 * math.pi * i / num_points
            px = a * math.cos(theta)
            py = b * math.sin(theta)
            # Rotate by principal axis angle and translate to center
            rx = cx + px * math.cos(angle) - py * math.sin(angle)
            ry = cy + px * math.sin(angle) + py * math.cos(angle)
            pt = Point()
            pt.x = float(rx)
            pt.y = float(ry)
            pt.z = 0.05
            points.append(pt)

        m.points = points
        return m

    def _lookup_tag_transform(
        self,
        camera_frame: str,
        tag_id: int,
        family: str,
        stamp: Time,
    ) -> Optional[np.ndarray]:
        """Look up 4x4 transform T_cam_tag from TF.

        Args:
            camera_frame: Name of the camera optical frame used for tag pose lookup.
            tag_id: AprilTag integer ID used to associate the same stationary landmark.
            family: AprilTag family string used to resolve the detector TF frame name.
            stamp: ROS observation timestamp used for lookup or publication.

        Returns:
            4x4 tag-to-camera optical transform, or None when lookup fails.
        """
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

        return None

    def on_get_robot_pose_request(self, request, response):
        """Service callback for /get_robot_pose (HW2 controller compatibility).

        Args:
            request: Pose service request; no input fields are required.
            response: Pose service response to populate with the current estimate.

        Returns:
            The supplied response populated with current pose and known tag IDs.
        """
        pose_est = self.ekf.get_robot_pose()
        now = self.get_clock().now()

        msg = PoseStamped()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = self.map_frame
        msg.pose.position.x = pose_est.x
        msg.pose.position.y = pose_est.y
        msg.pose.position.z = 0.0
        msg.pose.orientation.z = math.sin(pose_est.theta * 0.5)
        msg.pose.orientation.w = math.cos(pose_est.theta * 0.5)

        response.success = True
        response.message = "Pose estimated by HW3 EKF-SLAM"
        response.pose = msg
        response.tag_ids = list(self.ekf.idx_to_tag_id)
        return response

    def export_to_csv(self):
        """Export current trajectory and landmark map to CSV files.

        Args:
            None.

        Returns:
            None. State changes and publications are performed in place.
        """
        out_dir = Path(self.get_parameter('recording.default_output_dir').value)
        out_dir.mkdir(parents=True, exist_ok=True)

        traj_file = out_dir / "ekf_slam_trajectory.csv"
        map_file = out_dir / "ekf_slam_landmarks.csv"

        if self.trajectory_records:
            export_trajectory_csv(self.trajectory_records, traj_file)

        lms = {}
        for tag_id, lm in self.ekf.get_all_landmarks().items():
            l_cov = lm.covariance
            lms[tag_id] = {
                'tag_id': tag_id,
                'x': lm.x,
                'y': lm.y,
                'sigma_x': math.sqrt(max(0.0, l_cov[0, 0])),
                'sigma_y': math.sqrt(max(0.0, l_cov[1, 1])),
                'cov_xy': float(l_cov[0, 1]),
            }
        if lms:
            export_landmarks_csv(lms, map_file)


def main(args=None):
    """Run this module through its command-line entry point.

    Args:
        args: Optional command-line arguments; None uses the process arguments.

    Returns:
        None. State changes and publications are performed in place.
    """
    if not HAS_ROS2:
        print("ROS 2 packages not available.")
        return
    rclpy.init(args=args)
    node = EkfSlamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.export_to_csv()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
