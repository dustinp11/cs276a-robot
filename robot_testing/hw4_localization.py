"""AprilTag Visual Localization for Homework 4.

Uses known global coordinates of all 8 wall AprilTags (36h11) and 4 obstacle
AprilTags (41h12) to compute the robot's global position (x, y) and heading (yaw)
in the map coordinate frame.

Reuses the proven LandmarkMap, PoseEstimator, PoseProvider, and math transforms
from robot_testing.hw2_localization while enhancing frame resolution for multiple
AprilTag families (36h11 and 41h12).
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import threading
import time
from typing import List, Optional, Tuple

import numpy as np

from robot_testing.hw2_localization import (
    Detection,
    Landmark,
    LandmarkMap,
    LocalizationError,
    Observation,
    PoseEstimate,
    PoseEstimator,
    PoseProvider,
    quaternion_transform,
)


def normalize_family(val: str) -> str:
    """Normalize AprilTag family strings (e.g. tagStandard41h12, Standard41h12 -> 41h12)."""
    if not isinstance(val, str) or not val:
        return ""
    fam = val[3:] if val.startswith('tag') else val
    if fam.startswith('Standard'):
        fam = fam[8:]
    return fam


class HW4PoseEstimator(PoseEstimator):
    """Extends PoseEstimator to distinguish wall vs obstacle tags and normalize frame names."""

    def __init__(self, landmark_map: LandmarkMap):
        super().__init__(landmark_map)
        self.wall_tag_ids: set = set()
        self.obstacle_tag_ids: set = set()
        for (family, tag_id), lm in list(landmark_map.landmarks.items()):
            norm_fam = normalize_family(family)
            if norm_fam == '36h11':
                self.wall_tag_ids.add(tag_id)
            elif norm_fam == '41h12':
                self.obstacle_tag_ids.add(tag_id)
            # Register family aliases in landmark dictionary so both 41h12 and Standard41h12 resolve
            landmark_map.landmarks[(norm_fam, tag_id)] = lm
            landmark_map.landmarks[(f"tag{norm_fam}", tag_id)] = lm
            landmark_map.landmarks[(f"Standard{norm_fam}", tag_id)] = lm
            landmark_map.landmarks[(f"tagStandard{norm_fam}", tag_id)] = lm

    def classify_tag(self, family: str, tag_id: int) -> str:
        fam = normalize_family(family)
        if fam == '36h11':
            return f"Wall Tag {tag_id} (36h11)"
        if fam == '41h12':
            return f"Obstacle Tag {tag_id} (41h12)"
        return f"Tag {tag_id} ({family})"

    def estimate(self, observation, lookup):
        cam_frame = observation.camera_frame.lstrip('/')
        map_cam_frame = self.map.camera_frame.lstrip('/')
        target_cam = self.map.camera_frame
        if cam_frame != map_cam_frame and 'camera' not in cam_frame:
            raise LocalizationError(
                f"Detection camera frame '{observation.camera_frame}' does not match map '{self.map.camera_frame}'"
            )

        candidates = []
        seen = set()
        rejections = []

        for item in observation.detections:
            fam = normalize_family(item.family)
            key = (fam, item.tag_id)
            landmark = self.map.landmarks.get(key)
            if landmark is None:
                rejections.append(f"Tag {key} not surveyed")
                continue
            if key in seen:
                continue
            seen.add(key)

            if not math.isfinite(item.margin) or item.margin < self.quality.min_decision_margin:
                rejections.append(f"Tag {item.tag_id} low margin ({item.margin:.1f} < {self.quality.min_decision_margin})")
                continue
            if not 0 <= item.hamming <= self.quality.max_hamming:
                rejections.append(f"Tag {item.tag_id} high hamming ({item.hamming})")
                continue

            try:
                camera_tag = lookup(landmark.frame, observation)
            except (LookupError, LocalizationError) as err:
                rejections.append(f"Tag {item.tag_id} TF error: {err}")
                continue

            distance = np.linalg.norm(camera_tag[:3, 3])
            if not 0 < distance <= self.quality.max_tag_distance_m or camera_tag[2, 3] <= 0:
                rejections.append(f"Tag {item.tag_id} bad depth/dist ({distance:.2f}m)")
                continue

            base = landmark.map_transform @ np.linalg.inv(camera_tag) @ self.map.camera_to_base
            tilt = math.acos(float(np.clip(base[2, 2], -1.0, 1.0)))
            height_err = abs(base[2, 3])

            if not np.isfinite(base).all():
                rejections.append(f"Tag {item.tag_id} non-finite transform")
                continue
            if height_err > self.quality.max_height_error_m:
                rejections.append(f"Tag {item.tag_id} height err ({height_err:.2f}m > {self.quality.max_height_error_m}m)")
                continue
            if tilt > math.radians(self.quality.max_tilt_deg):
                rejections.append(f"Tag {item.tag_id} tilt ({math.degrees(tilt):.1f}° > {self.quality.max_tilt_deg}°)")
                continue

            yaw = math.atan2(base[1, 0], base[0, 0])
            weight = max(1.0, min(item.margin, 100.0)) / (1.0 + distance * distance)
            candidates.append((landmark.tag_id, base[0, 3], base[1, 3], yaw, weight))

        if not candidates:
            detail = "; ".join(rejections) if rejections else "No detections passed quality gates"
            raise LocalizationError(f"No usable mapped tag with matching TF and valid geometry ({detail})")

        groups = [[other for other in candidates if self._agree(anchor, other)] for anchor in candidates]
        inliers = max(groups, key=len)
        if len(inliers) <= len(candidates) // 2:
            raise LocalizationError('Visible tags disagree about robot pose')

        values = np.array([candidate[1:] for candidate in inliers])
        weights = values[:, 3]
        x, y = np.average(values[:, :2], axis=0, weights=weights)
        sine = np.average(np.sin(values[:, 2]), weights=weights)
        cosine = np.average(np.cos(values[:, 2]), weights=weights)
        return PoseEstimate(
            float(x), float(y), math.atan2(sine, cosine),
            observation.stamp_ns, tuple(sorted(item[0] for item in inliers))
        )


class HW4LocalizationService:
    """ROS 2 Node service and publisher for HW4 visual localization."""

    def __init__(self, node):
        from ament_index_python.packages import get_package_share_directory
        from apriltag_msgs.msg import AprilTagDetectionArray
        from geometry_msgs.msg import PoseStamped
        try:
            from hw2_interfaces.srv import GetRobotPose
        except ImportError:
            GetRobotPose = None
        from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
        from rclpy.time import Time
        from tf2_ros import Buffer, TransformException, TransformListener

        self.node = node
        self.time_type = Time
        self.tf_exception = TransformException

        # Default configuration file: config/hw4_landmarks.json
        possible_configs = [
            Path(__file__).resolve().parent.parent / 'config/hw4_landmarks.json',
            Path('config/hw4_landmarks.json'),
        ]
        try:
            possible_configs.insert(
                0,
                Path(get_package_share_directory('robot_testing')) / 'config/hw4_landmarks.json'
            )
        except Exception:
            pass

        default_config = str(possible_configs[0])
        for p in possible_configs:
            if p.exists():
                default_config = str(p)
                break

        node.declare_parameter('landmarks_file', default_config)
        node.declare_parameter('max_pose_age_s', 0.60)
        node.declare_parameter('request_timeout_s', 1.0)

        landmarks_file = node.get_parameter('landmarks_file').value
        node.get_logger().info(f'Loading HW4 landmarks from: {landmarks_file}')
        self.landmark_map = LandmarkMap(landmarks_file)
        self.estimator = HW4PoseEstimator(self.landmark_map)

        self.tf_buffer = Buffer(node=node)
        self.tf_listener = TransformListener(self.tf_buffer, node)

        self.provider = PoseProvider(
            self.estimator,
            self.lookup,
            lambda: node.get_clock().now().nanoseconds,
            node.get_parameter('max_pose_age_s').value,
            node.get_parameter('request_timeout_s').value,
        )

        self.pose_pub = node.create_publisher(PoseStamped, '/robot_pose', 10)
        self.subscription = node.create_subscription(
            AprilTagDetectionArray, '/detections', self.on_detections, 10
        )

        if GetRobotPose is not None:
            self.service_group = MutuallyExclusiveCallbackGroup()
            self.service = node.create_service(
                GetRobotPose, '/get_robot_pose', self.on_request,
                callback_group=self.service_group,
            )
            node.get_logger().info('Ready: /get_robot_pose service and /robot_pose publisher')
        else:
            self.service = None
            node.get_logger().info('hw2_interfaces not found; publishing on /robot_pose topic.')

        self.publish_timer = node.create_timer(0.1, self.publish_pose_tick)
        self.last_logged_tags = None

    def on_detections(self, message):
        try:
            stamp = message.header.stamp
            detections = tuple(
                Detection(item.id, item.family, item.hamming, item.decision_margin)
                for item in message.detections
            )
            if len(detections) > 0:
                tag_summary = [f"{d.family}:{d.tag_id}" for d in detections]
                self.node.get_logger().info(
                    f"AprilTag detections received ({len(detections)}): {tag_summary}",
                    throttle_duration_sec=2.0,
                )
            self.provider.observe(Observation(
                stamp.sec * 10**9 + stamp.nanosec,
                message.header.frame_id,
                detections,
            ))
        except Exception as e:
            self.node.get_logger().warn(f"Error processing detections: {e}")

    def lookup(self, tag_frame, observation):
        """Lookup transform with multi-candidate and time-fallback for apriltag_ros frames."""
        import re
        exact_stamp = self.time_type(nanoseconds=observation.stamp_ns,
                                     clock_type=self.node.get_clock().clock_type)
        # Try exact observation timestamp first, then latest available Time()
        stamps_to_try = [exact_stamp, self.time_type()]

        match = re.search(r'\d+', tag_frame)
        tid = match.group(0) if match else tag_frame

        candidates = [
            f"tag36h11:{tid}",
            f"tagStandard41h12:{tid}",
            tag_frame,
            f"tag_{tid}",
            f"36h11:{tid}",
            f"Standard41h12:{tid}",
            f"tag41h12:{tid}",
            f"41h12:{tid}",
            f"tag_36h11_{tid}",
            f"tag_41h12_{tid}",
            f"{tid}",
        ]

        cam_frames_to_try = [
            observation.camera_frame,
            observation.camera_frame.lstrip('/'),
            'camera_frame',
        ]

        last_error = None
        for st in stamps_to_try:
            for cam in cam_frames_to_try:
                for frame in candidates:
                    try:
                        result = self.tf_buffer.lookup_transform(cam, frame, st)
                        t, q = result.transform.translation, result.transform.rotation
                        return quaternion_transform((t.x, t.y, t.z), (q.x, q.y, q.z, q.w))
                    except self.tf_exception as exception:
                        last_error = exception
                        continue

        raise LookupError(f"Transform failed for {tag_frame} (candidates: {candidates}): {last_error}")

    def create_pose_message(self, estimate: PoseEstimate):
        from geometry_msgs.msg import PoseStamped
        msg = PoseStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp.sec = estimate.stamp_ns // 10**9
        msg.header.stamp.nanosec = estimate.stamp_ns % 10**9
        msg.pose.position.x = float(estimate.x)
        msg.pose.position.y = float(estimate.y)
        msg.pose.position.z = 0.0
        msg.pose.orientation.z = math.sin(estimate.yaw / 2.0)
        msg.pose.orientation.w = math.cos(estimate.yaw / 2.0)
        return msg

    def publish_pose_tick(self):
        with self.provider._lock:
            observation = self.provider._latest
        if observation is None:
            return
        age = (self.provider.now_ns() - observation.stamp_ns) / 1e9
        if not (observation.stamp_ns > 0 and 0 <= age <= self.provider.max_age_s):
            return
        try:
            estimate = self.provider.estimator.estimate(observation, self.lookup)
            msg = self.create_pose_message(estimate)
            self.pose_pub.publish(msg)
            # Periodic logging on change of tag set or position
            if estimate.tag_ids != self.last_logged_tags:
                self.last_logged_tags = estimate.tag_ids
                tags_desc = [
                    self.estimator.classify_tag('36h11' if tid in self.estimator.wall_tag_ids else '41h12', tid)
                    for tid in estimate.tag_ids
                ]
                self.node.get_logger().info(
                    f"Visual Pose: ({estimate.x:.3f}, {estimate.y:.3f}) m, "
                    f"yaw={math.degrees(estimate.yaw):.1f}° from {tags_desc}"
                )
        except Exception as e:
            self.node.get_logger().warn(
                f"Localization update skipped: {e}",
                throttle_duration_sec=2.0
            )

    def on_request(self, request, response):
        try:
            estimate = self.provider.request()
        except LocalizationError as exception:
            response.success = False
            response.message = str(exception)
            self.node.get_logger().warn(f'HW4 Localization failed: {response.message}')
            return response
        response.success = True
        response.message = 'Pose estimated from current landmark observations'
        response.pose = self.create_pose_message(estimate)
        response.tag_ids = list(estimate.tag_ids)
        return response


def main(args=None):
    import rclpy
    from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
    from rclpy.node import Node

    rclpy.init(args=args)
    node = Node('hw4_localization')
    service = None
    executor = MultiThreadedExecutor(num_threads=3)
    try:
        service = HW4LocalizationService(node)
        executor.add_node(node)
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if service is not None:
            service.provider.close()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
