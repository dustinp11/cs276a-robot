"""On-demand AprilTag localization; configuration lives in config/landmarks.json."""

from dataclasses import dataclass
from itertools import combinations
import json
import math
from pathlib import Path
import threading
import time

import numpy as np


class LocalizationError(ValueError):
    pass


def finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LocalizationError(f'{name} must be a finite number')
    if not math.isfinite(value):
        raise LocalizationError(f'{name} must be a finite number')
    return float(value)


def rotation(roll, pitch, yaw):
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    return np.array([
        [cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
        [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
        [-sp, cp*sr, cp*cr],
    ])


def transform(orientation, position):
    result = np.eye(4)
    result[:3, :3] = orientation
    result[:3, 3] = position
    return result


def quaternion_transform(position, quaternion):
    q = np.asarray(quaternion, dtype=float)
    norm = np.linalg.norm(q)
    if not np.isfinite(q).all() or abs(norm - 1.0) > 0.05:
        raise LocalizationError('Invalid TF quaternion')
    x, y, z, w = q / norm
    matrix = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])
    if not np.isfinite(position).all():
        raise LocalizationError('Invalid TF translation')
    return transform(matrix, position)


def angle_distance(a, b):
    return abs(math.atan2(math.sin(a-b), math.cos(a-b)))


@dataclass(frozen=True)
class Landmark:
    tag_id: int
    family: str
    frame: str
    map_transform: np.ndarray


@dataclass(frozen=True)
class Quality:
    min_decision_margin: float = 20.0
    max_hamming: int = 1
    max_tag_distance_m: float = 5.0
    max_height_error_m: float = 0.2
    max_tilt_deg: float = 20.0
    max_disagreement_m: float = 0.25
    max_disagreement_deg: float = 20.0

    def __post_init__(self):
        for name, value in vars(self).items():
            number = finite_number(value, name)
            if number < 0 or (name not in ('min_decision_margin', 'max_hamming')
                              and number == 0):
                raise LocalizationError(f'{name} is outside its allowed range')
        if type(self.max_hamming) is not int or self.max_hamming > 3:
            raise LocalizationError('max_hamming must be an integer from 0 to 3')
        if self.max_tilt_deg > 90 or self.max_disagreement_deg > 180:
            raise LocalizationError('Invalid angular quality limit')


class LandmarkMap:
    def __init__(self, path):
        with Path(path).open(encoding='utf-8') as source:
            data = json.load(source)
        if data.get('surveyed') is not True:
            raise LocalizationError('Measure the landmarks and camera mount, then set surveyed=true')
        self.frame = self._frame(data['map_frame'])
        self.camera_frame = self._frame(data['camera_frame'])
        self.quality = Quality(**data.get('quality', {}))
        mount = data['camera_mount']
        xyz = [finite_number(mount[f'{axis}_m'], f'camera_mount.{axis}_m')
               for axis in ('x', 'y', 'z')]
        rpy = [math.radians(finite_number(mount[f'{axis}_deg'], axis))
               for axis in ('roll', 'pitch', 'yaw')]
        optical_to_body = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
        self.camera_to_base = np.linalg.inv(transform(rotation(*rpy) @ optical_to_body, xyz))
        self.landmarks = {}
        frames = set()
        for item in data['landmarks']:
            tag_id = item['id']
            if type(tag_id) is not int or not 0 <= tag_id < 2**31:
                raise LocalizationError('Landmark IDs must be nonnegative int32 values')
            family = self.family(item['family'])
            key = (family, tag_id)
            frame = self._frame(item['frame'])
            if key in self.landmarks or frame in frames:
                raise LocalizationError('Duplicate landmark ID/family or TF frame')
            xyz = [finite_number(item[f'{axis}_m'], f'tag {tag_id}.{axis}_m')
                   for axis in ('x', 'y', 'z')]
            facing = math.radians(finite_number(item['facing_deg'], 'facing_deg'))
            s, c = math.sin(facing), math.cos(facing)
            # Local PnP axes: printed right, printed up, outward face normal.
            axes = np.array([[-s, 0., c], [c, 0., s], [0., 1., 0.]])
            self.landmarks[key] = Landmark(tag_id, family, frame, transform(axes, xyz))
            frames.add(frame)
        if not self.landmarks:
            raise LocalizationError('The landmark map is empty')

    @staticmethod
    def _frame(value):
        if not isinstance(value, str) or not value or value.startswith('/'):
            raise LocalizationError('Frame names must be nonempty without a leading slash')
        return value

    @staticmethod
    def family(value):
        if not isinstance(value, str) or not value:
            raise LocalizationError('A landmark family is required')
        return value[3:] if value.startswith('tag') else value


@dataclass(frozen=True)
class Detection:
    tag_id: int
    family: str
    hamming: int
    margin: float


@dataclass(frozen=True)
class Observation:
    stamp_ns: int
    camera_frame: str
    detections: tuple


@dataclass(frozen=True)
class PoseEstimate:
    x: float
    y: float
    yaw: float
    stamp_ns: int
    tag_ids: tuple


class PoseEstimator:
    def __init__(self, landmark_map):
        self.map = landmark_map
        self.quality = landmark_map.quality

    def estimate(self, observation, lookup):
        if observation.camera_frame != self.map.camera_frame:
            raise LocalizationError('Detection camera frame does not match the map configuration')
        candidates = []
        seen = set()
        for detection in observation.detections:
            key = (LandmarkMap.family(detection.family), detection.tag_id)
            landmark = self.map.landmarks.get(key)
            if (landmark is None or key in seen
                    or not math.isfinite(detection.margin)
                    or detection.margin < self.quality.min_decision_margin
                    or not 0 <= detection.hamming <= self.quality.max_hamming):
                continue
            seen.add(key)
            try:
                camera_tag = lookup(landmark.frame, observation)
            except (LookupError, LocalizationError):
                continue
            distance = np.linalg.norm(camera_tag[:3, 3])
            if not 0 < distance <= self.quality.max_tag_distance_m or camera_tag[2, 3] <= 0:
                continue
            base = landmark.map_transform @ np.linalg.inv(camera_tag) @ self.map.camera_to_base
            tilt = math.acos(float(np.clip(base[2, 2], -1., 1.)))
            if (not np.isfinite(base).all()
                    or abs(base[2, 3]) > self.quality.max_height_error_m
                    or tilt > math.radians(self.quality.max_tilt_deg)):
                continue
            yaw = math.atan2(base[1, 0], base[0, 0])
            weight = max(1., min(detection.margin, 100.)) / (1. + distance*distance)
            candidates.append((landmark.tag_id, base[0, 3], base[1, 3], yaw, weight))
        if not candidates:
            raise LocalizationError('No usable mapped tag with matching TF and valid geometry')
        groups = [[other for other in candidates if self._agree(anchor, other)]
                  for anchor in candidates]
        inliers = max(groups, key=len)
        if (len(inliers) <= len(candidates) // 2
                or any(not self._agree(a, b) for a, b in combinations(inliers, 2))):
            raise LocalizationError('Visible tags disagree about the robot pose')
        values = np.array([candidate[1:] for candidate in inliers])
        weights = values[:, 3]
        x, y = np.average(values[:, :2], axis=0, weights=weights)
        sine = np.average(np.sin(values[:, 2]), weights=weights)
        cosine = np.average(np.cos(values[:, 2]), weights=weights)
        return PoseEstimate(float(x), float(y), math.atan2(sine, cosine),
                            observation.stamp_ns, tuple(sorted(item[0] for item in inliers)))

    def _agree(self, a, b):
        return (math.hypot(a[1]-b[1], a[2]-b[2]) <= self.quality.max_disagreement_m
                and angle_distance(a[3], b[3]) <= math.radians(self.quality.max_disagreement_deg))


class PoseProvider:
    def __init__(self, estimator, lookup, now_ns, max_age_s=0.5, timeout_s=1.0):
        self.estimator = estimator
        self.lookup = lookup
        self.now_ns = now_ns
        self.max_age_s = finite_number(max_age_s, 'max_pose_age_s')
        self.timeout_s = finite_number(timeout_s, 'request_timeout_s')
        if self.max_age_s <= 0 or self.timeout_s < 0:
            raise LocalizationError('Pose age must be positive and request timeout nonnegative')
        self._lock = threading.Lock()
        self._latest = None
        self._stopped = threading.Event()

    def observe(self, observation):
        with self._lock:
            old = self._latest
            if (old is None or old.stamp_ns > self.now_ns()
                    or observation.stamp_ns >= old.stamp_ns):
                self._latest = observation

    def close(self):
        self._stopped.set()

    def request(self):
        deadline = time.monotonic() + self.timeout_s
        error = 'No detection frame received'
        while not self._stopped.is_set():
            with self._lock:
                observation = self._latest
            if observation is not None:
                age = (self.now_ns() - observation.stamp_ns) / 1e9
                if observation.stamp_ns > 0 and 0 <= age <= self.max_age_s:
                    try:
                        pose = self.estimator.estimate(observation, self.lookup)
                        if 0 <= (self.now_ns() - pose.stamp_ns) / 1e9 <= self.max_age_s:
                            return pose
                        error = 'Observation expired during estimation'
                    except LocalizationError as exception:
                        error = str(exception)
                else:
                    error = 'Latest detection frame is stale or future-dated'
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise LocalizationError(error)
            self._stopped.wait(min(0.01, remaining))
        raise LocalizationError('Localization is shutting down')


class LocalizationService:
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
        default_map = Path(get_package_share_directory('robot_testing')) / 'config/landmarks.json'
        node.declare_parameter('landmarks_file', str(default_map))
        node.declare_parameter('max_pose_age_s', 0.5)
        node.declare_parameter('request_timeout_s', 1.0)
        landmark_map = LandmarkMap(node.get_parameter('landmarks_file').value)
        if landmark_map.frame != 'map':
            raise LocalizationError('The control interface requires map_frame="map"')
        self.tf_buffer = Buffer(node=node)
        self.tf_listener = TransformListener(self.tf_buffer, node)
        self.provider = PoseProvider(
            PoseEstimator(landmark_map), self.lookup,
            lambda: node.get_clock().now().nanoseconds,
            node.get_parameter('max_pose_age_s').value,
            node.get_parameter('request_timeout_s').value,
        )
        self.pose_pub = node.create_publisher(PoseStamped, '/robot_pose', 10)
        self.subscription = node.create_subscription(
            AprilTagDetectionArray, '/detections', self.on_detections, 1
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

    def on_detections(self, message):
        stamp = message.header.stamp
        detections = tuple(Detection(item.id, item.family, item.hamming, item.decision_margin)
                           for item in message.detections)
        self.provider.observe(Observation(stamp.sec*10**9 + stamp.nanosec,
                                          message.header.frame_id, detections))

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
        except Exception:
            pass

    def create_pose_message(self, estimate):
        from geometry_msgs.msg import PoseStamped
        msg = PoseStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp.sec = estimate.stamp_ns // 10**9
        msg.header.stamp.nanosec = estimate.stamp_ns % 10**9
        msg.pose.position.x = float(estimate.x)
        msg.pose.position.y = float(estimate.y)
        msg.pose.position.z = 0.0
        msg.pose.orientation.z = math.sin(estimate.yaw / 2.)
        msg.pose.orientation.w = math.cos(estimate.yaw / 2.)
        return msg

    def lookup(self, tag_frame, observation):
        stamp = self.time_type(nanoseconds=observation.stamp_ns,
                               clock_type=self.node.get_clock().clock_type)
        try:
            result = self.tf_buffer.lookup_transform(observation.camera_frame, tag_frame, stamp)
        except self.tf_exception as exception:
            raise LookupError(str(exception)) from exception
        t, q = result.transform.translation, result.transform.rotation
        return quaternion_transform((t.x, t.y, t.z), (q.x, q.y, q.z, q.w))

    def on_request(self, request, response):
        try:
            estimate = self.provider.request()
        except LocalizationError as exception:
            response.success = False
            response.message = str(exception)
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
    node = Node('hw2_localization')
    localization = None
    executor = MultiThreadedExecutor(num_threads=3)
    try:
        localization = LocalizationService(node)
        executor.add_node(node)
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if localization is not None:
            localization.provider.close()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
