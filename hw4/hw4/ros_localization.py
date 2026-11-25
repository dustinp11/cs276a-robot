"""ROS detection and timestamp-matched TF adapter for known-map localization."""

from collections import deque
from functools import partial
import math
import json
import time
import numpy as np
from apriltag_msgs.msg import AprilTagDetectionArray
from geometry_msgs.msg import PoseStamped
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener

from hw3.coordinates import make_transform, quaternion_to_rotation_matrix
from hw4.localization import Localizer
from hw4.observations import CaptureBuffer
from hw4.tags import canonical_family, detection_topic, mapped_families, tag_frame


class PoseSource:
    """Cache distinct measured poses; never substitute the latest TF or a prediction."""

    def __init__(self, node, environment, settings, log):
        """Subscribe to detector frames while retaining their original capture time."""
        # preserve image identity across asynchronous topics
        self.node, self.log = node, log
        self.model = Localizer(environment, settings['camera_mount'], settings['localization'])
        self.policy = settings['localization']
        self.max_age = settings['navigation']['pose_timeout_s']
        self.samples, self.frames = deque(maxlen=30), CaptureBuffer()
        self.families = mapped_families(environment)
        self.setup_goal_reference(settings['navigation'])
        self.buffer = Buffer(node=node)
        self.listener = TransformListener(self.buffer, node)
        self.subscriptions = [node.create_subscription(
            AprilTagDetectionArray, detection_topic(family),
            partial(self.observe, family), 1) for family in self.families]
        self.publisher = node.create_publisher(PoseStamped, '/hw4/pose', 10)

    def setup_goal_reference(self, policy):
        """Keep reference-tag feedback separate from multi-tag consensus."""
        # record a single reference without weakening consensus
        self.goal_tag = canonical_family(policy['goal_reference_family']), policy['goal_reference_id']
        if self.goal_tag not in self.model.landmarks:
            raise ValueError('Configured goal reference tag is not in the measured map.')
        self.goal_samples = deque(maxlen=30)
        self.goal_log = self.log.open_csv('goal_observations.csv', [
            'time_s', 'image_stamp_ns', 'accepted', 'reason', 'tag', 'x_m', 'y_m', 'yaw_rad'])

    def record_goal_reference(self, now, ns, age, measurements):
        """Save the selected tag even when the multi-tag frame cannot establish consensus."""
        label = f'{self.goal_tag[0]}:{self.goal_tag[1]}'
        try:
            pose = self.model.estimate_tag(measurements, self.goal_tag, now - age)
        except (ValueError, np.linalg.LinAlgError) as error:
            self.goal_log.writerow([now, ns, False, str(error), label, '', '', ''])
            return

        # preserve capture timing for stopped confirmation
        self.goal_samples.append(pose)
        self.goal_log.writerow([now, ns, True, 'independent goal reference', label,
                               pose.x, pose.y, pose.yaw])

    def observe(self, family, message):
        """Combine family packets without counting an image more than once."""
        stamp = message.header.stamp.sec * 10**9 + message.header.stamp.nanosec
        self.frames.add(stamp, message.header.frame_id, family, message, time.monotonic())

    def lookup(self, frame, key, stamp):
        """Read only the configured tag frame at the image capture timestamp."""
        # match geometry to the capture timestamp
        result = self.buffer.lookup_transform(frame, tag_frame(key), stamp)
        translation, q = result.transform.translation, result.transform.rotation
        quaternion = np.array([q.x, q.y, q.z, q.w])
        if not np.isfinite(quaternion).all() or abs(np.linalg.norm(quaternion) - 1.0) > 0.05:
            raise ValueError('invalid tag quaternion')
        rotation = quaternion_to_rotation_matrix(q.x, q.y, q.z, q.w)
        return make_transform(rotation, [translation.x, translation.y, translation.z])

    def measurements(self, capture, stamp):
        """Collect usable family-aware transforms and report any still missing TF."""
        result, missing, seen = [], False, set()
        for family, message in capture.messages.items():
            for item in message.detections:
                try:
                    key = canonical_family(item.family), item.id
                except ValueError:
                    continue
                if key[0] != family or key not in self.model.landmarks or key in seen:
                    continue
                if not math.isfinite(item.decision_margin) or item.decision_margin < self.policy['min_decision_margin']:
                    continue
                if not 0 <= item.hamming <= self.policy['max_hamming']:
                    continue

                seen.add(key)

                # permit healthy tags despite missing transforms
                try:
                    result.append((key, self.lookup(capture.frame, key, stamp)))
                except TransformException:
                    missing = True

        return result, missing

    def candidate_details(self, measurements, selected=None):
        """Expose each tag's independent world pose before combining estimates."""
        # identify which landmark produces a position bias
        details = {}
        for key, transform in measurements:
            label = f'{key[0]}:{key[1]}'
            try:
                details[label] = {'pose_xy_yaw': self.model.candidate(key, transform),
                                  'camera_range_m': float(np.linalg.norm(transform[:3, 3])),
                                  'fusion_status': ('not fused' if selected is None else
                                                    'kept' if key in selected else 'outlier')}
            except (ValueError, np.linalg.LinAlgError) as error:
                details[label] = {'rejected': str(error)}
        return details

    def update(self, now):
        """Accept one fresh image after briefly combining both detector results."""
        # preserve distinct capture identities across families
        capture = self.frames.ready(now, self.families)
        if capture is None:
            return
        ns = capture.stamp_ns
        age = (self.node.get_clock().now().nanoseconds - ns) / 1e9
        if ns <= 0 or not 0 <= age <= self.max_age:
            self.reject(now, ns, 'stale or future-dated detection frame')
            return

        # allow timestamped transforms to arrive asynchronously
        stamp = Time(nanoseconds=ns, clock_type=self.node.get_clock().clock_type)
        measurements = []
        try:
            measurements, missing = self.measurements(capture, stamp)
            if missing and now - capture.received < 0.15:
                return
            self.record_goal_reference(now, ns, age, measurements)
            if missing and not measurements:
                self.reject(now, ns, 'timestamped TF unavailable for mapped tags', self.candidate_details(measurements))
                return
            pose = self.model.estimate(measurements, now - age)
        except (ValueError, np.linalg.LinAlgError) as error:
            self.reject(now, ns, str(error), self.candidate_details(measurements))
            return

        # publish one measured pose per image
        self.frames.finish(ns)
        self.samples.append(pose)
        reason = 'mapped landmark estimate; some TF unavailable' if missing else 'mapped landmark estimate'
        details = self.candidate_details(measurements, pose.tags)
        discarded = [tag for tag, item in details.items() if item.get('fusion_status') == 'outlier']
        if discarded:
            reason += '; discarded outliers: ' + ', '.join(discarded)

        # retain all candidates alongside the selected subset
        ranges = {f'{key[0]}:{key[1]}': round(float(np.linalg.norm(transform[:3, 3])), 4)
                  for key, transform in measurements if key in pose.tags}
        self.log.observations.writerow([now, ns, True, reason, pose.x, pose.y, pose.yaw, pose.tags, json.dumps(ranges), json.dumps(details)])
        output = PoseStamped()
        output.header.stamp = next(iter(capture.messages.values())).header.stamp
        output.header.frame_id = 'map'
        output.pose.position.x, output.pose.position.y = pose.x, pose.y
        output.pose.orientation.z, output.pose.orientation.w = math.sin(pose.yaw / 2), math.cos(pose.yaw / 2)
        self.publisher.publish(output)

    def reject(self, now, stamp, reason, candidates=None):
        """Record a rejection without refreshing the age of the previous pose."""
        self.frames.finish(stamp)
        self.log.observations.writerow([now, stamp, False, reason, '', '', '', '', '', json.dumps(candidates or {})])
