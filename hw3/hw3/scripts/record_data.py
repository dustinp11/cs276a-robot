#!/usr/bin/env python3
"""Record live AprilTag observations and motor commands to file for offline replay."""

import argparse
from pathlib import Path
import sys
import time

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.time import Time
    from geometry_msgs.msg import Twist
    from apriltag_msgs.msg import AprilTagDetectionArray
    from tf2_ros import Buffer, TransformException, TransformListener
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False
    Node = object

from hw3.coordinates import make_transform, quaternion_to_rotation_matrix
from hw3.data_recorder import DataRecorder


class ObservationRecorderNode(Node):
    """ROS 2 Node that listens to /detections, /tf, and /cmd_vel and saves to JSON."""

    def __init__(self, output_path: str):
        """Initialize timestamped observation and command recording.

        Args:
            output_path: Destination CSV or JSON file path.

        Returns:
            None. State changes and publications are performed in place.
        """
        if not HAS_ROS2:
            raise RuntimeError("ROS 2 is required to record live data.")
        super().__init__('observation_recorder_node')

        self.recorder = DataRecorder(output_path)
        self.tf_buffer = Buffer(node=self)
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(
            AprilTagDetectionArray,
            '/detections',
            self.on_detections,
            10
        )
        self.create_subscription(
            Twist,
            '/cmd_vel',
            self.on_cmd_vel,
            10
        )

        self.get_logger().info(f"Recording observations and commands to: {output_path}")

    def on_cmd_vel(self, msg: Twist):
        self.recorder.record_cmd_vel(msg.linear.x, msg.angular.z)

    def on_detections(self, msg: AprilTagDetectionArray):
        """Record usable timestamped AprilTag poses from one detection message.

        Args:
            msg: Incoming ROS message for this callback, with its original timestamp and frame.

        Returns:
            None. State changes and publications are performed in place.
        """
        stamp = msg.header.stamp
        stamp_s = stamp.sec + stamp.nanosec * 1e-9
        cam_frame = msg.header.frame_id if msg.header.frame_id else 'camera_frame'
        ros_time = Time(seconds=stamp.sec, nanoseconds=stamp.nanosec, clock_type=self.get_clock().clock_type)

        processed = []
        for det in msg.detections:
            tag_id = det.id
            family = det.family
            t_cam = self._lookup_tag(cam_frame, tag_id, family, ros_time)
            if t_cam is not None:
                processed.append({
                    'id': tag_id,
                    'family': family,
                    't_cam_tag': t_cam,
                    'margin': det.decision_margin,
                    'hamming': det.hamming,
                })

        if processed:
            self.recorder.record_frame(stamp_s, processed)

    def _lookup_tag(self, camera_frame, tag_id, family, stamp):
        """Resolve a detected tag pose in the camera optical frame.

        Args:
            camera_frame: Name of the camera optical frame used for tag pose lookup.
            tag_id: AprilTag integer ID used to associate the same stationary landmark.
            family: AprilTag family string used to resolve the detector TF frame name.
            stamp: ROS observation timestamp used for lookup or publication.

        Returns:
            4x4 tag-to-camera optical transform, or None when lookup fails.
        """
        for frame in [f"tag_{tag_id}", f"{family}:{tag_id}"]:
            try:
                tf_res = self.tf_buffer.lookup_transform(camera_frame, frame, stamp)
                t = tf_res.transform.translation
                q = tf_res.transform.rotation
                r = quaternion_to_rotation_matrix(q.x, q.y, q.z, q.w)
                return make_transform(r, [t.x, t.y, t.z])
            except TransformException:
                pass
        return None

    def save(self):
        self.recorder.save()
        self.get_logger().info(f"Saved {len(self.recorder.records)} frames to {self.recorder.output_path}")


def main(args=None):
    """Run this module through its command-line entry point.

    Args:
        args: Optional command-line arguments; None uses the process arguments.

    Returns:
        None. State changes and publications are performed in place.
    """
    parser = argparse.ArgumentParser(description="Record AprilTag observations and cmd_vel to JSON.")
    parser.add_argument('--output', '-o', default='hw3_results/recorded_run.json', help="Output JSON path")
    parsed_args, ros_args = parser.parse_known_args(args=sys.argv[1:] if args is None else args)

    if not HAS_ROS2:
        print("Error: ROS 2 is required to record live data.")
        sys.exit(1)

    rclpy.init(args=ros_args)
    node = ObservationRecorderNode(parsed_args.output)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
