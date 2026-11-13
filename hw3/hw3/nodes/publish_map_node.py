"""ROS 2 Node to republish saved EKF-SLAM landmarks and trajectory for RViz inspection.

Allows viewing the final landmark map and trajectory in RViz anytime after manual mode
without needing a live robot or running filter.
"""

import argparse
import csv
import math
from pathlib import Path
import sys
from typing import Dict, List, Optional
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from geometry_msgs.msg import Point, PoseStamped, TransformStamped
    from nav_msgs.msg import Path as NavPath
    from std_msgs.msg import ColorRGBA
    from visualization_msgs.msg import Marker, MarkerArray
    from tf2_ros import StaticTransformBroadcaster
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False
    Node = object


class MapPublisherNode(Node):
    """Republishes saved landmarks and trajectory with latched QoS for RViz."""

    def __init__(self, landmarks_csv: str, trajectory_csv: Optional[str] = None, map_frame: str = 'map'):
        if not HAS_ROS2:
            raise RuntimeError("ROS 2 is required.")
        super().__init__('map_publisher_node')

        self.map_frame = map_frame
        self.landmarks = self._load_landmarks(landmarks_csv)
        self.trajectory = self._load_trajectory(trajectory_csv) if trajectory_csv else []

        # Latched QoS (Transient Local) so late-joining RViz immediately receives the map
        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        self.marker_pub = self.create_publisher(MarkerArray, '/ekf_slam/landmarks', latched_qos)
        self.path_pub = self.create_publisher(NavPath, '/ekf_slam/trajectory', latched_qos)

        # Static transform broadcaster for map frame
        self.tf_broadcaster = StaticTransformBroadcaster(self)
        self._broadcast_static_tf()

        # Publish immediately, and timer at 1 Hz for continuous refresh
        self.publish_all()
        self.create_timer(1.0, self.publish_all)

        self.get_logger().info(
            f"Publishing {len(self.landmarks)} landmarks and {len(self.trajectory)} trajectory poses. "
            f"Open RViz and subscribe to /ekf_slam/landmarks and /ekf_slam/trajectory (frame: {self.map_frame})."
        )

    def _load_landmarks(self, path: str) -> Dict[int, dict]:
        lms = {}
        csv_path = Path(path)
        if not csv_path.exists():
            self.get_logger().warn(f"Landmarks file not found: {csv_path}")
            return lms
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                tag_id = int(row['tag_id'])
                lms[tag_id] = {
                    'tag_id': tag_id,
                    'x': float(row['x_m']),
                    'y': float(row['y_m']),
                    'sigma_x': float(row.get('sigma_x_m', 0.05)),
                    'sigma_y': float(row.get('sigma_y_m', 0.05)),
                    'cov_xy': float(row.get('cov_xy', 0.0)),
                }
        return lms

    def _load_trajectory(self, path: str) -> List[dict]:
        traj = []
        csv_path = Path(path)
        if not csv_path.exists():
            return traj
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                traj.append({
                    'timestamp_s': float(row['timestamp_s']),
                    'x': float(row['x_m']),
                    'y': float(row['y_m']),
                    'theta': float(row['theta_rad']),
                })
        return traj

    def _broadcast_static_tf(self):
        tf_msg = TransformStamped()
        tf_msg.header.stamp = self.get_clock().now().to_msg()
        tf_msg.header.frame_id = self.map_frame
        tf_msg.child_frame_id = "base_link"
        tf_msg.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(tf_msg)

    def publish_all(self):
        stamp = self.get_clock().now().to_msg()
        markers = []

        # Delete all previous markers
        del_marker = Marker()
        del_marker.action = Marker.DELETEALL
        markers.append(del_marker)

        mid = 0
        for tag_id, lm in self.landmarks.items():
            # 1. Sphere for landmark position
            m_pos = Marker()
            m_pos.header.stamp = stamp
            m_pos.header.frame_id = self.map_frame
            m_pos.ns = "landmarks"
            m_pos.id = mid
            mid += 1
            m_pos.type = Marker.SPHERE
            m_pos.action = Marker.ADD
            m_pos.pose.position.x = lm['x']
            m_pos.pose.position.y = lm['y']
            m_pos.pose.position.z = 0.15
            m_pos.pose.orientation.w = 1.0
            m_pos.scale.x = 0.14
            m_pos.scale.y = 0.14
            m_pos.scale.z = 0.14
            m_pos.color = ColorRGBA(r=0.1, g=0.85, b=0.2, a=0.95)
            markers.append(m_pos)

            # 2. Text label with Tag ID
            m_text = Marker()
            m_text.header.stamp = stamp
            m_text.header.frame_id = self.map_frame
            m_text.ns = "labels"
            m_text.id = mid
            mid += 1
            m_text.type = Marker.TEXT_VIEW_FACING
            m_text.action = Marker.ADD
            m_text.pose.position.x = lm['x']
            m_text.pose.position.y = lm['y']
            m_text.pose.position.z = 0.32
            m_text.scale.z = 0.15
            m_text.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            m_text.text = f"Tag {tag_id}"
            markers.append(m_text)

            # 3. Covariance Ellipse
            m_ellipse = Marker()
            m_ellipse.header.stamp = stamp
            m_ellipse.header.frame_id = self.map_frame
            m_ellipse.ns = f"ellipse_{tag_id}"
            m_ellipse.id = mid
            mid += 1
            m_ellipse.type = Marker.LINE_STRIP
            m_ellipse.action = Marker.ADD
            m_ellipse.scale.x = 0.02
            m_ellipse.color = ColorRGBA(r=0.2, g=0.6, b=1.0, a=0.85)

            # Compute ellipse axes from 2x2 covariance
            cov = np.array([
                [lm['sigma_x'] ** 2, lm['cov_xy']],
                [lm['cov_xy'], lm['sigma_y'] ** 2],
            ])
            w, v = np.linalg.eigh(cov)
            order = w.argsort()[::-1]
            w = w[order]
            v = v[:, order]
            a = 2.0 * math.sqrt(max(1e-6, w[0]))
            b = 2.0 * math.sqrt(max(1e-6, w[1]))
            angle = math.atan2(v[1, 0], v[0, 0])

            pts = []
            for i in range(33):
                th = 2.0 * math.pi * i / 32
                px = a * math.cos(th)
                py = b * math.sin(th)
                rx = lm['x'] + px * math.cos(angle) - py * math.sin(angle)
                ry = lm['y'] + px * math.sin(angle) + py * math.cos(angle)
                p = Point()
                p.x = float(rx)
                p.y = float(ry)
                p.z = 0.05
                pts.append(p)
            m_ellipse.points = pts
            markers.append(m_ellipse)

        m_array = MarkerArray()
        m_array.markers = markers
        self.marker_pub.publish(m_array)

        # Publish trajectory
        if self.trajectory:
            path_msg = NavPath()
            path_msg.header.stamp = stamp
            path_msg.header.frame_id = self.map_frame
            poses = []
            for item in self.trajectory:
                ps = PoseStamped()
                ps.header.stamp = stamp
                ps.header.frame_id = self.map_frame
                ps.pose.position.x = item['x']
                ps.pose.position.y = item['y']
                ps.pose.position.z = 0.0
                ps.pose.orientation.z = math.sin(item['theta'] * 0.5)
                ps.pose.orientation.w = math.cos(item['theta'] * 0.5)
                poses.append(ps)
            path_msg.poses = poses
            self.path_pub.publish(path_msg)


def main(args=None):
    parser = argparse.ArgumentParser(description="Republish EKF-SLAM map markers for RViz.")
    parser.add_argument('--landmarks', '-l', default='hw3_results/ekf_slam_landmarks.csv', help="Path to landmarks CSV")
    parser.add_argument('--trajectory', '-t', default='hw3_results/ekf_slam_trajectory.csv', help="Path to trajectory CSV")
    parser.add_argument('--frame', default='map', help="Map frame ID")
    parsed_args, ros_args = parser.parse_known_args(args=sys.argv[1:] if args is None else args)

    if not HAS_ROS2:
        print("ROS 2 is required.")
        sys.exit(1)

    rclpy.init(args=ros_args)
    node = MapPublisherNode(parsed_args.landmarks, parsed_args.trajectory, parsed_args.frame)
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
