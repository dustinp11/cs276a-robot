"""Publish estimated robot trajectory, landmark labels, and 95% covariance ellipses."""

import math

import numpy as np
from geometry_msgs.msg import Point, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Path
from rclpy.qos import DurabilityPolicy, QoSProfile
from visualization_msgs.msg import Marker, MarkerArray


class MapPublisher:
    """Expose RViz-compatible estimates in a single local map frame without ground truth."""

    def __init__(self, node, frame):
        """Create latched visualization publishers and a bounded estimated path."""
        self.frame = frame
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pose_pub = node.create_publisher(PoseWithCovarianceStamped, '/hw3/pose', qos)
        self.path_pub = node.create_publisher(Path, '/hw3/trajectory', qos)
        self.map_pub = node.create_publisher(MarkerArray, '/hw3/landmarks', qos)
        self.path = Path()
        self.path.header.frame_id = frame

    def publish(self, estimator, stamp):
        """Publish one posterior estimate, appending its position to the recorded path."""
        pose = self._pose(estimator, stamp)
        self.pose_pub.publish(pose)
        point = PoseStamped()
        point.header, point.pose = pose.header, pose.pose.pose
        self.path.poses.append(point)
        self.path.poses = self.path.poses[-10000:]
        self.path.header.stamp = stamp
        self.path_pub.publish(self.path)
        self.map_pub.publish(self._landmarks(estimator, stamp))

    def _pose(self, estimator, stamp):
        """Embed planar [x, y, yaw] mean/covariance into the ROS 6D pose layout."""
        message = PoseWithCovarianceStamped()
        message.header.frame_id, message.header.stamp = self.frame, stamp
        x, y, yaw = estimator.robot_pose
        message.pose.pose.position.x, message.pose.pose.position.y = float(x), float(y)
        message.pose.pose.orientation.z = math.sin(yaw/2.)
        message.pose.pose.orientation.w = math.cos(yaw/2.)
        covariance = np.zeros((6, 6))
        covariance[np.ix_([0, 1, 5], [0, 1, 5])] = estimator.covariance[:3, :3]
        message.pose.covariance = covariance.ravel().tolist()
        return message

    def _marker(self, number, namespace, kind, stamp):
        """Create a visible marker in the shared local map frame."""
        marker = Marker()
        marker.header.frame_id, marker.header.stamp = self.frame, stamp
        marker.ns, marker.id, marker.type, marker.action = namespace, number, kind, Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.2, 0.8, 0.3, 1.0
        return marker

    def _landmarks(self, estimator, stamp):
        """Draw each discovered center, detector identity, and positional covariance."""
        markers = MarkerArray()
        for number, key in enumerate(estimator.landmark_ids):
            position, covariance = estimator.landmark(key)
            center = self._marker(number, 'centers', Marker.SPHERE, stamp)
            center.pose.position.x, center.pose.position.y = map(float, position)
            center.scale.x = center.scale.y = center.scale.z = 0.08
            label = self._marker(number, 'ids', Marker.TEXT_VIEW_FACING, stamp)
            label.pose.position.x, label.pose.position.y = map(float, position)
            label.pose.position.z, label.scale.z, label.text = 0.15, 0.12, f'{key[0]}:{key[1]}'
            ellipse = self._ellipse(number, position, covariance, stamp)
            markers.markers.extend([center, label, ellipse])
        return markers

    def _ellipse(self, number, position, covariance, stamp):
        """Construct a 95% joint 2D Gaussian ellipse (chi-square value 5.991)."""
        marker = self._marker(number, 'covariance_95', Marker.LINE_STRIP, stamp)
        marker.scale.x = 0.015
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        radii = np.sqrt(np.maximum(eigenvalues, 0.) * 5.991464547)
        for angle in np.linspace(0., 2.*math.pi, 49):
            xy = position + eigenvectors @ (radii * [math.cos(angle), math.sin(angle)])
            point = Point()
            point.x, point.y = map(float, xy)
            marker.points.append(point)
        return marker
