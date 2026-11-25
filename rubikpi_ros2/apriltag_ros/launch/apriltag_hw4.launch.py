#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Launch file for HW4 AprilTag detection covering both 36h11 and 41h12 families."""
    image_topic_arg = DeclareLaunchArgument(
        'image_topic',
        default_value='/camera/image_raw',
        description='Input image topic'
    )

    camera_info_topic_arg = DeclareLaunchArgument(
        'camera_info_topic',
        default_value='/camera/camera_info',
        description='Camera info topic'
    )

    config_36h11 = PathJoinSubstitution([
        FindPackageShare('apriltag_ros'),
        'cfg',
        'tags_36h11.yaml'
    ])

    config_41h12 = PathJoinSubstitution([
        FindPackageShare('apriltag_ros'),
        'cfg',
        'tags_41h12.yaml'
    ])

    # Node 1: Wall tag detector (family 36h11, physical size 17.2 cm)
    node_36h11 = Node(
        package='apriltag_ros',
        executable='apriltag_node',
        name='apriltag_node_36h11',
        remappings=[
            ('image_rect', LaunchConfiguration('image_topic')),
            ('camera_info', LaunchConfiguration('camera_info_topic')),
            ('detections', '/detections'),
        ],
        parameters=[config_36h11],
        output='screen'
    )

    # Node 2: Obstacle tag detector (family Standard41h12, physical size 18.7 cm)
    node_41h12 = Node(
        package='apriltag_ros',
        executable='apriltag_node',
        name='apriltag_node_41h12',
        remappings=[
            ('image_rect', LaunchConfiguration('image_topic')),
            ('camera_info', LaunchConfiguration('camera_info_topic')),
            ('detections', '/detections'),
        ],
        parameters=[config_41h12],
        output='screen'
    )

    return LaunchDescription([
        image_topic_arg,
        camera_info_topic_arg,
        node_36h11,
        node_41h12,
    ])
