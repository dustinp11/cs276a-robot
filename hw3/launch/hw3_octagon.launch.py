"""Reuse the HW3 stack for eight 75 cm legs and measured half-second left turns."""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Launch the octagon profile, waiting for the existing start service.

    Args:
        None.

    Returns:
        LaunchDescription reusing sensors, SLAM, logs, and the polygon controller.
    """
    source = PythonLaunchDescriptionSource(PathJoinSubstitution([
        FindPackageShare('hw3'), 'launch', 'hw3_navigation.launch.py']))
    return LaunchDescription([IncludeLaunchDescription(source, launch_arguments={
        'mission': 'octagon',
    }.items())])
