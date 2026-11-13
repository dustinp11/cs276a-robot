from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Launch file for Homework 3 Manual Movement Testing.

    Used when manually moving or sliding the robot (e.g. in a square).
    Relies purely on visual motion estimation from stationary AprilTags;
    cmd_vel fallback is disabled so no motion is fabricated when stationary.
    """
    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('hw3'),
            'config',
            'hw3_params.yaml',
        ]),
        description='Path to HW3 parameters YAML file'
    )

    vo_node = Node(
        package='hw3',
        executable='visual_odometry_node',
        name='visual_odometry_node',
        output='screen',
        emulate_tty=True,
        parameters=[LaunchConfiguration('params_file')],
    )

    ekf_node = Node(
        package='hw3',
        executable='ekf_slam_node',
        name='ekf_slam_node',
        output='screen',
        emulate_tty=True,
        parameters=[
            LaunchConfiguration('params_file'),
            {
                'ekf_slam.motion_prediction_source': 'visual_odometry',
                'ekf_slam.use_cmd_vel_fallback': False,
            }
        ],
    )

    return LaunchDescription([
        params_file_arg,
        vo_node,
        ekf_node,
    ])
