from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Launch file for Homework 3 Manual Movement Testing.

    Used when picking up the robot and manually walking it (e.g. in a square).
    Relies purely on visual motion estimation from stationary AprilTags;
    cmd_vel fallback is disabled so no motion is fabricated when stationary or in absence of tags.

    Supports recording a rosbag directly by passing: record_bag:=true
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

    record_bag_arg = DeclareLaunchArgument(
        'record_bag',
        default_value='false',
        description='Whether to automatically record a rosbag of landmarks, trajectory, and TF'
    )

    bag_name_arg = DeclareLaunchArgument(
        'bag_name',
        default_value='hw3_results/manual_run_bag',
        description='Path/name for the output rosbag directory'
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

    bag_recorder = ExecuteProcess(
        condition=IfCondition(LaunchConfiguration('record_bag')),
        cmd=[
            'ros2', 'bag', 'record',
            '/ekf_slam/landmarks',
            '/ekf_slam/trajectory',
            '/ekf_slam/pose',
            '/tf',
            '/tf_static',
            '-o', LaunchConfiguration('bag_name'),
        ],
        output='screen',
    )

    return LaunchDescription([
        params_file_arg,
        record_bag_arg,
        bag_name_arg,
        vo_node,
        ekf_node,
        bag_recorder,
    ])
