from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Launch file for Homework 3: Visual Odometry and EKF-SLAM.

    Args:
        None.

    Returns:
        LaunchDescription for this HW3 workflow.
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

    launch_motors_arg = DeclareLaunchArgument(
        'launch_motors',
        default_value='false',
        description='Whether to start the hardware motor_control and velocity_mapping nodes'
    )

    # 1. Visual Odometry Node
    vo_node = Node(
        package='hw3',
        executable='visual_odometry_node',
        name='visual_odometry_node',
        output='screen',
        emulate_tty=True,
        parameters=[LaunchConfiguration('params_file')],
    )

    # 2. EKF-SLAM Node
    ekf_node = Node(
        package='hw3',
        executable='ekf_slam_node',
        name='ekf_slam_node',
        output='screen',
        emulate_tty=True,
        parameters=[LaunchConfiguration('params_file')],
    )

    # Optional Hardware Nodes (when driving robot motors)
    motor_node = Node(
        condition=IfCondition(LaunchConfiguration('launch_motors')),
        package='hw3',
        executable='motor_control',
        name='motor_control',
        output='screen',
        emulate_tty=True,
    )

    velocity_node = Node(
        condition=IfCondition(LaunchConfiguration('launch_motors')),
        package='hw3',
        executable='velocity_mapping',
        name='velocity_mapping',
        parameters=[LaunchConfiguration('params_file'), {'turn_calibration.enabled': True}],
        output='screen',
        emulate_tty=True,
    )

    return LaunchDescription([
        params_file_arg,
        launch_motors_arg,
        vo_node,
        ekf_node,
        motor_node,
        velocity_node,
    ])
