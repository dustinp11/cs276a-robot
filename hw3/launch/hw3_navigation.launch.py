from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Launch file for Homework 3: Closed-Loop Square Navigation with EKF-SLAM.

    Starts:
    1. Visual Odometry Node (inter-frame motion estimation)
    2. EKF-SLAM Node (dynamic landmark discovery & robot pose estimation)
    3. Motor Controller Node (hardware serial interface to /dev/ttyUSB0)
    4. Velocity Mapping Node (converts /cmd_vel into left/right /motor_commands)
    5. Square Navigation Node (closed-loop 7ft x 7ft square waypoint controller)
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

    auto_start_arg = DeclareLaunchArgument(
        'auto_start',
        default_value='false',
        description='Whether to start square navigation immediately on launch'
    )

    square_size_ft_arg = DeclareLaunchArgument(
        'square_size_ft',
        default_value='7.0',
        description='Side length of the square trajectory in feet (default 7.0 ft)'
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

    # 3. Motor Controller (hardware serial interface from robot_testing)
    motor_node = Node(
        package='robot_testing',
        executable='motor_control',
        name='motor_control',
        output='screen',
        emulate_tty=True,
    )

    # 4. Velocity Mapping (/cmd_vel -> /motor_commands from robot_testing)
    velocity_node = Node(
        package='robot_testing',
        executable='velocity_mapping',
        name='velocity_mapping',
        output='screen',
        emulate_tty=True,
    )

    # 5. Closed-Loop Square Navigation Controller
    navigation_node = Node(
        package='hw3',
        executable='navigation_node',
        name='square_navigation_node',
        output='screen',
        emulate_tty=True,
        parameters=[
            LaunchConfiguration('params_file'),
            {
                'auto_start': LaunchConfiguration('auto_start'),
                'square_size_ft': LaunchConfiguration('square_size_ft'),
            }
        ],
    )

    return LaunchDescription([
        params_file_arg,
        auto_start_arg,
        square_size_ft_arg,
        vo_node,
        ekf_node,
        motor_node,
        velocity_node,
        navigation_node,
    ])
