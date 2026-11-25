from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """
    Launch file for Homework 4 Safe Voronoi-Based Path Planning and Navigation.

    Starts all required nodes for safe path navigation:
    1. motor_control - communicates with robot hardware via serial
    2. velocity_mapping - converts Twist commands (/cmd_vel) to motor commands (/motor_commands)
    3. hw4_localization - estimates robot pose from visible AprilTags using known landmark coordinates
    4. hw4_control - computes safe Voronoi path and executes closed-loop waypoint tracking
    """
    landmarks_file_arg = DeclareLaunchArgument(
        'landmarks_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('robot_testing'),
            'config',
            'hw4_landmarks.json',
        ]),
        description='Path to the HW4 landmarks JSON configuration file'
    )

    start_x_arg = DeclareLaunchArgument(
        'start_x_cm',
        default_value='234.5',
        description='Start X coordinate (cm) of camera reference point'
    )
    start_y_arg = DeclareLaunchArgument(
        'start_y_cm',
        default_value='63.0',
        description='Start Y coordinate (cm) of camera reference point'
    )
    goal_x_arg = DeclareLaunchArgument(
        'goal_x_cm',
        default_value='65.0',
        description='Goal X coordinate (cm) of camera reference point'
    )
    goal_y_arg = DeclareLaunchArgument(
        'goal_y_cm',
        default_value='229.5',
        description='Goal Y coordinate (cm) of camera reference point'
    )
    speed_arg = DeclareLaunchArgument(
        'speed_mps',
        default_value='0.18',
        description='Calibrated nominal forward speed (m/s)'
    )
    turn_time_arg = DeclareLaunchArgument(
        'turn_time_360_s',
        default_value='4.8',
        description='Calibrated seconds per 360-degree in-place turn'
    )
    turn_power_arg = DeclareLaunchArgument(
        'turn_motor_power',
        default_value='0.38',
        description='Motor command magnitude for in-place turning (0.0 to 0.5)'
    )

    # Node 1: Motor Controller (hardware serial interface)
    motor_controller = Node(
        package='robot_testing',
        executable='motor_control',
        name='motor_control',
        output='screen',
        emulate_tty=True,
    )

    # Node 2: Velocity Mapping (/cmd_vel -> /motor_commands)
    velocity_mapping = Node(
        package='robot_testing',
        executable='velocity_mapping',
        name='velocity_mapping',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'turn_motor_power': LaunchConfiguration('turn_motor_power'),
        }],
    )

    # Node 3: AprilTag Localization Node (/detections -> /robot_pose and /get_robot_pose)
    localization_node = Node(
        package='robot_testing',
        executable='hw4_localization',
        name='hw4_localization',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'landmarks_file': LaunchConfiguration('landmarks_file'),
        }],
    )

    # Node 4: HW4 Safe Voronoi Navigation Node (hw4_safe)
    control_node = Node(
        package='robot_testing',
        executable='hw4_safe',
        name='hw4_safe',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'start_x_cm': LaunchConfiguration('start_x_cm'),
            'start_y_cm': LaunchConfiguration('start_y_cm'),
            'goal_x_cm': LaunchConfiguration('goal_x_cm'),
            'goal_y_cm': LaunchConfiguration('goal_y_cm'),
            'speed_mps': LaunchConfiguration('speed_mps'),
            'turn_time_360_s': LaunchConfiguration('turn_time_360_s'),
            'turn_motor_power': LaunchConfiguration('turn_motor_power'),
        }],
    )

    return LaunchDescription([
        landmarks_file_arg,
        start_x_arg,
        start_y_arg,
        goal_x_arg,
        goal_y_arg,
        speed_arg,
        turn_time_arg,
        turn_power_arg,
        motor_controller,
        velocity_mapping,
        localization_node,
        control_node,
    ])
