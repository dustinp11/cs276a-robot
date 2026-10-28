from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """
    Launch file for Homework 2.

    Starts all required nodes for robot motion and AprilTag localization:
    1. motor_control - communicates with robot hardware via serial
    2. velocity_mapping - converts Twist commands (/cmd_vel) to motor commands (/motor_cmd)
    3. hw2_localization - estimates robot pose from AprilTags and serves /get_robot_pose
    4. hw2_control - stop-and-measure state machine driving robot to (1, 0)
    """
    landmarks_file_arg = DeclareLaunchArgument(
        'landmarks_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('robot_testing'),
            'config',
            'landmarks.json',
        ]),
        description='Path to the landmarks JSON configuration file'
    )

    # Node 1: Motor Controller (hardware serial interface)
    motor_controller = Node(
        package='robot_testing',
        executable='motor_control',
        name='motor_control',
        output='screen',
        emulate_tty=True,
    )

    # Node 2: Velocity Mapping (/cmd_vel -> /motor_cmd)
    velocity_mapping = Node(
        package='robot_testing',
        executable='velocity_mapping',
        name='velocity_mapping',
        output='screen',
        emulate_tty=True,
    )

    # Node 3: AprilTag Localization Service (/get_robot_pose)
    localization_node = Node(
        package='robot_testing',
        executable='hw2_localization',
        name='hw2_localization',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'landmarks_file': LaunchConfiguration('landmarks_file'),
        }],
    )

    # Node 4: HW2 Waypoint Controller
    control_node = Node(
        package='robot_testing',
        executable='hw2_control',
        name='hw2_control',
        output='screen',
        emulate_tty=True,
    )

    return LaunchDescription([
        landmarks_file_arg,
        motor_controller,
        velocity_mapping,
        localization_node,
        control_node,
    ])
