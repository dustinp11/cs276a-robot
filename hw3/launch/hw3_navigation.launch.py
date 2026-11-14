"""Launch camera, 36h11 detection, estimation, and a calibrated six-foot square."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _config_path(filename):
    """Resolve an installed HW3 configuration file."""
    return PathJoinSubstitution([FindPackageShare('hw3'), 'config', filename])


def _launch_arguments():
    """Declare shared configuration, sensors, and explicit mission-start options.

    Args:
        None.

    Returns:
        List of launch argument declarations; square lengths are in feet.
    """
    return [
        DeclareLaunchArgument('params_file', default_value=_config_path('hw3_params.yaml')),
        DeclareLaunchArgument('camera_parameter_path', default_value=EnvironmentVariable(
            'CAMERA_CALIBRATION', default_value=_config_path('camera_parameter.yaml'))),
        DeclareLaunchArgument('launch_sensors', default_value='true',
                              description='Start camera and 36h11 detector together'),
        DeclareLaunchArgument('auto_start', default_value='false',
                              description='Start motion automatically instead of waiting for the service'),
        DeclareLaunchArgument('mission', default_value='square',
                              description='square or octagon calibrated motion profile'),
        DeclareLaunchArgument('square_size_ft', default_value='6.0',
                              description='Square side length in feet'),
    ]


def _include_sensor(package, filename, arguments):
    """Include a sensor launch when launch_sensors is enabled."""
    source = PythonLaunchDescriptionSource(PathJoinSubstitution([
        FindPackageShare(package), 'launch', filename]))
    return IncludeLaunchDescription(source, launch_arguments=arguments.items(),
                                    condition=IfCondition(LaunchConfiguration('launch_sensors')))


def _sensor_nodes():
    """Reuse the camera and AprilTag launches with the existing rectified topics.

    Args:
        None.

    Returns:
        Camera and tag36h11 detector launch inclusions.
    """
    camera = _include_sensor('robot_vision_camera', 'robot_vision_camera.launch.py', {
        'image_rectify': 'true',
        'camera_parameter_path': LaunchConfiguration('camera_parameter_path'),
    })
    detector = _include_sensor('apriltag_ros', 'apriltag_launch.py', {
        'image_topic': '/camera/image_raw',
        'camera_info_topic': '/camera/camera_info',
        'config_file': 'tags_36h11.yaml',
    })
    return [camera, detector]


def _hw3_node(executable, name=None, overrides=None):
    """Create an HW3 node using shared parameters and optional per-node overrides."""
    return Node(package='hw3', executable=executable, name=name or executable,
                output='screen', emulate_tty=True,
                parameters=[LaunchConfiguration('params_file'), overrides or {}])


def _control_nodes(context):
    """Connect sensor estimation to calibrated mapping, serial motors, and navigation.

    Args:
        context: Launch context used to select the calibrated motion profile.

    Returns:
        Five ROS nodes; navigation remains idle unless explicitly started.
    """
    mission = LaunchConfiguration('mission').perform(context)
    if mission not in ('square', 'octagon'):
        raise ValueError('Mission must be square or octagon.')
    turn_overrides = {'turn_calibration.enabled': True}
    navigation_overrides = {}
    if mission == 'octagon':
        turn_overrides.update({'turn_calibration.ccw_left_input': -0.35,
                               'turn_calibration.ccw_right_input': 0.35,
                               'turn_calibration.ccw_radps': 1.5707963267948966})
        navigation_overrides = {'navigation.side_count': 8,
                                'navigation.manual_motion_enabled': True,
                                'navigation.manual_forward_pause_s': 1.0,
                                'navigation.manual_midpoint_pause_s': 1.0,
                                'navigation.startup_scan_enabled': False,
                                'navigation.turn_pulse_rad': 0.08726646259971647,
                                'navigation.side_length_m': 0.75,
                                'navigation.late_scan_start_corner': 99,  # regular corners; no look-around or homing
                                'navigation.corner_turn_duration_s': 0.5,
                                'navigation.corner_stage_angle_rad': 0.7853981633974483,
                                'navigation.corner_ekf_weight': 0.0}
    return [
        _hw3_node('visual_odometry_node'),
        _hw3_node('ekf_slam_node', overrides={
            'ekf_slam.motion_prediction_source': 'cmd_vel',
            'ekf_slam.command_topic': '/cmd_vel_applied',
        }),
        _hw3_node('motor_control'),
        _hw3_node('velocity_mapping', overrides=turn_overrides),
        _hw3_node('navigation_node', name='square_navigation_node', overrides={
            **turn_overrides,
            'navigation.auto_start': LaunchConfiguration('auto_start'),
            'navigation.square_size_ft': LaunchConfiguration('square_size_ft'),
            **navigation_overrides,
        }),
    ]


def generate_launch_description():
    """Launch the complete square experiment, including its camera and detector."""
    return LaunchDescription(_launch_arguments() + _sensor_nodes() + [OpaqueFunction(function=_control_nodes)])
