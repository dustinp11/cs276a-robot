"""Launch raw camera detection, EKF-SLAM, square control, and HW3 motor interfaces."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Use all Standard41h12 IDs, measured tag size, and raw-image distortion-aware PnP."""
    share = FindPackageShare('hw3_code')
    config = PathJoinSubstitution([share, 'config', 'hw3.yaml'])
    camera_path = PathJoinSubstitution([share, 'config', 'camera_parameter.yaml'])
    arguments = _arguments(camera_path)
    camera = _camera()
    detector = _detector()
    driver = Node(package='hw3_code', executable='motor_control', name='hw3_motor_control',
                  output='screen', parameters=[config, {'motor_command_limit': ParameterValue(LaunchConfiguration('motor_limit'), value_type=float)}],
                  remappings=[('motor_commands', '/hw3/motor_commands')])
    mapper = Node(package='hw3_code', executable='velocity_mapping', name='hw3_velocity_mapping',
                  output='screen', parameters=[config, {
                      'turn_motor_input': ParameterValue(LaunchConfiguration('turn_input'), value_type=float),
                      'motor_command_limit': ParameterValue(LaunchConfiguration('motor_limit'), value_type=float)}])
    slam = Node(package='hw3_code', executable='slam', name='hw3_slam', output='screen',
                parameters=[config, _slam_overrides()])
    rviz = Node(package='rviz2', executable='rviz2', condition=IfCondition(LaunchConfiguration('rviz')),
                arguments=['-d', PathJoinSubstitution([share, 'config', 'hw3.rviz'])])
    shutdowns = [RegisterEventHandler(OnProcessExit(target_action=node, on_exit=[
        EmitEvent(event=Shutdown(reason='A required HW3 node exited'))]))
        for node in (camera, detector, driver, mapper, slam)]
    return LaunchDescription(arguments + shutdowns + [camera, detector, driver, mapper, slam, rviz])


def _arguments(camera_path):
    """Expose experiment settings while keeping physical dimensions in explicit units."""
    defaults = {'side_ft': '7.0', 'laps': '1', 'tag_size_m': '0.172', 'auto_start': 'true',
                'frame_rate': '10', 'web_port': '8080', 'rviz': 'false', 'output_directory': '~/hw3_runs',
                'camera_parameter_path': camera_path, 'turn_input': '0.6', 'motor_limit': '0.6'}
    return [DeclareLaunchArgument(name, default_value=value) for name, value in defaults.items()]


def _camera():
    """Keep images raw so the HW3 adapter can use matching K and distortion coefficients."""
    parameters = {'camera_id': 0, 'width': 1280, 'height': 720, 'input_format': 'NV12',
                  'output_format': 'BGR', 'topic_name': 'hw3/image_raw',
                  'image_rectify': False, 'image_compress': False,
                  'camera_parameter_path': LaunchConfiguration('camera_parameter_path'),
                  'frame_rate': ParameterValue(LaunchConfiguration('frame_rate'), value_type=int)}
    return Node(package='robot_vision_camera', executable='robot_vision_camera_node',
                name='hw3_camera', parameters=[parameters], output='screen')


def _detector():
    """Detect the full family; disable detector pose/TF to avoid raw/rectified K mismatch."""
    parameters = {'family': 'Standard41h12', 'size': ParameterValue(LaunchConfiguration('tag_size_m'), value_type=float),
                  'image_transport': 'raw', 'pose_estimation_method': '', 'max_hamming': 1,
                  'detector.threads': 4, 'detector.decimate': 1.0, 'detector.refine': True,
                  'detector.blur': 0.0, 'detector.sharpening': 0.25, 'detector.debug': False}
    return Node(package='apriltag_ros', executable='apriltag_node', name='hw3_detector',
                parameters=[parameters], output='screen', remappings=[
                    ('image_rect', '/hw3/image_raw'), ('camera_info', '/hw3/camera_info'),
                    ('detections', '/hw3/detections')])


def _slam_overrides():
    """Convert launch strings to typed estimator/controller parameters."""
    return {'web_port': ParameterValue(LaunchConfiguration('web_port'), value_type=int),
            'square_side_ft': ParameterValue(LaunchConfiguration('side_ft'), value_type=float),
            'laps': ParameterValue(LaunchConfiguration('laps'), value_type=int),
            'tag_size_m': ParameterValue(LaunchConfiguration('tag_size_m'), value_type=float),
            'auto_start': ParameterValue(LaunchConfiguration('auto_start'), value_type=bool),
            'output_directory': LaunchConfiguration('output_directory')}
