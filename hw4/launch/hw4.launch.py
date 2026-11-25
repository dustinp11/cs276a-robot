"""Stable run-1 launch: HW3 hardware, calibrated camera, mapped tags, and HW4."""

from pathlib import Path
import shutil

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, ExecuteProcess, GroupAction, IncludeLaunchDescription, OpaqueFunction, RegisterEventHandler, SetEnvironmentVariable
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from hw4.config import load_validated_environment
from hw4.settings import load_settings
from hw4.tags import detection_topic, detector_parameters, mapped_families


def generate_launch_description():
    """Keep the existing command interface and require a unique output directory."""
    config = PathJoinSubstitution([FindPackageShare('hw4'), 'config'])
    camera = PathJoinSubstitution([FindPackageShare('hw3'), 'config', 'camera_parameter.yaml'])
    arguments = [DeclareLaunchArgument(name, default_value=PathJoinSubstitution([config, filename]))
                 for name, filename in [('environment_file', 'environment.json'),
                                        ('motion_file', 'motion.yaml'), ('navigation_file', 'navigation.yaml')]]
    arguments += [DeclareLaunchArgument('output_dir', description='Absolute path to a new result directory'),
                  DeclareLaunchArgument('camera_parameter_path', default_value=EnvironmentVariable(
                      'CAMERA_CALIBRATION', default_value=camera))]
    return LaunchDescription(arguments + [OpaqueFunction(function=configured_nodes)])


def configured_nodes(context):
    """Validate measured input before opening camera or motor hardware."""
    values = {name: LaunchConfiguration(name).perform(context) for name in
              ('environment_file', 'motion_file', 'navigation_file', 'output_dir', 'camera_parameter_path')}
    environment = load_validated_environment(values['environment_file'])
    load_settings(values['navigation_file'])
    output = Path(values['output_dir'])
    if not output.is_absolute() or (output / 'session.lock').exists():
        raise ValueError('Choose a new absolute output_dir; do not reuse a previous mission directory.')

    # archive actual camera intrinsics before launch
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(values['camera_parameter_path'], output / 'camera_parameter.yaml')
    motion = values['motion_file']
    nodes = [Node(package='hw3', executable=name, output='screen', parameters=[motion])
             for name in ('motor_control', 'velocity_mapping')]
    nodes += [Node(package='hw4', executable='hw4_navigation', output='screen',
                   parameters=[motion, {key: value for key, value in values.items() if key != 'camera_parameter_path'}]),
              recorder(values['output_dir'])]
    handlers = [RegisterEventHandler(OnProcessExit(target_action=node, on_exit=[EmitEvent(
        event=Shutdown(reason='HW4 essential process exited; ending run.'))])) for node in nodes]
    return handlers + [camera_launch(values['camera_parameter_path'])] + detectors(environment) + nodes


def camera_launch(calibration):
    """Reuse HW3's camera settings with scoped GStreamer startup diagnostics."""
    # expose pipeline failures without altering calibration
    source = PythonLaunchDescriptionSource(PathJoinSubstitution([
        FindPackageShare('robot_vision_camera'), 'launch', 'robot_vision_camera.launch.py']))
    camera = IncludeLaunchDescription(source, launch_arguments={
        'image_rectify': 'true', 'camera_parameter_path': calibration,
    }.items())
    return GroupAction(actions=[
        SetEnvironmentVariable('GST_DEBUG', EnvironmentVariable('GST_DEBUG', default_value='3')),
        SetEnvironmentVariable('GST_DEBUG_NO_COLOR', '1'),
        camera,
    ], scoped=True)


def detectors(environment):
    """Launch one detector per mapped family with separate topics and TF frames."""
    # separate detection streams without duplicating identities
    config = PathJoinSubstitution([FindPackageShare('apriltag_ros'), 'cfg', 'tags_36h11.yaml'])
    return [Node(package='apriltag_ros', executable='apriltag_node', name=f'apriltag_{family}',
                 parameters=[config, detector_parameters(environment, family)], output='screen',
                 remappings=[('image_rect', '/camera/image_raw'), ('camera_info', '/camera/camera_info'),
                             ('detections', detection_topic(family))])
            for family in mapped_families(environment)]


def recorder(output):
    """Record commands and measured localization without saving raw camera video."""
    return ExecuteProcess(cmd=[
        'ros2', 'bag', 'record', '-o', str(Path(output) / 'telemetry'),
        '/cmd_vel', '/cmd_vel_applied', '/motor_commands', '/imu_data',
        detection_topic('36h11'), detection_topic('41h12'), '/tf', '/tf_static', '/hw4/pose', '/hw4/status',
    ], output='screen')
