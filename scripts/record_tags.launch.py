"""Record stationary 36h11 observations without motor or visualization nodes."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def camera_node():
    """Reuse the calibrated camera executable at 5 fps, without its GUI bridge."""
    return Node(
        package='robot_vision_camera', executable='robot_vision_camera_node',
        name='robot_vision_camera', output='screen', parameters=[{
            'frame_rate': 5, 'width': 1280, 'height': 720,
            'topic_name': 'camera/image_raw', 'image_compress': True,
            'image_rectify': True,
            'camera_parameter_path': ParameterValue(
                LaunchConfiguration('calibration'), value_type=str),
        }],
    )


def detector_node():
    """Detect every 36h11 ID and publish timestamped camera-relative tag TF."""
    return Node(
        package='apriltag_ros', executable='apriltag_node', name='apriltag',
        output='screen', remappings=[
            ('image_rect', '/camera/image_raw'), ('camera_info', '/camera/camera_info'),
        ], parameters=[{
            'family': '36h11', 'image_transport': 'raw',
            'size': ParameterValue(LaunchConfiguration('tag_size_m'), value_type=float),
            'max_hamming': 1, 'pose_estimation_method': 'pnp',
            'detector.threads': 4, 'detector.decimate': 1.0,
            'detector.refine': True, 'detector.sharpening': 0.25,
        }],
    )


def recorder_process():
    """Save compressed frames, intrinsics, detections, transforms, and ROS logs."""
    return ExecuteProcess(
        cmd=['ros2', 'bag', 'record', '--output', LaunchConfiguration('output'),
             '--topics', '/camera/image_raw/compressed', '/camera/camera_info',
             '/detections', '/tf', '/tf_static', '/rosout', '/parameter_events'],
        output='screen', sigterm_timeout='20', sigkill_timeout='5',
    )


def stop_on_exit(process):
    """Stop the whole session if a camera, detector, or recorder process exits."""
    return RegisterEventHandler(OnProcessExit(
        target_action=process,
        on_exit=[EmitEvent(event=Shutdown(reason='Recording session process exited'))],
    ))


def generate_launch_description():
    """Launch only stationary vision and recording; Ctrl-C finalizes the bag."""
    arguments = [
        DeclareLaunchArgument('calibration', description='Absolute camera calibration path'),
        DeclareLaunchArgument('output', description='New rosbag directory'),
        DeclareLaunchArgument('tag_size_m', default_value='0.172'),
    ]
    processes = [recorder_process(), camera_node(), detector_node()]
    handlers = [stop_on_exit(process) for process in processes]
    return LaunchDescription(arguments + handlers + processes)
