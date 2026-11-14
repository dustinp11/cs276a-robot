"""HW3 Refactored Package.

Consolidated into 4 primary modules:
- motor_control: Motor driver, velocity mapping, turn and forward calibration
- slam_vision: Shared coordinates, visual odometry, EKF-SLAM, and data recording
- square_navigation: Two-lap square navigation (dead-reckoning + return-to-origin + closed loop)
- octagon_navigation: Octagon 8-side navigation (timed 45-degree turns and pauses)
"""

from .motor_control import (
    ForwardCalibration,
    MotorCommand,
    MotorControllerNode,
    TurnCalibration,
    VelocityMapper,
    VelocityToMotorNode,
    read_forward_calibration,
    read_turn_calibration,
)
from .slam_vision import (
    BaseNavigationNode,
    CameraExtrinsics,
    DataRecorder,
    EkfSlam,
    EkfSlamNode,
    MapPublisherNode,
    NavigationOutput,
    NavigationPolicy,
    NavigationState,
    VisualFeedback,
    VisualMotionEstimator,
    VisualOdometryNode,
    Waypoint,
    wrap_angle,
)
from .square_navigation import (
    SquareNavigationNode,
    SquareNavigator,
    navigation_node_main,
)
from .octagon_navigation import (
    OctagonNavigationNode,
    OctagonNavigator,
    octagon_navigation_main,
)

__all__ = [
    'ForwardCalibration',
    'TurnCalibration',
    'VelocityMapper',
    'MotorCommand',
    'MotorControllerNode',
    'VelocityToMotorNode',
    'read_forward_calibration',
    'read_turn_calibration',
    'wrap_angle',
    'CameraExtrinsics',
    'EkfSlam',
    'EkfSlamNode',
    'VisualMotionEstimator',
    'VisualOdometryNode',
    'DataRecorder',
    'MapPublisherNode',
    'NavigationState',
    'NavigationPolicy',
    'NavigationOutput',
    'Waypoint',
    'VisualFeedback',
    'BaseNavigationNode',
    'SquareNavigator',
    'SquareNavigationNode',
    'navigation_node_main',
    'OctagonNavigator',
    'OctagonNavigationNode',
    'octagon_navigation_main',
]
