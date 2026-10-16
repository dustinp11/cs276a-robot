from collections import deque
import math
from unittest.mock import Mock

from robot_testing.robot_test import (
    ANGULAR_SPEED_CCW,
    ANGULAR_SPEED_CW,
    RobotTestNode,
)


def create_node():
    node = RobotTestNode.__new__(RobotTestNode)
    node.command_queue = deque()
    return node


def test_rotate_uses_counterclockwise_for_small_positive_turn():
    node = create_node()

    node.rotate(0.5)

    command, duration, movement_type, angle = node.command_queue.popleft()
    assert command == [-0.5, 0.5]
    assert math.isclose(duration, 0.5 / ANGULAR_SPEED_CCW)
    assert movement_type == 'rotate'
    assert angle == 0.5
    assert not node.command_queue


def test_rotate_uses_clockwise_for_small_negative_turn():
    node = create_node()

    node.rotate(-0.5)

    command, duration, movement_type, angle = node.command_queue.popleft()
    assert command == [0.5, -0.5]
    assert math.isclose(duration, 0.5 / ANGULAR_SPEED_CW)
    assert movement_type == 'rotate'
    assert math.isclose(angle, -0.5)
    assert not node.command_queue


def test_planner_queues_motion_through_each_waypoint():
    node = create_node()
    node.waypoints = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
    ]
    node.get_logger = Mock()

    node.plan_waypoints()

    assert [command[2] for command in node.command_queue] == [
        'move',
        'rotate',
        'move',
        'rotate',
    ]
