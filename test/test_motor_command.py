"""Check motor timing without connecting to robot hardware."""

from unittest.mock import Mock, patch

from robot_testing.robot_test import RobotTestNode


def create_node():
    """Create a node with a mocked motor publisher."""
    node = RobotTestNode.__new__(RobotTestNode)
    node.publisher = Mock()
    node.publisher.get_subscription_count.return_value = 1
    node.get_logger = Mock()
    node.start_time = None
    node.finished = False
    return node


def tick(node, now):
    """Invoke one timer callback at a chosen wall-clock time."""
    with patch('robot_testing.robot_test.time.monotonic', return_value=now):
        node.publish_motor_command()
    return node.publisher.publish.call_args.args[0].data


def test_runs_for_ten_seconds_then_stays_stopped():
    """Check the command before, at, and after the deadline."""
    node = create_node()
    node.publisher.get_subscription_count.return_value = 0
    assert tick(node, 100.0) == [0.0, 0.0]
    assert node.start_time is None
    node.publisher.get_subscription_count.return_value = 1
    assert tick(node, 110.0) == [-0.3, 0.3]
    assert tick(node, 119.999) == [-0.3, 0.3]
    assert tick(node, 120.0) == [0.0, 0.0]
    assert tick(node, 130.0) == [0.0, 0.0]


def test_subscriber_loss_stops_without_restarting():
    """Abort a running test if the motor controller disconnects."""
    node = create_node()
    assert tick(node, 100.0) == [-0.3, 0.3]
    node.publisher.get_subscription_count.return_value = 0
    assert tick(node, 101.0) == [0.0, 0.0]
    node.publisher.get_subscription_count.return_value = 1
    assert tick(node, 102.0) == [0.0, 0.0]
