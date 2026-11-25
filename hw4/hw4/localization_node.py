"""Record a manual fixed-position rotation using the navigation localization pipeline."""

import math
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from hw4.config import load_validated_environment
from hw4.localization_log import LocalizationLog
from hw4.localization_report import write_report
from hw4.ros_localization import PoseSource
from hw4.settings import load_settings


class LocalizationNode(Node):
    """Observe tags only; create no motor publishers, navigator, or motion services."""

    def __init__(self):
        """Use the configured active goal unless explicit reference coordinates are supplied."""
        # share localization inputs with normal navigation
        super().__init__('hw4_localization_test')
        environment, settings, paths, reference = self.setup_inputs()

        # record feedback without a navigation controller
        output = self.declare_parameter('output_dir', '').value
        self.log = LocalizationLog(output, paths, reference)
        self.poses = PoseSource(self, environment, settings, self.log)
        self.timer = self.create_timer(0.05, self.tick)
        self.log.status('recording', 'manual rotation localization test', reference_xy_m=reference)
        self.get_logger().info(f'Recording at reference {reference} m. Rotate by hand; Ctrl+C finishes. Output: {output}')

    def setup_inputs(self):
        """Load shared settings and validate the user-supplied reference in world meters."""
        # retain independently supplied physical reference coordinates
        paths = {name: self.declare_parameter(parameter, '').value for name, parameter in [
            ('environment.json', 'environment_file'), ('navigation.yaml', 'navigation_file')]}
        environment = load_validated_environment(paths['environment.json'])
        settings = load_settings(paths['navigation.yaml'])
        reference = tuple(float(self.declare_parameter(f'reference_{axis}_m', '').value
                                or environment['goal'][f'{axis}_m']) for axis in ('x', 'y'))
        if not all(math.isfinite(v) and 0 <= v <= environment['workspace'][bound]
                   for v, bound in zip(reference, ('width_m', 'height_m'))):
            raise ValueError('Reference center must be finite and inside the measured workspace.')

        return environment, settings, paths, reference

    def tick(self):
        """Process timestamped detections and periodically show recording progress."""
        self.poses.update(time.monotonic())
        self.get_logger().info(
            f'Recorded {self.log.capture_count} images; {self.log.usable_count} with usable poses.',
            throttle_duration_sec=2.0)

    def shutdown(self):
        """Flush recordings and generate the per-tag position report."""
        self.log.status('recorded', 'manual rotation ended; physical coverage not automatically verified')
        self.log.close()
        write_report(self.log.directory)


def main(args=None):
    """Run the motor-free recorder until the user ends the manual sweep."""
    # close logs on normal launch shutdown
    rclpy.init(args=args)
    node = None
    try:
        node = LocalizationNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.shutdown()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
