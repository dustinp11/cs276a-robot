"""Run 1: known-map planning and stopped-feedback navigation through HW3 motors."""

from dataclasses import asdict, replace
import json
import math
import time
import traceback

import rclpy
from geometry_msgs.msg import Twist
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

from hw3.turn_calibration import read_forward_calibration, read_turn_calibration
from hw4.config import load_validated_environment
from hw4.planner import get_mode
from hw4.preview import prepare_plans
from hw4.results import RunLog, save_plans
from hw4.ros_localization import PoseSource
from hw4.settings import load_settings


class MissionNode(Node):
    """Keep ROS callbacks responsive while a timer executes one bounded mission."""

    def __init__(self):
        """Validate inputs, preserve calibration, and remain idle until explicitly started."""
        # share effective calibration before accepting missions
        super().__init__('hw4_navigation')
        self.forward = read_forward_calibration(self)
        self.turn = read_turn_calibration(self)
        if self.turn is None:
            raise ValueError('HW4 requires enabled HW3 turn calibration.')
        self.setup_inputs()
        self.setup_interfaces()
        self.get_logger().info('Run 1 ready. Start with /hw4/start; stop with /hw4/stop.')

    def setup_inputs(self):
        """Archive effective configs before accepting mission commands."""
        # archive exactly what this session uses
        paths = {name: self.declare_parameter(parameter, '').value for name, parameter in [
            ('environment.json', 'environment_file'), ('motion.yaml', 'motion_file'),
            ('navigation.yaml', 'navigation_file')]}
        self.environment = load_validated_environment(paths['environment.json'])
        self.settings = load_settings(paths['navigation.yaml'])
        self.policy = self.settings['navigation']
        output = self.declare_parameter('output_dir', '').value
        self.log = RunLog(output, paths)
        self.log.status('ready', 'awaiting explicit start', mode=self.policy['mode'],
                        forward=asdict(self.forward), turn=asdict(self.turn))

    def setup_interfaces(self):
        """Create fixed services and the 20 Hz controller with no startup motion."""
        # keep callbacks responsive without automatic movement
        self.velocity = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/hw4/status', 10)
        self.poses = PoseSource(self, self.environment, self.settings, self.log)
        self.navigator, self.used, self.last_status = None, False, None
        self.start_service = self.create_service(Trigger, '/hw4/start', self.start)
        self.stop_service = self.create_service(Trigger, '/hw4/stop', self.stop)
        self.timer = self.create_timer(0.05, self.tick)

    def start(self, request, response):
        """Plan and execute the configured mode once per result session."""
        # prevent restarts and competing motor publishers
        if self.used:
            response.success, response.message = False, 'This session already started; launch a new result session.'
            return response
        if self.count_publishers('/cmd_vel') != 1 or self.count_publishers('/motor_commands') != 1:
            response.success, response.message = False, 'Expected exactly one command publisher and one HW3 motor mapper.'
            return response
        self.used = True
        self.publish_command((0.0, 0.0))

        # freeze plans before beginning any motion
        try:
            grid, plans = prepare_plans(self.environment, self.policy)
            save_plans(self.log.directory, grid, self.environment, plans)
            control_grid = replace(grid, margin=self.environment['robot']['safety_margin_m'])
            mode = get_mode(self.policy['mode'])
            self.navigator = mode.create_navigator(control_grid, plans[self.policy['mode']], self.forward,
                                       self.turn, self.policy, time.monotonic(),
                                       math.radians(self.environment['start']['yaw_deg']))
            self.log.status('started', 'plans saved; waiting for measured pose', mode=self.policy['mode'])
            response.success, response.message = True, 'Run 1 accepted; acquiring pose before motion.'
        except Exception as error:
            if self.navigator is not None:
                self.navigator.finish('failed', str(error))
            self.publish_command((0.0, 0.0))
            self.log.status('failed', str(error), traceback=traceback.format_exc())
            response.success, response.message = False, str(error)
        return response

    def stop(self, request, response):
        """Cancel the current mission without advancing its target or allowing restart."""
        # cancel without claiming a measured arrival
        self.publish_command((0.0, 0.0))
        if self.navigator is not None and self.navigator.finished:
            response.success, response.message = True, f'Stop sent; result remains {self.navigator.state}.'
            return response
        self.used = True
        if self.navigator is not None:
            self.navigator.finish('stopped', 'user requested stop')
        self.log.status('stopped', 'user requested stop')
        response.success, response.message = True, 'Stop requested.'
        return response

    def tick(self):
        """Process visual feedback and publish bounded commands without blocking waits."""
        # stop immediately if a callback fails
        try:
            now = time.monotonic()
            self.poses.update(now)
            if self.navigator is not None:
                command = self.navigator.tick(now, self.poses.samples)
                self.publish_command(command)
                self.record_tick(now, command)
        except Exception as error:
            self.used = True
            self.timer.cancel()
            self.publish_command((0.0, 0.0))
            if self.navigator is not None:
                self.navigator.finish('failed', str(error))
            self.log.status('failed', str(error), traceback=traceback.format_exc())
            self.get_logger().error(str(error))

    def publish_command(self, command):
        """Send separate forward or pivot requests through the calibrated HW3 mapper."""
        message = Twist()
        message.linear.x, message.angular.z = command
        self.velocity.publish(message)

    def record_tick(self, now, command):
        """Record measured feedback separately from command-predicted pose and state."""
        # keep observations separate from command predictions
        nav = self.navigator
        sample = self.poses.samples[-1] if self.poses.samples else None
        measured = (sample.x, sample.y, sample.yaw) if sample else ('', '', '')
        age = now - sample.stamp if sample else ''
        self.log.commands.writerow([now, nav.state, nav.index, nav.reason, *command, age,
                                   *measured, *(nav.expected or ('', '', '')),
                                   nav.corrections, nav.commanded_distance, nav.segment_budget,
                                   nav.grid.clearance(sample.xy) if sample else '',
                                   nav.turn_target_yaw, nav.turn_progress])
        status = nav.state, nav.reason, nav.index
        if status == self.last_status:
            return

        # persist each navigation phase transition once
        self.last_status = status
        details = {'waypoint': nav.index, 'elapsed_s': now - nav.started,
                   'goal_reached': nav.state == 'complete', 'pose': measured,
                   'pose_age_s': age, 'mode': self.policy['mode'],
                   'accepted_pose': (nav.measured.x, nav.measured.y, nav.measured.yaw) if nav.measured else None,
                   'commanded_segment_distance_m': nav.commanded_distance,
                   'segment_budget_m': nav.segment_budget}
        self.log.status(nav.state, nav.reason, **details)
        self.status_pub.publish(String(data=json.dumps({'state': nav.state, 'reason': nav.reason, **details})))
        self.get_logger().info(f'{nav.state}: {nav.reason}; waypoint {nav.index}')

    def shutdown(self):
        """Stop an unfinished mission, preserve its failure state, and close the log."""
        # preserve explicit outcome when launch ends
        if rclpy.ok():
            self.publish_command((0.0, 0.0))
        if self.navigator is not None and not self.navigator.finished:
            self.navigator.finish('stopped', 'launch ended before measured arrival')
            self.log.status('stopped', self.navigator.reason, goal_reached=False)
        self.log.close()


def main(args=None):
    """Run the mission node; the motor driver retains its independent watchdog."""
    # always clean up when execution ends
    rclpy.init(args=args)
    node = None
    try:
        node = MissionNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.shutdown()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
