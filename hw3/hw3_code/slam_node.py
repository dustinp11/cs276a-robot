"""Run live, delayed-vision EKF-SLAM and a 7 ft square without a known landmark map."""

import math
import time

import rclpy
from apriltag_msgs.msg import AprilTagDetectionArray
from geometry_msgs.msg import Twist, TwistStamped
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Float32MultiArray, String
from std_srvs.srv import Trigger

from .ekf_slam import EkfSlam
from .motion import MotionHistory
from .observations import ObservationAdapter
from .recording import RunRecorder
from .settings import DEFAULTS, validate_parameters
from .visualization import MapPublisher
from .web_view import BrowserMap
from .waypoint_control import ControlSettings, FixedSquareController


class SlamNode(Node):
    """Keep estimation, control, logging, and visualizations on one chronological executor."""

    def __init__(self):
        """Load settings, construct an empty-map estimator, and connect the live pipeline."""
        super().__init__('hw3_slam')
        self.values = {key: self.declare_parameter(key, value).value for key, value in DEFAULTS.items()}
        validate_parameters(self.values)
        self._create_estimation()
        self._create_state()
        self.recorder = RunRecorder(self.values['output_directory'], self.values)
        self.visualization = MapPublisher(self, self.values['frame_id'])
        self.browser = BrowserMap(self.values['web_host'], self.values['web_port'])
        self.get_logger().info(f"Browser map: http://ROBOT_IP:{self.values['web_port']}/")
        self._create_interfaces()
        self.timer = self.create_timer(self.controller.settings.dt, self.tick)
        self.get_logger().info(f"Square side={self.values['square_side_ft']} ft; logs={self.recorder.directory}; empty map, Standard41h12")

    def _create_estimation(self):
        """Configure the EKF, raw-camera observations, and command-motion uncertainty."""
        v = self.values
        self.estimator = EkfSlam(gate_threshold=v['gate_threshold'])
        self.adapter = ObservationAdapter(v['tag_size_m'], v['camera_xyz_m'], v['camera_rpy_deg'],
                                          v['range_std_m'], v['range_scale'], v['bearing_std_rad'],
                                          v['max_reprojection_px'])
        self.motion = MotionHistory(v['distance_noise'], v['turn_noise'], v['idle_distance_noise'],
                                    v['idle_turn_noise'], v['command_hold_s'])
        settings = ControlSettings(max_speed=v['max_speed_m_s'], rotation_speed=v['max_turn_rad_s'])
        self.controller = FixedSquareController(v['square_side_ft'], v['laps'], settings)

    def _create_state(self):
        """Initialize lifecycle and observation queues without assuming a landmark count."""
        self.pending, self.estimator_time, self.last_frame = [], None, None
        self.last_vision, self.seen_frames, self.skipped_frames = None, 0, 0
        self.armed, self.fault = self.values['auto_start'], None
        self.started_wall, self.last_tick_wall = time.monotonic(), time.monotonic()
        self.last_ros_time, self.last_publish, self.last_lap = None, 0., 0
        self.reason = 'waiting for motor motion stream'
        self.logged_reason = None
        self.vision_problem = None
        self.completed_recorded = False
        self.route_progress, self.route_time = [0., 0.], None
        self.last_phase = None

    def _create_interfaces(self):
        """Subscribe to HW3-only inputs and expose command, status, start, and stop interfaces."""
        self.command_pub = self.create_publisher(Twist, '/hw3/cmd_vel', 1)
        self.status_pub = self.create_publisher(String, '/hw3/status', 1)
        self.camera_sub = self.create_subscription(CameraInfo, '/hw3/camera_info', self.on_camera, qos_profile_sensor_data)
        self.detection_sub = self.create_subscription(AprilTagDetectionArray, '/hw3/detections', self.on_detections, 10)
        self.motor_sub = self.create_subscription(Float32MultiArray, '/hw3/motor_commands', self.on_motor_commands, 50)
        self.motion_sub = self.create_subscription(TwistStamped, '/hw3/applied_motion', self.on_motion, 50)
        self.start_service = self.create_service(Trigger, '/hw3/start', self.on_start)
        self.stop_service = self.create_service(Trigger, '/hw3/stop', self.on_stop)

    def now_seconds(self):
        """Return ROS time as seconds for sensor/command timestamp comparisons."""
        return self.get_clock().now().nanoseconds / 1e9

    def on_camera(self, message):
        """Use raw pinhole CameraInfo only at the configured calibration resolution."""
        if (message.width, message.height) != (self.values['image_width'], self.values['image_height']):
            self._vision_warning('Camera resolution differs from configured calibration resolution')
            return
        if message.distortion_model not in ('plumb_bob', 'rational_polynomial'):
            self._vision_warning('Unsupported raw camera distortion model')
            return
        try:
            first = self.adapter.matrix is None
            self.vision_problem = None
            self.adapter.set_camera(message.k, message.d, message.header.frame_id)
            if first:
                self.recorder.write('camera_calibration', self.now_seconds(), frame=message.header.frame_id,
                                    width=message.width, height=message.height, matrix=list(message.k),
                                    distortion=list(message.d), distortion_model=message.distortion_model)
        except ValueError as error:
            self._vision_warning(str(error))

    def _vision_warning(self, reason):
        """Disable invalid camera observations without preventing the fixed square drive."""
        self.adapter.matrix = None
        if reason != self.vision_problem:
            self.vision_problem = reason
            self.get_logger().warning(f'Mapping unavailable: {reason}; fixed square drive is independent')

    def on_motion(self, message):
        """Record the mapper's applied-model velocity, never treating it as encoder data."""
        stamp = Time.from_msg(message.header.stamp).nanoseconds / 1e9
        if stamp > self.now_seconds() + 0.1:
            self.fail('Motion timestamp is in the future')
            return
        if self.motion.add(stamp, message.twist.linear.x, message.twist.angular.z):
            if self.estimator_time is None:
                self.estimator_time = stamp
                self.route_time = stamp
                self.recorder.write('origin', stamp, frame=self.values['frame_id'])
            self.recorder.write('applied_motion', stamp, linear=message.twist.linear.x, angular=message.twist.angular.z)

    def on_motor_commands(self, message):
        """Log emitted wheel inputs so a skipped turn can be distinguished from wheel stall."""
        if len(message.data) >= 2 and all(math.isfinite(value) for value in message.data[:2]):
            self.recorder.write('motor_commands', self.now_seconds(),
                                left=float(message.data[0]), right=float(message.data[1]))

    def on_detections(self, message):
        """Convert each fresh Standard41h12 frame once and queue it by capture timestamp."""
        stamp = Time.from_msg(message.header.stamp).nanoseconds / 1e9
        if self.adapter.matrix is None or message.header.frame_id != self.adapter.frame:
            return
        if stamp > self.now_seconds()+0.1 or (self.last_frame is not None and stamp <= self.last_frame):
            self.skipped_frames += 1
            return
        self.last_frame = stamp
        observations = self._convert_frame(message)
        if observations:
            self.pending.append((stamp, observations))
            if len(self.pending) > 100:
                self.fail('Observation queue exceeded its bound')
                self.pending.clear()

    def _convert_frame(self, message):
        """Reject bad decodes, poor geometry, and duplicate IDs within a camera frame."""
        observations, seen = [], set()
        for detection in message.detections:
            family = detection.family.removeprefix('tag').removeprefix('Standard')
            if family != '41h12' or detection.id in seen:
                continue
            if not 0 <= detection.hamming <= self.values['max_hamming'] or detection.decision_margin < self.values['min_decision_margin']:
                continue
            seen.add(detection.id)
            item = self._convert_detection(detection)
            if item is not None:
                observations.append(item)
        return observations

    def _convert_detection(self, detection):
        """Run calibrated raw-image PnP and enforce finite quality/range limits."""
        try:
            corners = [[point.x, point.y] for point in detection.corners]
            distance, bearing, covariance, error = self.adapter.convert(corners)
            if distance > self.values['max_range_m'] or not math.isfinite(detection.decision_margin):
                return None
            return (('41h12', int(detection.id)), distance, bearing, covariance, error)
        except (ValueError, RuntimeError) as error:
            self.get_logger().debug(f'Rejected detection: {error}')
            return None

    def tick(self):
        """Advance delayed estimates, predict a current control pose, and emit bounded commands."""
        now, wall = self.now_seconds(), time.monotonic()
        if wall-self.last_tick_wall > 0.5 or (self.last_ros_time is not None and now < self.last_ros_time):
            self.fail('Control timing gap or ROS clock reset; restart this run')
        self.last_tick_wall, self.last_ros_time = wall, now
        try:
            self._advance_route_motion(now)
            self._advance_estimation(now)
            command = self._control(now, wall)
            self.send_command(*command)
            self._publish_results(now)
        except Exception as error:
            self.fail(f'Estimation/control error: {error}')
            self.send_command(0., 0.)

    def _advance_route_motion(self, now):
        """Accumulate dispatched command motion without incorporating EKF pose corrections."""
        if self.route_time is None:
            return
        distance, turn = self.motion.totals(self.route_time, now)
        self.route_progress[0] += distance
        self.route_progress[1] += turn
        self.route_time = now

    def _log_phase(self, now):
        """Record every corner transition with independent drive/turn progress."""
        phase = (self.controller.index, self.controller.stage)
        if phase != self.last_phase:
            self.last_phase = phase
            self.recorder.write('route_phase', now, side=self.controller.index+1,
                                stage=self.controller.stage, progress=self.route_progress.copy())
            self.get_logger().info(f'Square: {self.reason}; command motion={self.route_progress}')

    def _advance_estimation(self, now):
        """Apply observations at capture time, then advance to the fixed-delay horizon."""
        if self.estimator_time is None:
            return
        target = now-self.values['vision_delay_s']
        self.pending.sort(key=lambda item: item[0])
        while self.pending and self.pending[0][0] <= target:
            stamp, observations = self.pending.pop(0)
            if stamp < self.estimator_time:
                self.skipped_frames += 1
                continue
            self._predict_to(stamp)
            self._apply_frame(stamp, observations)
        if target >= self.estimator_time:
            self._predict_to(target)
            self.motion.prune(self.estimator_time)

    def _predict_to(self, stamp):
        """Predict the persistent estimator through chronological applied-command history."""
        self.motion.propagate(self.estimator, self.estimator_time, stamp)
        self.estimator_time = stamp

    def _apply_frame(self, stamp, observations):
        """Update/initialize map entries once and log the observation and gating decision."""
        accepted = False
        for key, distance, bearing, covariance, error in observations:
            result = self.estimator.observe(key, distance, bearing, covariance)
            accepted |= result != 'rejected'
            self.recorder.write('observation', stamp, landmark_id=list(key), range_m=distance,
                                bearing_rad=bearing, covariance=covariance.tolist(), reprojection_px=error, result=result)
        if accepted:
            self.last_vision = stamp
            self.seen_frames += 1

    def _control(self, now, wall):
        """Execute the fixed route from command motion; EKF corrections only affect the map."""
        reason = self._pause_reason(now, wall)
        if reason:
            self.reason = reason
            return 0., 0.
        command = self.controller.step(self.route_progress)
        self.reason = ('complete' if self.controller.done else
                       f'{self.controller.stage}; side={self.controller.index+1}')
        self._log_phase(now)
        self._record_progress(now)
        return command

    def _pause_reason(self, now, wall):
        """Require motor communication and startup readiness; vision never gates the route."""
        if self.fault:
            return self.fault
        if self.controller.done:
            return 'complete'
        if wall-self.started_wall > self.values['max_run_s']:
            self.fail('Maximum run duration exceeded')
            return self.fault
        if not self.armed:
            return 'paused by operator'
        if not self.motion.samples or not 0 <= now-self.motion.samples[-1][0] <= self.values['motion_timeout_s']:
            return 'waiting for applied-motion heartbeat'
        if self.estimator_time is None or wall-self.started_wall < self.values['startup_wait_s']:
            return 'waiting for startup'
        return None

    def _record_progress(self, now):
        """Save full map/covariance snapshots at each completed lap and at mission completion."""
        lap = self.controller.index // 4
        if lap > self.last_lap:
            self.last_lap = lap
            self.recorder.snapshot(self.estimator, self.estimator_time, lap=lap, event='lap_complete')
            self.get_logger().info(f'Completed square lap {lap}')
        if self.controller.done and not self.completed_recorded:
            self.completed_recorded = True
            self.recorder.write('complete', now, laps=lap)
            self.get_logger().info('Square complete; stopped. Mapping/logging remains active until shutdown.')

    def _publish_results(self, now):
        """Record/post visual estimates at 2 Hz, with a status heartbeat on the same cadence."""
        if now-self.last_publish < 0.5:
            return
        self.last_publish = now
        if self.estimator_time is not None:
            self.recorder.snapshot(self.estimator, self.estimator_time, lap=self.last_lap, event='periodic')
            self.visualization.publish(self.estimator, Time(seconds=self.estimator_time).to_msg())
            age = None if self.last_vision is None else max(0., now-self.last_vision)
            self.browser.update(self.estimator, self.estimator_time, self.reason, self.last_lap, age, self.skipped_frames)
        status = String()
        status.data = f'{self.reason}; landmarks={len(self.estimator.landmark_ids)}; lap={self.last_lap}; skipped_frames={self.skipped_frames}'
        self.status_pub.publish(status)
        if self.reason != self.logged_reason:
            self.get_logger().info(status.data)
            self.logged_reason = self.reason

    def send_command(self, linear, angular):
        """Publish an HW3-only velocity command and log its ROS timestamp."""
        message = Twist()
        message.linear.x, message.angular.z = float(linear), float(angular)
        self.command_pub.publish(message)
        self.recorder.write('command', self.now_seconds(), linear=float(linear), angular=float(angular),
                            stage=self.controller.stage, side=self.controller.index+1)

    def fail(self, reason):
        """Latch a fault so a numerical/timing failure cannot silently resume motion."""
        if self.fault is None:
            self.fault = reason
            self.get_logger().error(reason)
            self.recorder.write('fault', self.now_seconds(), reason=reason)
        self.armed = False

    def on_start(self, request, response):
        """Resume a paused mission without resetting the map or local reference frame."""
        response.success = self.fault is None and not self.controller.done
        self.armed = response.success
        response.message = 'Armed; waiting for valid inputs' if response.success else 'Faulted or complete; start a new run'
        return response

    def on_stop(self, request, response):
        """Pause immediately while continuing to estimate, log, and visualize."""
        self.armed = False
        self.send_command(0., 0.)
        response.success, response.message = True, 'Paused'
        return response

    def destroy_node(self):
        """Record the final estimate and close the log, sending zero if ROS is active."""
        if rclpy.ok():
            self.send_command(0., 0.)
        if self.estimator_time is not None:
            self.recorder.snapshot(self.estimator, self.estimator_time, lap=self.last_lap, event='shutdown')
        self.recorder.close()
        self.browser.close()
        super().destroy_node()


def main(args=None):
    """Run the live square SLAM node in a single-threaded ROS executor."""
    rclpy.init(args=args)
    node = None
    try:
        node = SlamNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
