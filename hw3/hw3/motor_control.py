"""Motor control, velocity mapping, turn and forward calibration, and serial motor driver.

Consolidates:
- ForwardCalibration & TurnCalibration (parameters and measured limits)
- VelocityMapper & MotorCommand (translating cmd_vel to calibrated motor inputs)
- VelocityToMotorNode (ROS 2 node publishing /motor_commands and /cmd_vel_applied)
- MotorControllerNode (ROS 2 serial motor driver with watchdog and IMU publication)
"""

from dataclasses import asdict, dataclass, fields
import json
import math
import threading
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray
import serial


# -----------------------------------------------------------------------------
# Calibration Data Structures & Helpers
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class TurnCalibration:
    """Signed [left, right] turn inputs and positive nominal rates in rad/s."""

    ccw_left_input: float = -0.40
    ccw_right_input: float = 0.37
    cw_left_input: float = 0.37
    cw_right_input: float = -0.37
    ccw_radps: float = 0.0
    cw_radps: float = 0.0

    def __post_init__(self):
        """Validate signed pivot pairs and direction-specific measured rates."""
        motors = (self.ccw_left_input, self.ccw_right_input,
                  self.cw_left_input, self.cw_right_input)
        if not all(math.isfinite(value) and 0 < abs(value) <= 0.5 for value in motors):
            raise ValueError('Turn motor inputs must be finite, nonzero, and within [-0.5, 0.5].')
        if not (self.ccw_left_input < 0 < self.ccw_right_input and self.cw_right_input < 0 < self.cw_left_input):
            raise ValueError('CCW requires [-left, +right]; CW requires [+left, -right].')
        if not all(math.isfinite(value) and value > 0 for value in (self.ccw_radps, self.cw_radps)):
            raise ValueError('Set positive nominal rates for both CCW and CW turns.')

    def rate(self, requested_omega: float) -> float:
        """Return the achievable signed yaw rate; zero requests remain stopped."""
        if not math.isfinite(requested_omega):
            raise ValueError('Requested angular velocity must be finite.')
        if requested_omega == 0:
            return 0.0
        return self.ccw_radps if requested_omega > 0 else -self.cw_radps

    def motors(self, requested_omega: float) -> tuple[float, float]:
        """Return the exact measured pair, preserving unequal left/right magnitudes."""
        rate = self.rate(requested_omega)
        if rate > 0:
            return self.ccw_left_input, self.ccw_right_input
        if rate < 0:
            return self.cw_left_input, self.cw_right_input
        return 0.0, 0.0


def read_turn_calibration(node: Node) -> TurnCalibration | None:
    """Declare shared parameters and return calibration only when explicitly enabled."""
    defaults = {'enabled': False, **{field.name: field.default for field in fields(TurnCalibration)}}
    values = {key: node.declare_parameter(f'turn_calibration.{key}', value).value
              for key, value in defaults.items()}
    if not values.pop('enabled'):
        return None
    return TurnCalibration(**values)


@dataclass(frozen=True)
class ForwardCalibration:
    """Straight motor trim and nominal forward speed in meters/second."""

    left_input: float = 0.185
    right_input: float = 0.15
    speed_mps: float = 0.203

    def __post_init__(self):
        """Require finite forward-only inputs within the driver's range and positive speed."""
        values = (self.left_input, self.right_input, self.speed_mps)
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError('Forward motor inputs and measured speed must be positive and finite.')
        if max(self.left_input, self.right_input) > 0.5:
            raise ValueError('Forward inputs must not exceed the driver limit of 0.5.')


def read_forward_calibration(node: Node) -> ForwardCalibration:
    """Read fixed straight-motion calibration from ROS parameters."""
    defaults = asdict(ForwardCalibration())
    values = {key: node.declare_parameter(f'forward_calibration.{key}', value).value
              for key, value in defaults.items()}
    return ForwardCalibration(**values)


# -----------------------------------------------------------------------------
# Velocity Mapping Logic & Node
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class MotorCommand:
    """Left/right motor inputs and model velocities (m/s, rad/s), not odometry."""

    left: float
    right: float
    linear_mps: float
    angular_radps: float


class VelocityMapper:
    """Keep straight travel separate from calibrated fixed-rate pivoting."""

    def __init__(self, calibration: TurnCalibration, forward_calibration: ForwardCalibration = None):
        """Store independent measured forward and turning calibrations."""
        self.calibration = calibration
        self.forward = forward_calibration or ForwardCalibration()

    def map_command(self, linear_mps: float, angular_radps: float) -> MotorCommand:
        """Map a straight or pivot request; nonfinite requests produce a stop."""
        if linear_mps < 0 or not all(math.isfinite(v) for v in (linear_mps, angular_radps)):
            return MotorCommand(0.0, 0.0, 0.0, 0.0)
        if linear_mps:
            return MotorCommand(self.forward.left_input, self.forward.right_input,
                                self.forward.speed_mps, 0.0)
        left, right = self.calibration.motors(angular_radps)
        return MotorCommand(left, right, 0.0, self.calibration.rate(angular_radps))


class VelocityToMotorNode(Node):
    """Publish motor inputs and model velocity from the same mapping decision."""

    def __init__(self):
        """Initialize velocity mapping node and ROS publishers/subscriptions."""
        super().__init__('velocity_mapping')
        calibration = read_turn_calibration(self)
        if calibration is None:
            raise ValueError('HW3 mapping requires enabled, measured turn calibration.')
        self.mapper = VelocityMapper(calibration, read_forward_calibration(self))
        self.motor_pub = self.create_publisher(Float32MultiArray, '/motor_commands', 10)
        self.applied_pub = self.create_publisher(Twist, '/cmd_vel_applied', 10)
        self.subscription = self.create_subscription(Twist, '/cmd_vel', self.on_command, 10)

    def on_command(self, message: Twist) -> None:
        """Map and publish one straight, pivot, or stopped command."""
        command = self.mapper.map_command(message.linear.x, message.angular.z)
        self.motor_pub.publish(Float32MultiArray(data=[command.left, command.right]))
        applied = Twist()
        applied.linear.x = command.linear_mps
        applied.angular.z = command.angular_radps
        self.applied_pub.publish(applied)


def velocity_mapping_main(args=None):
    """Run the mapping adapter until ROS shutdown."""
    rclpy.init(args=args)
    node = VelocityToMotorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


# -----------------------------------------------------------------------------
# Serial Motor Driver Logic & Node
# -----------------------------------------------------------------------------

def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """Convert device Euler angles in degrees to quaternion ordering (w, x, y, z)."""
    r, p, y = [math.radians(value) / 2 for value in (roll, pitch, yaw)]
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)


class MotorControllerNode(Node):
    """Own the serial connection in one worker to serialize motor and IMU traffic."""

    def __init__(self):
        """Initialize serial motor controller node and background worker thread."""
        super().__init__('motor_control')
        self.robot = None
        self.stop_event = threading.Event()
        self.command_lock = threading.Lock()
        self.motor_command = {'T': 1, 'L': 0.0, 'R': 0.0}
        self.last_command_time = time.monotonic()
        self.timeout_duration = 0.15
        self.imu_publisher = self.create_publisher(Imu, '/imu_data', 10)
        self.subscription = self.create_subscription(
            Float32MultiArray, '/motor_commands', self.motor_command_callback, 10)
        self.worker = threading.Thread(target=self._serial_loop, daemon=True)
        self.worker.start()

    def motor_command_callback(self, message: Float32MultiArray):
        """Cache clipped finite [left, right] inputs, stopping on invalid messages."""
        values = list(message.data[:2])
        if len(values) != 2 or not all(math.isfinite(v) for v in values):
            values = [0.0, 0.0]
            self.get_logger().error('Invalid motor command; stopping.')
        left, right = [max(-0.5, min(0.5, float(v))) for v in values]
        with self.command_lock:
            self.motor_command = {'T': 1, 'L': left, 'R': right}
            self.last_command_time = time.monotonic()

    def _current_command(self, now: float) -> dict:
        """Return a command snapshot, replacing expired commands with zero inputs."""
        with self.command_lock:
            if now - self.last_command_time > self.timeout_duration:
                return {'T': 1, 'L': 0.0, 'R': 0.0}
            return dict(self.motor_command)

    def _connect(self) -> bool:
        """Open the existing USB serial endpoint; retry while stopped if unavailable."""
        try:
            self.robot = serial.Serial('/dev/ttyUSB0', 115200, timeout=0.01, write_timeout=0.1)
            self.get_logger().info('Motor serial connected; watchdog is 0.15 s.')
            return True
        except (serial.SerialException, OSError) as error:
            self.get_logger().error(f'Motor serial unavailable: {error}')
            self.stop_event.wait(1.0)
            return False

    def _write(self, packet: dict):
        """Send one newline-delimited device JSON packet from the serial worker."""
        self.robot.write((json.dumps(packet, allow_nan=False) + '\n').encode('utf-8'))

    def _serial_loop(self):
        """Schedule motor TX at 20 Hz, IMU requests at 50 Hz, and serial reception."""
        try:
            while not self.stop_event.is_set() and rclpy.ok():
                if self.robot is None and not self._connect():
                    continue
                self._connected_session()
        finally:
            self._close_serial()

    def _connected_session(self):
        """Exchange packets until shutdown or a serial failure requires reconnecting."""
        next_motor = next_imu = 0.0
        buffer = ''
        try:
            while not self.stop_event.is_set() and rclpy.ok():
                now = time.monotonic()
                if now >= next_motor:
                    self._write(self._current_command(now))
                    next_motor = now + 0.05
                if now >= next_imu:
                    self._write({'T': 126})
                    next_imu = now + 0.02
                buffer = self._read_lines(buffer)
        except (serial.SerialException, OSError) as error:
            self.get_logger().error(f'Motor serial error: {error}')
            self._close_serial()
            self.stop_event.wait(0.1)

    def _read_lines(self, buffer: str) -> str:
        """Read a bounded chunk and process complete JSON lines, retaining a partial line."""
        data = self.robot.read(min(max(1, self.robot.in_waiting), 4096))
        buffer += data.decode('utf-8', errors='ignore')
        while '\n' in buffer:
            line, buffer = buffer.split('\n', 1)
            self._on_device_line(line)
        return buffer[-65536:]

    def _on_device_line(self, line: str):
        """Publish only valid IMU response packets (device type 1002)."""
        try:
            packet = json.loads(line)
            if isinstance(packet, dict) and packet.get('T') == 1002:
                self.imu_publisher.publish(self._imu_message(packet))
        except (ValueError, TypeError, OverflowError):
            self.get_logger().debug('Ignoring malformed IMU response.')

    def _imu_message(self, packet: dict) -> Imu:
        """Convert device IMU fields into ROS sensor_msgs/Imu."""
        message = Imu()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = 'imu_link'
        angles = [float(packet.get(key, 0)) for key in ('r', 'p', 'y')]
        w, x, y, z = euler_to_quaternion(*angles)
        message.orientation.w, message.orientation.x = w, x
        message.orientation.y, message.orientation.z = y, z
        for axis in 'xyz':
            setattr(message.angular_velocity, axis, float(packet.get('g' + axis, 0)))
            setattr(message.linear_acceleration, axis, float(packet.get('a' + axis, 0)) * 9.81 / 1000)
        return message

    def _close_serial(self):
        """Attempt a final motor stop before closing the serial connection."""
        if self.robot is None:
            return
        try:
            self._write({'T': 1, 'L': 0.0, 'R': 0.0})
        except (serial.SerialException, OSError):
            pass
        finally:
            self.robot.close()
            self.robot = None

    def destroy_node(self):
        """Request shutdown and join serial I/O before destroying ROS publishers."""
        self.stop_event.set()
        self.worker.join()
        return super().destroy_node()


def motor_control_main(args=None):
    """Run the hardware motor controller node until ROS shutdown."""
    rclpy.init(args=args)
    node = MotorControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
