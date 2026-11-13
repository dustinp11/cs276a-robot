"""HW3 serial motor driver using the HW2 protocol, watchdog, and IMU topics."""

import json
import math
import threading
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray
import serial


def euler_to_quaternion(roll, pitch, yaw):
    """Convert device Euler angles in degrees to quaternion ordering (w, x, y, z)."""
    r, p, y = [math.radians(value) / 2 for value in (roll, pitch, yaw)]
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)


class MotorControllerNode(Node):
    """Own the serial connection in one worker to serialize motor and IMU traffic."""

    def __init__(self):
        """Initialize stopped motor state and start serial I/O after ROS setup.

        Args:
            None.

        Returns:
            None.
        """
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

    def motor_command_callback(self, message):
        """Cache clipped finite [left, right] inputs, stopping on invalid messages."""
        values = list(message.data[:2])
        if len(values) != 2 or not all(math.isfinite(v) for v in values):
            values = [0.0, 0.0]
            self.get_logger().error('Invalid motor command; stopping.')
        left, right = [max(-0.5, min(0.5, float(v))) for v in values]
        with self.command_lock:
            self.motor_command = {'T': 1, 'L': left, 'R': right}
            self.last_command_time = time.monotonic()

    def _current_command(self, now):
        """Return a command snapshot, replacing expired commands with zero inputs."""
        with self.command_lock:
            if now - self.last_command_time > self.timeout_duration:
                return {'T': 1, 'L': 0.0, 'R': 0.0}
            return dict(self.motor_command)

    def _connect(self):
        """Open the existing USB serial endpoint; retry while stopped if unavailable."""
        try:
            self.robot = serial.Serial('/dev/ttyUSB0', 115200, timeout=0.01, write_timeout=0.1)
            self.get_logger().info('Motor serial connected; watchdog is 0.15 s.')
            return True
        except (serial.SerialException, OSError) as error:
            self.get_logger().error(f'Motor serial unavailable: {error}')
            self.stop_event.wait(1.0)
            return False

    def _write(self, packet):
        """Send one newline-delimited device JSON packet from the serial worker."""
        self.robot.write((json.dumps(packet, allow_nan=False) + '\n').encode('utf-8'))

    def _serial_loop(self):
        """Schedule motor TX at 20 Hz, IMU requests at 50 Hz, and serial reception.

        Args:
            None.

        Returns:
            None. The worker stops and closes its serial connection on exit.
        """
        try:
            while not self.stop_event.is_set() and rclpy.ok():
                if self.robot is None and not self._connect():
                    continue
                self._connected_session()
        finally:
            self._close_serial()

    def _connected_session(self):
        """Exchange packets until shutdown or a serial failure requires reconnecting.

        Args:
            None.

        Returns:
            None. Commands that expire while disconnected are not replayed.
        """
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

    def _read_lines(self, buffer):
        """Read a bounded chunk and process complete JSON lines, retaining a partial line."""
        data = self.robot.read(min(max(1, self.robot.in_waiting), 4096))
        buffer += data.decode('utf-8', errors='ignore')
        while '\n' in buffer:
            line, buffer = buffer.split('\n', 1)
            self._on_device_line(line)
        return buffer[-65536:]

    def _on_device_line(self, line):
        """Publish only valid IMU response packets (device type 1002)."""
        try:
            packet = json.loads(line)
            if isinstance(packet, dict) and packet.get('T') == 1002:
                self.imu_publisher.publish(self._imu_message(packet))
        except (ValueError, TypeError, OverflowError):
            self.get_logger().debug('Ignoring malformed IMU response.')

    def _imu_message(self, packet):
        """Convert the existing device IMU fields into the unchanged ROS interface.

        Args:
            packet: Device JSON dictionary. Euler fields r/p/y are degrees;
                acceleration fields ax/ay/az retain HW2's mg-to-m/s² conversion.
                Gyro gx/gy/gz values are passed through as in HW2; their device
                units must be confirmed before using them for heading feedback.

        Returns:
            sensor_msgs.msg.Imu in imu_link. This driver does not fuse the IMU.
        """
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
        """Attempt a final motor stop before closing the serial connection.

        Args:
            None.

        Returns:
            None. State changes and publications are performed in place.
        """
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


def main(args=None):
    """Run the HW3 hardware node and stop motors during cleanup.

    Args:
        args: Optional ROS command-line arguments.

    Returns:
        None.
    """
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
