#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray
import numpy as np


class VelocityToMotorNode(Node):
    def __init__(self):
        super().__init__('velocity_to_motor_node')
        self.wheel_base = 0.127  # [m]

        self.cmd_max = 1.5

        # Calibrated parameters accounting for power discrepancy between left and right wheels:
        # Linear movement: motor input [0.18, 0.15] produces 0.20 m/s
        self.nominal_linear_velocity = 0.20  # [m/s]
        self.forward_left_cmd = 0.18
        self.forward_right_cmd = 0.15

        # Angular movement: 90 deg/s (1.5708 rad/s)
        # CCW turn in place: motor input [-0.40, 0.37]
        # CW turn in place: motor input [0.37, -0.37]
        self.ccw_turn_left_cmd = -0.40
        self.ccw_turn_right_cmd = 0.37
        self.cw_turn_left_cmd = 0.37
        self.cw_turn_right_cmd = -0.37

        self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.motor_pub = self.create_publisher(Float32MultiArray, '/motor_commands', 10)
        
    def _wheel_velocities(self, linear: float, angular: float) -> tuple[float, float]:
        half_b = 0.5 * self.wheel_base
        v_left  = linear - half_b * angular
        v_right = linear + half_b * angular
        return v_left, v_right

    def cmd_vel_callback(self, msg: Twist):
        lin = float(msg.linear.x)
        ang = float(msg.angular.z)
        
        v_left, v_right = self._wheel_velocities(lin, ang)
        if lin != 0.0:
            # Scale motor commands based on calibrated [0.18, 0.15] at 0.20 m/s
            scale_left = self.forward_left_cmd / self.nominal_linear_velocity
            scale_right = self.forward_right_cmd / self.nominal_linear_velocity
            left_cmd = float(np.clip(scale_left * v_left, -self.cmd_max, self.cmd_max))
            right_cmd = float(np.clip(scale_right * v_right, -self.cmd_max, self.cmd_max))
        elif ang != 0.0:
            if ang > 0.0:
                # CCW turn in place: calibrated [-0.40, 0.37] (90 deg/s)
                left_cmd = self.ccw_turn_left_cmd
                right_cmd = self.ccw_turn_right_cmd
            else:
                # CW turn in place: calibrated [0.37, -0.37] (90 deg/s)
                left_cmd = self.cw_turn_left_cmd
                right_cmd = self.cw_turn_right_cmd
        else:
            left_cmd = 0.0
            right_cmd = 0.0
            
        out = Float32MultiArray()
        out.data = [left_cmd, right_cmd]
        self.motor_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = VelocityToMotorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Stopped by keyboard interrupt')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
