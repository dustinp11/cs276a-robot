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
        self.left_linear_deadzone = 0.20
        self.left_linear_slope = 2.3 
        self.right_linear_deadzone = 0.20
        self.right_linear_slope = 2.3 
        
        self.left_angular_deadzone = 0.225
        self.left_angular_slope = 15.0
        self.right_angular_deadzone = 0.22
        self.right_angular_slope = 15.0

        # Calibrated parameters: turn in place using motor input [-0.35, 0.35] (CCW) and [0.35, -0.35] (CW)
        # Measured 360 deg turn times: CCW = 5.0s (1.2566 rad/s), CW = 4.8s (1.3090 rad/s)
        self.ccw_turn_left_cmd = -0.35
        self.ccw_turn_right_cmd = 0.35
        self.cw_turn_left_cmd = 0.35
        self.cw_turn_right_cmd = -0.35

        self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.motor_pub = self.create_publisher(Float32MultiArray, '/motor_commands', 10)
        
    def _wheel_velocities(self, linear: float, angular: float) -> tuple[float, float]:
        half_b = 0.5 * self.wheel_base
        v_left  = linear - half_b * angular
        v_right = linear + half_b * angular
        return v_left, v_right
    
    def _map_with_deadzone(self, value: float, deadzone: float, slope: float) -> float:
        cmd_mag = deadzone + abs(value) / max(slope, 1e-9)
        return np.copysign(cmd_mag, value)

    def cmd_vel_callback(self, msg: Twist):
        lin = float(msg.linear.x)
        ang = float(msg.angular.z)
        
        
        v_left, v_right = self._wheel_velocities(lin, ang)
        if lin != 0.0:
            left_cmd = self._map_with_deadzone(v_left, self.left_linear_deadzone, self.left_linear_slope)
            right_cmd = self._map_with_deadzone(v_right, self.right_linear_deadzone, self.right_linear_slope)
        elif ang != 0.0:
            if ang > 0.0:
                # CCW turn in place: calibrated [-0.35, 0.35] (360 deg in 5.0s)
                left_cmd = self.ccw_turn_left_cmd
                right_cmd = self.ccw_turn_right_cmd
            else:
                # CW turn in place: calibrated [0.35, -0.35] (360 deg in 4.8s)
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
