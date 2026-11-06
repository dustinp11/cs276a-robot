"""Our HW2 wheel-to-motor equations with adjustable turning input for the current surface."""

import math

MOTOR_DEFAULTS = {
    'wheel_base_m': 0.127, 'linear_deadzone': 0.08, 'linear_slope': 1.772915625,
    'turn_motor_input': 0.5, 'angular_slope': 1e6, 'motor_command_limit': 0.5,
    'turn_rate_ccw': 1.6755160819145563, 'turn_rate_cw': 1.6755160819145563,
    'command_timeout_s': 0.2,
}


class MotorModel:
    """Preserve our differential-drive mapping; report nominal motion separately."""

    def __init__(self, values):
        """Use the confirmed motor input 0.5 and 3.75-second full-turn timing."""
        self.values = dict(values)
        if any(not math.isfinite(value) or value <= 0 for value in self.values.values()):
            raise ValueError('Motor model values must be finite and positive')
        if self.values['turn_motor_input'] > self.values['motor_command_limit']:
            raise ValueError('turn_motor_input exceeds motor_command_limit')
        if self.values['linear_deadzone'] >= self.values['motor_command_limit']:
            raise ValueError('linear_deadzone must be below motor_command_limit')

    def map_command(self, linear, angular):
        """Return left/right inputs and nominal m/s and rad/s using our original equations."""
        if not all(math.isfinite(value) for value in (linear, angular)):
            return 0., 0., 0., 0.
        if linear == 0. and angular == 0.:
            return 0., 0., 0., 0.
        half_base = self.values['wheel_base_m']/2.
        wheels = (linear-half_base*angular, linear+half_base*angular)
        deadzone = self.values['linear_deadzone'] if linear else self.values['turn_motor_input']
        slope = self.values['linear_slope'] if linear else self.values['angular_slope']
        commands = [math.copysign(deadzone+abs(speed)/slope, speed) for speed in wheels]
        nominal = self._nominal_motion(commands, bool(linear))
        return *commands, *nominal

    def _nominal_motion(self, commands, linear_mode):
        """Estimate motion after driver clipping; fixed-input turns do not track requested rate."""
        limit = self.values['motor_command_limit']
        left, right = [max(-limit, min(limit, value)) for value in commands]
        if not linear_mode:
            sign = 1. if right > left else -1.
            return 0., sign*self.values['turn_rate_ccw' if sign > 0 else 'turn_rate_cw']
        velocities = [math.copysign(max(0., abs(value)-self.values['linear_deadzone'])*self.values['linear_slope'], value)
                      for value in (left, right)]
        return sum(velocities)/2., (velocities[1]-velocities[0])/self.values['wheel_base_m']
