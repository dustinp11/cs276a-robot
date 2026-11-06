"""Existing straight-motion calibration and nominal pulse-density turn model."""

import math

MOTOR_DEFAULTS = {
    'linear_deadzone': 0.08, 'linear_slope': 1.772915625,
    'turn_motor_input': 0.5, 'turn_rate_ccw': 1.6755160819145563,
    'turn_rate_cw': 1.6755160819145563, 'command_timeout_s': 0.2,
    'max_linear_m_s': 0.2,
}


class MotorModel:
    """Translate body requests into bounded motor inputs and matching nominal velocities."""

    def __init__(self, values):
        """Validate parameters inherited from HW2; no wheel feedback is implied."""
        self.values = dict(values)
        if any(not math.isfinite(value) or value <= 0 for value in self.values.values()):
            raise ValueError('Motor model values must be finite and positive')
        if self.values['linear_deadzone'] >= 0.5 or self.values['turn_motor_input'] > 0.5:
            raise ValueError('Motor inputs must respect the driver limit of 0.5')
        self.accumulator, self.turn_sign = 0., 0

    def map_command(self, linear, angular):
        """Return left/right inputs and nominal m/s and rad/s; allow separate drive/turn."""
        if not all(math.isfinite(value) for value in (linear, angular)) or (linear and angular):
            linear, angular = 0., 0.
        if linear:
            self.accumulator, self.turn_sign = 0., 0
            return self._straight_command(linear)
        if angular:
            return self._turn_command(angular)
        self.accumulator, self.turn_sign = 0., 0
        return 0., 0., 0., 0.

    def _straight_command(self, linear):
        """Apply the inherited straight-motion deadzone/slope and motor saturation."""
        speed = min(abs(linear), self.values['max_linear_m_s'])
        command = min(0.5, self.values['linear_deadzone'] + speed/self.values['linear_slope'])
        applied = (command-self.values['linear_deadzone'])*self.values['linear_slope']
        sign = math.copysign(1., linear)
        return sign*command, sign*command, sign*applied, 0.

    def _turn_command(self, angular):
        """Pulse full nominal turns to achieve lower average requested yaw rates."""
        sign = 1 if angular > 0 else -1
        if sign != self.turn_sign:
            self.accumulator = 0.
        self.turn_sign = sign
        full_rate = self.values['turn_rate_ccw' if sign > 0 else 'turn_rate_cw']
        self.accumulator += min(1., abs(angular)/full_rate)
        if self.accumulator < 1.:
            return 0., 0., 0., 0.
        self.accumulator -= 1.
        command = sign*self.values['turn_motor_input']
        return -command, command, 0., sign*full_rate
