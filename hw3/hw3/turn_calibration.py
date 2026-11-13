"""Measured forward and in-place turn calibration shared by mapping and navigation."""

from dataclasses import asdict, dataclass, fields
import math


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
        """Validate signed pivot pairs and direction-specific measured rates.

        Args:
            None.

        Returns:
            None. Raises ValueError for missing rates, invalid signs, or inputs
            outside the motor driver's [-0.5, 0.5] range.
        """
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


def read_turn_calibration(node):
    """Declare shared parameters and return calibration only when explicitly enabled."""
    defaults = {'enabled': False, **{field.name: field.default for field in fields(TurnCalibration)}}
    values = {key: node.declare_parameter(f'turn_calibration.{key}', value).value
              for key, value in defaults.items()}
    if not values.pop('enabled'):
        return None
    return TurnCalibration(**values)


@dataclass(frozen=True)
class ForwardCalibration:
    """Measured straight motor pair and corresponding forward speed in meters/second."""

    left_input: float = 0.18
    right_input: float = 0.15
    speed_mps: float = 0.203

    def __post_init__(self):
        """Require finite forward-only inputs within the driver's range and positive speed."""
        values = (self.left_input, self.right_input, self.speed_mps)
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError('Forward motor inputs and measured speed must be positive and finite.')
        if max(self.left_input, self.right_input) > 0.5:
            raise ValueError('Forward inputs must not exceed the driver limit of 0.5.')


def read_forward_calibration(node):
    """Read the same fixed straight-motion calibration for navigation and mapping."""
    defaults = asdict(ForwardCalibration())
    values = {key: node.declare_parameter(f'forward_calibration.{key}', value).value
              for key, value in defaults.items()}
    return ForwardCalibration(**values)
