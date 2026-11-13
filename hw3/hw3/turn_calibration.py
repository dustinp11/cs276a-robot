"""Fixed in-place turn calibration shared by mapping and HW3 navigation."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TurnCalibration:
    """Motor magnitude and positive measured yaw rates in rad/s for each direction."""

    ccw_motor_input: float = 0.43
    cw_motor_input: float = 0.40
    ccw_radps: float = 0.0
    cw_radps: float = 0.0

    def __post_init__(self):
        """Reject missing rates or inputs outside the existing driver's limits."""
        values = (self.ccw_motor_input, self.cw_motor_input, self.ccw_radps, self.cw_radps)
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError('Set positive measured turn rates for both CCW and CW.')
        if max(self.ccw_motor_input, self.cw_motor_input) > 0.5:
            raise ValueError('Turn motor input must not exceed the driver limit of 0.5.')

    def rate(self, requested_omega: float) -> float:
        """Return the achievable signed yaw rate; zero requests remain stopped."""
        if not math.isfinite(requested_omega):
            raise ValueError('Requested angular velocity must be finite.')
        if requested_omega == 0:
            return 0.0
        return self.ccw_radps if requested_omega > 0 else -self.cw_radps

    def motors(self, requested_omega: float) -> tuple[float, float]:
        """Return [left, right] fixed inputs; positive yaw means [-input, +input]."""
        rate = self.rate(requested_omega)
        magnitude = self.ccw_motor_input if rate > 0 else self.cw_motor_input
        value = math.copysign(magnitude, rate) if rate else 0.0
        return -value, value


def read_turn_calibration(node):
    """Declare shared parameters and return calibration only when explicitly enabled."""
    defaults = {'enabled': False, 'ccw_motor_input': 0.43, 'cw_motor_input': 0.40,
                'ccw_radps': 0.0, 'cw_radps': 0.0}
    values = {key: node.declare_parameter(f'turn_calibration.{key}', value).value
              for key, value in defaults.items()}
    if not values.pop('enabled'):
        return None
    return TurnCalibration(**values)
