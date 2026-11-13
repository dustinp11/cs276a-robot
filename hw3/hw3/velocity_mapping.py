"""Convert planar commands into HW3 motor inputs and nominal applied velocities."""

from dataclasses import dataclass
import math

from .turn_calibration import TurnCalibration


@dataclass(frozen=True)
class MotorCommand:
    """Left/right motor inputs and model velocities (m/s, rad/s), not odometry."""

    left: float
    right: float
    linear_mps: float
    angular_radps: float


class VelocityMapper:
    """Keep straight travel separate from calibrated fixed-rate pivoting."""

    def __init__(self, calibration: TurnCalibration):
        """Store validated turn calibration and reuse HW2's linear calibration."""
        self.calibration = calibration
        self.linear_deadzone = 0.08
        self.linear_slope = 1.772915625

    def map_command(self, linear_mps: float, angular_radps: float) -> MotorCommand:
        """Map a straight or pivot request; nonfinite requests produce a stop.

        Args:
            linear_mps: Forward speed in m/s; negative requests reverse motion.
            angular_radps: Signed yaw request in rad/s, used only when linear is zero.

        Returns:
            Motor inputs in [-0.5, 0.5] and the corresponding calibrated model
            velocity. A straight command suppresses angular steering.
        """
        if not all(math.isfinite(v) for v in (linear_mps, angular_radps)):
            return MotorCommand(0.0, 0.0, 0.0, 0.0)
        if linear_mps:
            motor, speed = self._straight_input(linear_mps)
            return MotorCommand(motor, motor, speed, 0.0)
        left, right = self.calibration.motors(angular_radps)
        return MotorCommand(left, right, 0.0, self.calibration.rate(angular_radps))

    def _straight_input(self, speed: float) -> tuple[float, float]:
        """Clip motor input and return its achievable nominal linear speed in m/s."""
        magnitude = min(0.5, self.linear_deadzone + abs(speed) / self.linear_slope)
        achieved = (magnitude - self.linear_deadzone) * self.linear_slope
        return math.copysign(magnitude, speed), math.copysign(achieved, speed)
