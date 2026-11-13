"""Convert planar commands into HW3 motor inputs and nominal applied velocities."""

from dataclasses import dataclass
import math

from .turn_calibration import ForwardCalibration, TurnCalibration


@dataclass(frozen=True)
class MotorCommand:
    """Left/right motor inputs and model velocities (m/s, rad/s), not odometry."""

    left: float
    right: float
    linear_mps: float
    angular_radps: float


class VelocityMapper:
    """Keep straight travel separate from calibrated fixed-rate pivoting."""

    def __init__(self, calibration: TurnCalibration, forward_calibration=None):
        """Store independent measured forward and turning calibrations."""
        self.calibration = calibration
        self.forward = forward_calibration or ForwardCalibration()

    def map_command(self, linear_mps: float, angular_radps: float) -> MotorCommand:
        """Map a straight or pivot request; nonfinite requests produce a stop.

        Args:
            linear_mps: Positive requests select the measured fixed forward speed;
                zero allows pivoting and negative requests stop (reverse is uncalibrated).
            angular_radps: Signed yaw request in rad/s, used only when linear is zero.

        Returns:
            Motor inputs in [-0.5, 0.5] and the corresponding calibrated model
            velocity. A straight command suppresses angular steering.
        """
        if linear_mps < 0 or not all(math.isfinite(v) for v in (linear_mps, angular_radps)):
            return MotorCommand(0.0, 0.0, 0.0, 0.0)
        if linear_mps:
            return MotorCommand(self.forward.left_input, self.forward.right_input,
                                self.forward.speed_mps, 0.0)
        left, right = self.calibration.motors(angular_radps)
        return MotorCommand(left, right, 0.0, self.calibration.rate(angular_radps))
