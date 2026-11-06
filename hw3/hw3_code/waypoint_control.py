"""ROS-independent turn/drive/turn controller adapted from hw_2_solution.

Only pose feedback and coordinate goals are used. Localization, tag identities,
TF, publishers, and motor interfaces belong outside this controller.
"""

from dataclasses import dataclass
import math

import numpy as np


def wrap_angle(angle):
    """Normalize an angle in radians to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class ControlSettings:
    """Controller settings in meters, seconds, and radians; defaults suit slow square traversal."""

    dt: float = 0.05
    kp: float = 0.8
    ki: float = 0.01
    kd: float = 0.005
    max_speed: float = 0.567333
    rotation_speed: float = 0.8
    position_tolerance: float = 0.10
    yaw_tolerance: float = 0.1
    heading_tolerance: float = 0.08
    realign_threshold: float = 0.2
    max_pose_age: float = 0.5
    turn_gain: float = 1.5

    def __post_init__(self):
        """Reject nonphysical settings before starting a mission."""
        for name, value in vars(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if self.heading_tolerance >= self.realign_threshold:
            raise ValueError('heading_tolerance must be below realign_threshold')


class WaypointController:
    """Follow [x, y, yaw] goals using a pose supplied by an external estimator.

    Adapted stages and gains come from hw_2_solution/hw2_solution.py. Unlike that
    implementation, this class uses one pose for both arrival and steering, stops
    on missing/stale feedback, and skips turning toward a goal already reached.
    """

    def __init__(self, waypoints, settings=None):
        """Copy finite coordinate goals; no map or landmark identities are needed."""
        self.settings = settings or ControlSettings()
        self.waypoints = np.asarray(waypoints, dtype=float).copy()
        if self.waypoints.ndim != 2 or self.waypoints.shape[1] != 3:
            raise ValueError('waypoints must have shape (N, 3) in meters/radians')
        if not np.isfinite(self.waypoints).all():
            raise ValueError('waypoints must be finite')
        self.index = 0
        self.stage = 'rotate_to_goal'
        self._reset_pid()

    @property
    def done(self):
        """Return whether all coordinate goals have been completed."""
        return self.index >= len(self.waypoints)

    def _reset_pid(self):
        """Clear distance-controller memory between goals or feedback interruptions."""
        self.integral = 0.0
        self.previous_error = 0.0

    def step(self, pose, pose_age=0.0):
        """Return one (linear m/s, angular rad/s) command at the configured timestep.

        Args:
            pose: Estimated [x, y, yaw] in the goals' frame, or None if unavailable.
            pose_age: Seconds since the pose timestamp, measured by the caller.

        Returns:
            tuple: (forward velocity, yaw rate); zero when done or feedback is invalid.
        """
        if self.done or not self._valid_feedback(pose, pose_age):
            self._reset_pid()
            return 0.0, 0.0
        pose = np.asarray(pose, dtype=float)
        goal = self.waypoints[self.index]
        distance = float(np.linalg.norm(goal[:2] - pose[:2]))
        if distance < self.settings.position_tolerance:
            self.stage = 'rotate_to_orient'
        return self._stage_command(pose, goal, distance)

    def _valid_feedback(self, pose, age):
        """Check feedback dimensions, values, and freshness before advancing state."""
        if pose is None or not math.isfinite(age) or not 0 <= age <= self.settings.max_pose_age:
            return False
        values = np.asarray(pose, dtype=float)
        return values.shape == (3,) and bool(np.isfinite(values).all())

    def _stage_command(self, pose, goal, distance):
        """Select the command for the current turn/drive/turn stage."""
        if self.stage == 'rotate_to_orient':
            return self._finish_heading(wrap_angle(goal[2] - pose[2]))
        bearing = math.atan2(goal[1] - pose[1], goal[0] - pose[0])
        error = wrap_angle(bearing - pose[2])
        if self.stage == 'rotate_to_goal':
            return self._face_goal(error)
        if abs(error) > self.settings.realign_threshold:
            self.stage = 'rotate_to_goal'
            return 0.0, 0.0
        return self._forward_speed(distance), 0.0

    def _face_goal(self, error):
        """Slow the turn near alignment, then enter straight driving."""
        if abs(error) < self.settings.heading_tolerance:
            self.stage = 'drive'
            return 0.0, 0.0
        return 0.0, self._turn_speed(error)

    def _finish_heading(self, error):
        """Reach the requested final heading, then advance to the next goal."""
        if abs(error) >= self.settings.yaw_tolerance:
            return 0.0, self._turn_speed(error)
        self.index += 1
        self.stage = 'rotate_to_goal'
        self._reset_pid()
        return 0.0, 0.0

    def _forward_speed(self, distance):
        """Use the reference distance PID, including its unscaled error-difference term."""
        settings = self.settings
        self.integral += settings.ki * distance * settings.dt
        derivative = settings.kd * (distance - self.previous_error)
        self.previous_error = distance
        speed = settings.kp * distance + self.integral + derivative
        return min(settings.max_speed, max(0.0, speed))

    def _turn_speed(self, error):
        """Bound proportional turning to reduce overshoot near the desired heading."""
        limit = self.settings.rotation_speed
        return max(-limit, min(limit, self.settings.turn_gain * error))


def square_waypoints(side_ft=7.0, laps=1):
    """Return counterclockwise square vertices in the local start frame, in m/rad."""
    if not math.isfinite(side_ft) or side_ft <= 0 or type(laps) is not int or laps < 1:
        raise ValueError('Square side must be positive and laps must be a positive integer')
    side = side_ft * 0.3048
    vertices = [[side, 0., math.pi/2.], [side, side, math.pi],
                [0., side, -math.pi/2.], [0., 0., 0.]]
    return vertices * laps


class FixedSquareController:
    """Run a calibrated timed square, without pose, landmarks, or motion-history inputs."""

    def __init__(self, side_ft=7.0, laps=1, settings=None, turn_time_s=0.48, settle_time_s=1.0):
        """Time straight legs from calibrated speed and left turns from measured turn time."""
        square_waypoints(side_ft, laps)
        if not all(math.isfinite(value) and value > 0 for value in (turn_time_s, settle_time_s)):
            raise ValueError('Turn and settle durations must be positive')
        self.settings = settings or ControlSettings()
        self.drive_time_s = side_ft*0.3048/self.settings.max_speed
        self.turn_time_s, self.settle_time_s, self.laps = turn_time_s, settle_time_s, laps
        self.index, self.stage, self.elapsed = 0, 'drive', 0.
        self.last_time, self.running = None, False

    @property
    def done(self):
        """Return whether four timed legs and left turns per lap have completed."""
        return self.index >= 4*self.laps

    def step(self, now, enabled=True):
        """Return a timed command using monotonic seconds; freeze the clock when paused."""
        if not math.isfinite(now) or (self.last_time is not None and now < self.last_time):
            raise ValueError('Controller clock must be finite and monotonic')
        if self.running and self.last_time is not None:
            self.elapsed += now-self.last_time
        self.last_time, self.running = now, enabled
        if self.done or not enabled:
            return 0., 0.
        if self.elapsed >= self._duration()-1e-9:
            self._advance()
        return self._command()

    def _duration(self):
        """Return the active stage duration, including explicit stops around each turn."""
        if self.stage == 'drive':
            return self.drive_time_s
        if self.stage == 'turn_left':
            return self.turn_time_s
        return self.settle_time_s

    def _advance(self):
        """Advance one stage only, so delayed callbacks cannot skip an entire physical turn."""
        transitions = {'drive': 'stop_before_turn', 'stop_before_turn': 'turn_left',
                       'turn_left': 'stop_after_turn', 'stop_after_turn': 'drive'}
        if self.stage == 'stop_after_turn':
            self.index += 1
        self.stage, self.elapsed = transitions[self.stage], 0.

    def _command(self):
        """Issue forward, opposing-wheel turn, or zero commands for the selected stage."""
        if self.done:
            return 0., 0.
        if self.stage == 'drive':
            return self.settings.max_speed, 0.
        if self.stage == 'turn_left':
            return 0., self.settings.rotation_speed
        return 0., 0.
