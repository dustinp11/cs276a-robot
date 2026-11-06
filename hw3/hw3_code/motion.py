"""Timestamped applied-command history for delayed vision and current-pose prediction."""

import math

import numpy as np


class MotionHistory:
    """Integrate reported applied commands; these are not wheel encoder measurements."""

    def __init__(self, distance_noise, turn_noise, idle_distance, idle_turn, timeout=0.15):
        """Store variance rates for distance/turn increments and motor command expiry."""
        self.samples = []
        self.distance_noise, self.turn_noise = distance_noise, turn_noise
        self.idle_distance, self.idle_turn = idle_distance, idle_turn
        self.timeout = timeout

    def add(self, stamp, linear, angular):
        """Accept a finite chronological applied velocity, in seconds, m/s, and rad/s."""
        if not all(math.isfinite(value) for value in (stamp, linear, angular)):
            return False
        if self.samples and stamp <= self.samples[-1][0]:
            return False
        self.samples.append((stamp, linear, angular))
        return True

    def propagate(self, estimator, start, end):
        """Predict across command changes and expiry, using short midpoint increments."""
        if end < start:
            raise ValueError('Cannot predict backwards')
        samples = self.samples
        boundaries = sorted({start, end} | {s[0] for s in samples if start < s[0] < end}
                            | {s[0]+self.timeout for s in samples if start < s[0]+self.timeout < end})
        index = 0
        for left, right in zip(boundaries, boundaries[1:]):
            while index+1 < len(samples) and samples[index+1][0] <= left + 1e-9:
                index += 1
            valid = samples and samples[index][0] <= left + 1e-9 and left < samples[index][0]+self.timeout-1e-9
            linear, angular = samples[index][1:] if valid else (0., 0.)
            self._segment(estimator, linear, angular, right-left)

    def _segment(self, estimator, linear, angular, duration):
        """Bound model discretization error and map variance rates into each increment."""
        count = max(1, math.ceil(duration / 0.05))
        dt = duration / count
        distance, turn = linear*dt, angular*dt
        variances = [self.distance_noise**2*abs(distance) + self.idle_distance**2*dt,
                     self.turn_noise**2*abs(turn) + self.idle_turn**2*dt]
        for _ in range(count):
            estimator.predict(distance, turn, np.diag(variances))

    def prune(self, before):
        """Retain the last command at/before estimator time plus all newer commands."""
        while len(self.samples) > 1 and self.samples[1][0] <= before:
            self.samples.pop(0)
