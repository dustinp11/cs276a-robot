"""Bounded tag reacquisition using calibrated turns and stopped observations."""

import math
from hw3.coordinates import wrap_angle
from hw4.checkpoint import checkpoint_pose


class TagSearch:
    """Turn up to one revolution; predict heading separately from measured feedback."""

    def __init__(self, now, expected, turn, policy):
        """Keep the current world-meter position and use the calibrated CCW rate."""
        # preserve pose while bounding recovery motion
        self.expected, self.policy = expected, policy
        self.omega = turn.rate(1.0)
        self.rotated, self.pose = 0.0, None
        self.started, self.deadline = now, now + policy['search_timeout_s']
        self.last_tick, self.phase_start, self.phase = now, now, 'turning'
        self.command = 0.0, self.omega

    def pause(self, now):
        """Stop for fresh post-settle observations after seeing a candidate or finishing a sweep."""
        self.phase, self.command = 'observing', (0.0, 0.0)
        self.after = now + self.policy['settle_s']
        self.observe_until = self.after + self.policy['search_observation_wait_s']

    def resume(self, now):
        """Continue the remaining sweep without resetting the total search budget."""
        self.pose = None
        if self.rotated >= 2 * math.pi or now >= self.deadline:
            self.phase, self.command = 'done', (0.0, 0.0)
        else:
            self.last_tick, self.phase_start, self.phase = now, now, 'turning'
            self.command = 0.0, self.omega
        return self.command

    def tick(self, now, samples):
        """Return a motor command; expose a stopped pose or an exhausted search via phase."""
        # integrate only commanded rotation intervals
        if self.phase == 'turning':
            angle = self.omega * (now - self.last_tick)
            x, y, yaw = self.expected
            self.expected = x, y, wrap_angle(yaw + angle)
            self.rotated += abs(angle)
        self.last_tick = now
        if now >= self.deadline:
            self.phase, self.command = 'done', (0.0, 0.0)
            return self.command

        # stop when a fresh fused observation becomes available
        if self.phase == 'turning':
            visible = any(self.phase_start < pose.stamp <= now
                          and now - pose.stamp <= self.policy['pose_timeout_s'] for pose in samples)
            if visible or self.rotated >= 2 * math.pi:
                self.pause(now)
        elif self.phase == 'observing' and now >= self.after:
            self.pose = checkpoint_pose(samples, self.after, now, self.policy)
            if self.pose is not None:
                self.phase = 'found'
            elif now >= self.observe_until:
                return self.resume(now)
        return self.command
