"""Combine independent detector packets into one observation per camera image."""

from dataclasses import dataclass, field


@dataclass
class Capture:
    """Packets from one camera capture with a bounded detector synchronization wait."""

    stamp_ns: int
    frame: str
    received: float
    messages: dict = field(default_factory=dict)


class CaptureBuffer:
    """Prevent two family packets from counting as two independent camera frames."""

    def __init__(self):
        """Initialize a small queue and the last consumed capture timestamp."""
        self.pending, self.last_stamp = {}, 0

    def add(self, stamp_ns, frame, family, message, now):
        """Retain one packet per family and ignore consumed or duplicate images."""
        # combine only matching camera capture identities
        if stamp_ns <= self.last_stamp:
            return
        capture = self.pending.setdefault(stamp_ns, Capture(stamp_ns, frame, now))
        if frame == capture.frame:
            capture.messages.setdefault(family, message)
        while len(self.pending) > 5:
            self.finish(min(self.pending))

    def ready(self, now, families, wait_s=0.15):
        """Wait briefly for all families, allowing a healthy detector to stand alone."""
        if not self.pending:
            return None
        capture = self.pending[min(self.pending)]
        complete = set(families).issubset(capture.messages)
        return capture if complete or now - capture.received >= wait_s else None

    def finish(self, stamp_ns):
        """Consume a capture once and discard older queued packets."""
        self.last_stamp = max(self.last_stamp, stamp_ns)
        self.pending = {stamp: capture for stamp, capture in self.pending.items() if stamp > self.last_stamp}
