"""Write reproducible timestamped HW3 estimates, observations, and motion inputs."""

from datetime import datetime, timezone
import json
from pathlib import Path


class RunRecorder:
    """Keep each run in its own directory without overwriting previous evidence."""

    def __init__(self, directory, parameters):
        """Create a new run directory and record the configuration used for it."""
        base = Path(directory).expanduser()
        self.directory = base / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory / 'parameters.json').write_text(json.dumps(parameters, indent=2))
        self.output = (self.directory / 'run.jsonl').open('x', buffering=1)

    def write(self, kind, stamp, **values):
        """Append one JSON event; stamp is ROS time in seconds."""
        self.output.write(json.dumps({'kind': kind, 'stamp': stamp, **values}, allow_nan=False) + '\n')

    def snapshot(self, estimator, stamp, **values):
        """Save full state/covariance and ID order for trajectories and landmark ellipses."""
        self.write('estimate', stamp, state=estimator.state.tolist(),
                   covariance=estimator.covariance.tolist(),
                   landmark_ids=[list(key) for key in estimator.landmark_ids], **values)

    def close(self):
        """Close the run log after the final estimate and stop event."""
        self.output.close()
