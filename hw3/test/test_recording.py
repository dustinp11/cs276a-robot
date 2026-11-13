"""Unit tests for Data Recording, Replaying, and Map Exporting (Step 4.1 & 4.4)."""

from pathlib import Path
import tempfile
import unittest
import numpy as np

from hw3.coordinates import make_transform
from hw3.data_recorder import (
    DataRecorder,
    DataReplayer,
    export_landmarks_csv,
    export_trajectory_csv,
)


class TestDataRecording(unittest.TestCase):
    def test_record_and_replay(self):
        """Test recording observations to JSON and replaying through VO and EKF."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            record_file = Path(tmp_dir) / "test_records.json"
            recorder = DataRecorder(record_file)

            # Record two frames of tag observations moving forward
            det1 = [{
                'id': 0,
                'family': '36h11',
                't_cam_tag': make_transform(np.eye(3), [0.0, 0.0, 2.0]),
                'margin': 50.0,
                'hamming': 0,
            }]
            recorder.record_frame(stamp_s=1.0, detections=det1, cmd_v=0.2, cmd_omega=0.0)

            det2 = [{
                'id': 0,
                'family': '36h11',
                't_cam_tag': make_transform(np.eye(3), [0.0, 0.0, 1.8]),
                'margin': 50.0,
                'hamming': 0,
            }]
            recorder.record_frame(stamp_s=1.1, detections=det2, cmd_v=0.2, cmd_omega=0.0)

            recorder.save()
            self.assertTrue(record_file.exists())

            # Now replay the data
            replayer = DataReplayer(record_file)
            results = replayer.run(use_vo=True)

            traj = results['trajectory']
            lms = results['landmarks']
            self.assertEqual(len(traj), 2)
            self.assertIn(0, lms)
            self.assertAlmostEqual(lms[0]['x'], 2.0, places=2)

            # Test CSV export
            traj_csv = Path(tmp_dir) / "traj.csv"
            lms_csv = Path(tmp_dir) / "lms.csv"
            export_trajectory_csv(traj, traj_csv)
            export_landmarks_csv(lms, lms_csv)

            self.assertTrue(traj_csv.exists())
            self.assertTrue(lms_csv.exists())
            self.assertGreater(traj_csv.stat().st_size, 0)
            self.assertGreater(lms_csv.stat().st_size, 0)


if __name__ == '__main__':
    unittest.main()
