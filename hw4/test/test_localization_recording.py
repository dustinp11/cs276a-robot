"""Check manual-rotation artifacts with logged poses, without camera or ROS."""

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent / 'hw3')]
from hw4.localization_log import LocalizationLog
from hw4.localization_report import write_report


class RecordingChecks(unittest.TestCase):
    """Exercise per-tag logging and report generation using actual conflicting estimates."""

    def test_report_preserves_each_tag_error_instead_of_the_fused_error(self):
        """Reproduce the run's 2.5 cm versus 8.4 cm individual position errors."""
        # preserve independent estimates from the supplied run
        candidates = {
            '36h11:1': {'pose_xy_yaw': [0.771975, 1.528359, 1.12814], 'camera_range_m': 1.8368},
            '36h11:3': {'pose_xy_yaw': [0.814076, 1.484941, 1.15465], 'camera_range_m': 1.4461},
            '41h12:1': {'rejected': 'implausible base height or tilt'}}
        paths = {'environment.json': ROOT / 'config/environment.json',
                 'navigation.yaml': ROOT / 'config/navigation.yaml'}
        with tempfile.TemporaryDirectory() as directory:
            log = LocalizationLog(directory, paths, (0.75, 1.54))
            log.observations.writerow([10, 1000, True, 'mapped landmark estimate',
                                       0.7930255, 1.50665, 1.1414, (), '{}', json.dumps(candidates)])
            log.close()
            summary = write_report(directory)
            self.assertEqual(log.capture_count, 1)
            self.assertAlmostEqual(summary['36h11:1']['rmse_cm'], 2.49, places=2)
            self.assertAlmostEqual(summary['36h11:3']['rmse_cm'], 8.45, places=2)
            self.assertNotIn('41h12:1', summary)

            # keep rejected candidates and valid svg output
            with (Path(directory) / 'per_tag.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[-1]['x_m'], '')
            self.assertIn('height', rows[-1]['reason'])
            ET.parse(Path(directory) / 'tag_positions.svg')

    def test_empty_recording_does_not_claim_tag_accuracy(self):
        """Produce an empty summary when no camera observations arrived."""
        paths = {'environment.json': ROOT / 'config/environment.json'}
        with tempfile.TemporaryDirectory() as directory:
            log = LocalizationLog(directory, paths, (0.75, 1.54))
            log.close()
            self.assertEqual(write_report(directory), {})
            with (Path(directory) / 'tag_summary.csv').open() as stream:
                self.assertEqual(list(csv.DictReader(stream)), [])
