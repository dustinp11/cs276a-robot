"""Hardware-free geometry, detection, export, and failure-path checks."""

import argparse
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from calibrate import board_points, fit_camera, run


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.matrix = np.array([[820., 0., 640.], [0., 800., 360.], [0., 0., 1.]])
        self.objects = board_points(9, 6, 0.025)
        rng = np.random.default_rng(276)
        self.poses = [(rng.uniform(-0.5, 0.5, 3),
                       np.array([rng.uniform(-0.17, 0.02), rng.uniform(-0.12, 0.01),
                                 rng.uniform(0.55, 0.85)])) for _ in range(18)]

    def test_recover_known_distorted_camera(self):
        distortion = np.array([-0.20, 0.06, 0.001, -0.002, -0.01])
        observations = [cv2.projectPoints(self.objects, r, t, self.matrix, distortion)[0]
                        for r, t in self.poses]
        rms, matrix, recovered, _, _, errors = fit_camera(
            [self.objects.copy() for _ in observations], observations, (1280, 720))
        self.assertLess(rms, 0.001)
        np.testing.assert_allclose(matrix, self.matrix, atol=0.03)
        np.testing.assert_allclose(recovered.ravel(), distortion, atol=0.003)
        self.assertLess(max(errors), 0.001)

    def render_dataset(self, directory):
        # One square border around the 9 x 6 inner corners.
        tile = 80
        board = np.full((7 * tile, 10 * tile), 255, np.uint8)
        for row in range(7):
            for col in range(10):
                if (row + col) % 2 == 0:
                    board[row*tile:(row+1)*tile, col*tile:(col+1)*tile] = 0
        outside = np.array([[-1, -1, 0], [9, -1, 0], [9, 6, 0], [-1, 6, 0]],
                           np.float32) * 0.025
        source = np.array([[0, 0], [800, 0], [800, 560], [0, 560]], np.float32)
        for index, (rotation, translation) in enumerate(self.poses):
            target = cv2.projectPoints(outside, rotation, translation, self.matrix,
                                       np.zeros(5))[0].reshape(4, 2)
            transform = cv2.getPerspectiveTransform(source, target)
            frame = cv2.warpPerspective(board, transform, (1280, 720), borderValue=180)
            self.assertTrue(cv2.imwrite(str(directory / f"{index:02d}.png"), frame))

    def test_image_pipeline_and_yaml_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"
            images.mkdir()
            self.render_dataset(images)
            cv2.imwrite(str(images / "blank.png"), np.zeros((720, 1280), np.uint8))
            args = argparse.Namespace(images=images, output=root / "result", columns=9,
                                      rows=6, square_size_m=0.025, min_views=10)
            run(args)
            import json
            report = json.loads((args.output / "calibration.json").read_text())
            self.assertEqual(len(report["views"]), 18)
            self.assertEqual(len(report["rejected"]), 1)
            self.assertLess(report["rms_reprojection_px"], 0.5)
            np.testing.assert_allclose(np.array(report["camera_matrix"])[[0, 1], [0, 1]],
                                       self.matrix[[0, 1], [0, 1]], rtol=0.02)
            storage = cv2.FileStorage(str(args.output / "camera_parameter.yaml"), cv2.FILE_STORAGE_READ)
            try:
                self.assertEqual(storage.getNode("image_width").real(), 1280)
                self.assertEqual(storage.getNode("distortion_model").string(), "plumb_bob")
                np.testing.assert_allclose(storage.getNode("camera_matrix").mat(), report["camera_matrix"])
                self.assertEqual(storage.getNode("distortion_coefficients").mat().shape, (1, 5))
                projection = storage.getNode("projection_matrix").mat()
                np.testing.assert_allclose(projection[:, :3], report["undistorted_camera_matrix"])
            finally:
                storage.release()
            self.assertTrue((args.output / "preview_raw_and_undistorted.png").is_file())
            with self.assertRaises(FileExistsError):
                run(args)

    def test_mixed_sizes_and_insufficient_views(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"
            images.mkdir()
            cv2.imwrite(str(images / "a.png"), np.zeros((720, 1280), np.uint8))
            args = argparse.Namespace(images=images, output=root / "few", columns=9,
                                      rows=6, square_size_m=0.025, min_views=10)
            with self.assertRaisesRegex(ValueError, "Only 0 usable views"):
                run(args)
            cv2.imwrite(str(images / "b.png"), np.zeros((480, 640), np.uint8))
            args.output = root / "mixed"
            with self.assertRaisesRegex(ValueError, "Mixed image resolutions"):
                run(args)


if __name__ == "__main__":
    unittest.main()
