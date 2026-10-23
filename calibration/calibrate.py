"""Offline monocular chessboard calibration; no ROS or robot-control dependencies."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import numpy as np


def board_points(columns, rows, square_size_m):
    points = np.zeros((columns * rows, 3), np.float32)
    points[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    return points * square_size_m


def detect_corners(image, pattern):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCornersSB(
        gray, pattern, cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    return corners if found else None


def fit_camera(objects, observations, size):
    rms, matrix, distortion, rotations, translations = cv2.calibrateCamera(
        objects, observations, size, None, None
    )
    if (not np.isfinite(rms) or not np.isfinite(matrix).all()
            or not np.isfinite(distortion).all()
            or matrix[0, 0] <= 0 or matrix[1, 1] <= 0):
        raise ValueError("Calibration returned invalid parameters; check board geometry and views.")
    errors = []
    for obj, obs, rotation, translation in zip(
        objects, observations, rotations, translations
    ):
        projected, _ = cv2.projectPoints(obj, rotation, translation, matrix, distortion)
        errors.append(float(np.sqrt(np.mean(np.sum(
            (obs.reshape(-1, 2) - projected.reshape(-1, 2)) ** 2, axis=1
        )))))
    return rms, matrix, distortion, rotations, translations, errors


def write_camera_yaml(path, size, matrix, distortion, new_matrix):
    """OpenCV FileStorage format consumed by the repository's camera driver."""
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    if not storage.isOpened():
        raise OSError(f"Cannot write {path}")
    try:
        storage.write("image_width", size[0])
        storage.write("image_height", size[1])
        storage.write("camera_name", "camera")
        storage.write("distortion_model", "plumb_bob")
        storage.write("camera_matrix", matrix)
        storage.write("distortion_coefficients", distortion.reshape(1, -1))
        storage.write("rectification_matrix", np.eye(3))
        storage.write("projection_matrix", np.column_stack((new_matrix, np.zeros(3))))
    finally:
        storage.release()


def run(args):
    if args.columns < 3 or args.rows < 3:
        raise ValueError("Use at least three inner corners along each board dimension.")
    if not np.isfinite(args.square_size_m) or args.square_size_m <= 0:
        raise ValueError("--square-size-m must be a positive measured length in meters.")
    if args.min_views < 3:
        raise ValueError("--min-views must be at least 3; 15–25 varied views are recommended.")
    paths = sorted(p for p in args.images.iterdir()
                   if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
    if not paths:
        raise ValueError(f"No images in {args.images}")
    # A fresh directory avoids mixing diagnostics from different runs.
    args.output.mkdir(parents=True, exist_ok=False)
    diagnostics = args.output / "corners"
    diagnostics.mkdir()
    pattern = (args.columns, args.rows)
    objects, observations, accepted, rejected = [], [], [], []
    obj = board_points(*pattern, args.square_size_m)
    size = None
    first_image = None
    for index, path in enumerate(paths):
        image = cv2.imread(str(path))
        if image is None:
            rejected.append({"image": str(path), "reason": "unreadable"})
            continue
        current_size = (image.shape[1], image.shape[0])
        if size is None:
            size = current_size
        if current_size != size:
            raise ValueError(f"Mixed image resolutions: {path} is {current_size}, expected {size}.")
        corners = detect_corners(image, pattern)
        if corners is None:
            rejected.append({"image": str(path), "reason": "full chessboard not detected"})
            continue
        objects.append(obj.copy())
        observations.append(corners)
        accepted.append(str(path))
        if first_image is None:
            first_image = image.copy()
        cv2.drawChessboardCorners(image, pattern, corners, True)
        if not cv2.imwrite(str(diagnostics / f"{index:04d}_{path.stem}.png"), image):
            raise OSError("Failed to save corner diagnostic image")
    (args.output / "detections.json").write_text(json.dumps(
        {"accepted": accepted, "rejected": rejected}, indent=2) + "\n")
    if len(accepted) < args.min_views:
        raise ValueError(f"Only {len(accepted)} usable views; need {args.min_views}. See detections.json.")
    rms, matrix, distortion, rotations, translations, errors = fit_camera(
        objects, observations, size
    )
    # Alpha=0 and full size mirror the existing vision driver's remap settings.
    new_matrix, roi = cv2.getOptimalNewCameraMatrix(matrix, distortion, size, 0.0, size)
    write_camera_yaml(args.output / "camera_parameter.yaml", size, matrix, distortion, new_matrix)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "opencv_version": cv2.__version__, "model": "plumb_bob",
        "distortion_order": ["k1", "k2", "p1", "p2", "k3"],
        "image_size_wh": list(size), "board_inner_corners_cr": list(pattern),
        "square_size_m": args.square_size_m, "rms_reprojection_px": float(rms),
        "camera_matrix": matrix.tolist(), "distortion": distortion.ravel().tolist(),
        "undistorted_camera_matrix": new_matrix.tolist(), "valid_roi_xywh": list(map(int, roi)),
        "rejected": rejected,
        "views": [{"image": path, "rms_reprojection_px": error,
                   "rvec_board_to_camera": rotation.ravel().tolist(),
                   "tvec_board_to_camera_m": translation.ravel().tolist()}
                  for path, error, rotation, translation in zip(
                      accepted, errors, rotations, translations)],
    }
    (args.output / "calibration.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    corrected = cv2.undistort(first_image, matrix, distortion, None, new_matrix)
    if not cv2.imwrite(str(args.output / "preview_raw_and_undistorted.png"),
                       np.hstack((first_image, corrected))):
        raise OSError("Failed to write preview")
    print(f"Calibrated {len(accepted)} views; RMS reprojection error: {rms:.4f} px")
    print(f"Results: {args.output.resolve()}")
    print("Review corner overlays, per-view errors, and straight lines in the preview before use.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="Directory of RAW distorted images")
    parser.add_argument("--columns", type=int, required=True, help="Inner corner columns, not squares")
    parser.add_argument("--rows", type=int, required=True, help="Inner corner rows, not squares")
    parser.add_argument("--square-size-m", type=float, required=True)
    parser.add_argument("--min-views", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True, help="New results directory")
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, OSError, cv2.error) as error:
        parser.exit(1, f"Calibration failed: {error}\n")


if __name__ == "__main__":
    main()
