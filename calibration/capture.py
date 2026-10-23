"""Manually save raw frames from a local camera, video, or GStreamer pipeline."""

import argparse
from pathlib import Path
import time

import cv2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--device", type=int, help="OpenCV camera index")
    source.add_argument("--video", help="Recorded raw camera video path")
    source.add_argument("--pipeline", help="GStreamer pipeline ending in appsink; needs OpenCV GStreamer support")
    parser.add_argument("--output", type=Path, required=True, help="New image directory")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.pipeline:
        cap = cv2.VideoCapture(args.pipeline, cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(args.device if args.device is not None else args.video)
    count = 0
    try:
        if not cap.isOpened():
            raise RuntimeError("Cannot open source. Check device/backend; ROS cameras can use capture_ros.py.")
        print("SPACE: save unmodified frame; Q/ESC: quit. Requires a desktop display.")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imshow("Raw calibration capture", frame)
            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q")):
                break
            if key == 32:
                path = args.output / f"frame_{time.time_ns()}.png"
                if not cv2.imwrite(str(path), frame):
                    raise OSError(f"Cannot save {path}")
                count += 1
                print(f"Saved {count}: {path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
