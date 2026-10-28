"""Save periodic raw/distorted images from a ROS 2 camera topic."""

import argparse
from pathlib import Path
import time

import cv2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="/camera/image_raw/compressed")
    parser.add_argument("--raw", action="store_true", help="Subscribe to sensor_msgs/Image instead of CompressedImage")
    parser.add_argument("--output", type=Path, help="New output directory (defaults under calibration/images)")
    parser.add_argument("--interval", type=float, default=5.0, help="Seconds between saved frames")
    parser.add_argument("--count", type=int, default=25)
    args = parser.parse_args()
    if not 0 < args.interval < float("inf") or args.count < 1:
        parser.error("interval must be finite and positive; count must be positive")
    # Keep ROS imports optional for the offline PC workflow.
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage, Image
    from cv_bridge import CvBridge

    output = args.output or (
        Path(__file__).resolve().parent / "images" /
        f"capture_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    output.mkdir(parents=True, exist_ok=False)
    rclpy.init(args=[])
    node = Node("calibration_image_collector")
    bridge = CvBridge()
    count = 0
    last_saved = float("-inf")

    def receive(message):
        nonlocal count, last_saved
        if count >= args.count or time.monotonic() - last_saved < args.interval:
            return
        frame = (bridge.imgmsg_to_cv2(message, desired_encoding="bgr8") if args.raw
                 else bridge.compressed_imgmsg_to_cv2(message, desired_encoding="bgr8"))
        path = output / f"frame_{time.time_ns()}.png"
        if not cv2.imwrite(str(path), frame):
            raise OSError(f"Cannot save {path}")
        count += 1
        last_saved = time.monotonic()
        node.get_logger().info(f"Saved {count}/{args.count}: {path}")

    subscription = node.create_subscription(
        Image if args.raw else CompressedImage, args.topic, receive, qos_profile_sensor_data
    )
    node.get_logger().info(
        f"Saving {args.count} raw/distorted images, at least {args.interval:g}s apart, to {output}. "
        "Move the calibration board through varied poses."
    )
    try:
        while rclpy.ok() and count < args.count:
            rclpy.spin_once(node, timeout_sec=1.0)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
