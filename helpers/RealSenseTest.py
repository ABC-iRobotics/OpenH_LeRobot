#!/usr/bin/env python3
import os
import csv
import time
import argparse
from pathlib import Path

import numpy as np
import cv2

try:
    import pyrealsense2 as rs
except ImportError as e:
    raise SystemExit(
        "ERROR: pyrealsense2 not found. Install it (often via Intel RealSense packages) "
        "or pip if available for your platform."
    ) from e


def main():
    parser = argparse.ArgumentParser(description="Record RealSense color frames at ~30 FPS + CSV (unix ns).")
    parser.add_argument("--out", type=str, default="realsense_recording", help="Output folder")
    parser.add_argument("--width", type=int, default=640, help="Color width")
    parser.add_argument("--height", type=int, default=480, help="Color height")
    parser.add_argument("--fps", type=int, default=60, help="Target FPS")
    parser.add_argument("--format", type=str, default="jpg", choices=["jpg", "png"], help="Image format")
    parser.add_argument("--jpeg_quality", type=int, default=90, help="JPG quality (1-100)")
    parser.add_argument("--max_frames", type=int, default=0,
                        help="Stop after N frames (0 = run until 'q' is pressed)")
    parser.add_argument("--show", action="store_true", help="Show live preview window")
    args = parser.parse_args()

    out_dir = Path(args.out)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "log.csv"

    # RealSense pipeline setup
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)

    # Start streaming
    profile = pipeline.start(config)

    # Try to stabilize exposure/auto settings for a short moment
    time.sleep(0.2)

    # Timing control (best-effort)
    period_s = 1.0 / float(args.fps)
    next_t = time.perf_counter()

    frame_idx = 0
    print(f"[INFO] Saving frames to: {img_dir}")
    print(f"[INFO] Writing CSV log to: {csv_path}")
    print("[INFO] Press 'q' in the preview window to stop (or Ctrl+C).")

    # Open CSV once and flush as we go
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "unix_timestamp_ns"])

        try:
            while True:
                # Throttle to ~target FPS
                now = time.perf_counter()
                if now < next_t:
                    time.sleep(max(0.0, next_t - now))
                next_t += period_s

                # Wait for a new frameset (timeout in ms)
                frames = pipeline.wait_for_frames(timeout_ms=1000)
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                # Convert to numpy
                color_image = np.asanyarray(color_frame.get_data())

                # Timestamp: UNIX epoch in nanoseconds (PC clock)
                ts_ns = time.time_ns()

                # Filename
                image_name = f"frame_{frame_idx:06d}.{args.format}"
                image_path = str(img_dir / image_name)

                # Write image
                if args.format == "jpg":
                    ok = cv2.imwrite(image_path, color_image,
                                     [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)])
                else:
                    ok = cv2.imwrite(image_path, color_image)

                if not ok:
                    print(f"[WARN] Failed to write {image_path}")
                    continue

                # CSV log line
                writer.writerow([image_name, ts_ns])
                f.flush()

                frame_idx += 1

                # Optional preview
                if args.show:
                    cv2.imshow("RealSense Color (press q to stop)", color_image)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break

                # Optional stop after max_frames
                if args.max_frames > 0 and frame_idx >= args.max_frames:
                    break

        except KeyboardInterrupt:
            pass
        finally:
            pipeline.stop()
            if args.show:
                cv2.destroyAllWindows()

    print(f"[DONE] Saved {frame_idx} frames.")
    print(f"[DONE] CSV: {csv_path}")


if __name__ == "__main__":
    main()
