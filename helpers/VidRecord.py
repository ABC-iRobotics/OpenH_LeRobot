#!/usr/bin/env python3
import os
import signal
import subprocess
import time
from datetime import datetime

import cv2
import numpy as np
import pyrealsense2 as rs


# -------------------------
# GStreamer DeckLink record
# -------------------------
def start_decklink_gst_record(device_number: int, out_path: str, fps="30/1"):
    """Record DeckLink -> H.264 MP4 using gst-launch (no OpenCV GStreamer needed)."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    cmd = [
        "gst-launch-1.0",
        "-e",
        "decklinkvideosrc", "mode=pal", f"device-number={device_number}",
        "!", "videorate",
        "!", f"video/x-raw,framerate={fps}",
        "!", "videoconvert",
        "!", "video/x-raw,format=I420",
        "!", "x264enc", "speed-preset=veryfast", "tune=zerolatency", "bitrate=8000", "key-int-max=30",
        "!", "h264parse",
        "!", "mp4mux",
        "!", "filesink", f"location={out_path}",
    ]

    print(f"▶️ Starting DeckLink{device_number} recorder:\n   " + " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid,  # new process group
    )
    return proc


def stop_process(proc: subprocess.Popen, name: str, timeout_s=5):
    """Stop a process cleanly (SIGINT -> SIGTERM -> SIGKILL)."""
    if proc is None or proc.poll() is not None:
        return

    print(f"⏹️ Stopping {name} ...")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        proc.wait(timeout=timeout_s)
        print(f"✅ {name} stopped (SIGINT).")
        return
    except Exception:
        pass

    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=2)
        print(f"✅ {name} stopped (SIGTERM).")
        return
    except Exception:
        pass

    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        print(f"⚠️ {name} killed (SIGKILL).")
    except Exception:
        pass


# -------------------------
# RealSense (color+depth)
# -------------------------
def open_realsense_stream(width=640, height=480, fps=30):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    profile = pipeline.start(config)
    print("✅ RealSense pipeline started (color + depth)")
    return pipeline, profile


def safe_read_realsense_frames(pipeline: rs.pipeline, timeout_ms=100):
    """Return (color_bgr, depth_z16) or (None, None) on failure."""
    try:
        frames = pipeline.wait_for_frames(timeout_ms=timeout_ms)
        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        if not color or not depth:
            return None, None
        return np.asanyarray(color.get_data()), np.asanyarray(depth.get_data())
    except Exception:
        return None, None


def make_opencv_writer(path: str, fps: float, size_wh, fourcc_str="mp4v", is_color=True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    w, h = size_wh
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h), isColor=is_color)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {path}. Try installing ffmpeg or use .avi.")
    return writer


# -------------------------
# ROS 2 bag record
# -------------------------
def start_rosbag_record(bag_dir: str, topics: list[str]):
    """
    Start ros2 bag recording into a directory.
    NOTE: This records as long as the process runs.
    """
    os.makedirs(os.path.dirname(bag_dir), exist_ok=True)

    cmd = ["ros2", "bag", "record", "-o", bag_dir] + topics
    print("▶️ Starting ros2 bag record:\n   " + " ".join(cmd))

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid,  # new process group
    )
    return proc


def main():
    out_root = "recordings"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(out_root, ts)
    os.makedirs(session_dir, exist_ok=True)

    # Output paths
    deck0_path = os.path.join(session_dir, f"{ts}_Endoscope_0.mp4")
    deck1_path = os.path.join(session_dir, f"{ts}_Endoscope_1.mp4")
    rs_color_path = os.path.join(session_dir, f"{ts}_RealSense_COLOR.mp4")
    rs_depth_path = os.path.join(session_dir, f"{ts}_RealSense_DEPTH.mp4")  # 8-bit visualization
    bag_dir = os.path.join(session_dir, f"{ts}_rosbag")

    # ROS topics to record
    topics = [
        "/PSM1/measured_cp",
        "/PSM1/measured_cv",
        "/PSM1/measured_js",
        "/PSM2/measured_cp",
        "/PSM2/measured_cv",
        "/PSM2/measured_js",
        "/console1/camera",
        "/console1/clutch",
    ]

    # Start DeckLink recordings (gst-launch)
    deck0_proc = start_decklink_gst_record(0, deck0_path, fps="30/1")
    deck1_proc = start_decklink_gst_record(1, deck1_path, fps="30/1")

    # Start RealSense recordings (OpenCV writers)
    rs_w, rs_h, rs_fps = 640, 480, 30
    rs_pipeline, rs_profile = open_realsense_stream(rs_w, rs_h, rs_fps)

    rs_color_writer = make_opencv_writer(rs_color_path, float(rs_fps), (rs_w, rs_h), "mp4v", is_color=True)
    rs_depth_writer = make_opencv_writer(rs_depth_path, float(rs_fps), (rs_w, rs_h), "mp4v", is_color=False)

    # Depth scaling (for visualization video)
    depth_max_m = 2.0
    depth_scale = float(rs_profile.get_device().first_depth_sensor().get_depth_scale())
    depth_max_units = int(depth_max_m / depth_scale)

    # Start rosbag recording
    bag_proc = start_rosbag_record(bag_dir, topics)

    print("🎥 Recording: DeckLink0/1 + RealSense color/depth + ROS2 bag")
    print("   Press ESC to stop.")

    show_preview = True
    t0 = time.time()
    frames_written = 0

    try:
        while True:
            color_bgr, depth_z16 = safe_read_realsense_frames(rs_pipeline, timeout_ms=100)

            if color_bgr is None or depth_z16 is None:
                color_bgr = np.zeros((rs_h, rs_w, 3), dtype=np.uint8)
                cv2.putText(color_bgr, "RealSense COLOR - No Signal", (20, rs_h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                depth_vis = np.zeros((rs_h, rs_w), dtype=np.uint8)
                cv2.putText(depth_vis, "RealSense DEPTH - No Signal", (20, rs_h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)
            else:
                # Enforce sizes
                if (color_bgr.shape[1], color_bgr.shape[0]) != (rs_w, rs_h):
                    color_bgr = cv2.resize(color_bgr, (rs_w, rs_h))
                if (depth_z16.shape[1], depth_z16.shape[0]) != (rs_w, rs_h):
                    depth_z16 = cv2.resize(depth_z16, (rs_w, rs_h), interpolation=cv2.INTER_NEAREST)

                # Depth -> 8-bit grayscale visualization for video
                depth_clip = np.clip(depth_z16, 0, depth_max_units).astype(np.uint16)
                depth_vis = (depth_clip.astype(np.float32) / float(depth_max_units) * 255.0).astype(np.uint8)

            # Write RealSense videos
            rs_color_writer.write(color_bgr)
            rs_depth_writer.write(depth_vis)
            frames_written += 1

            # Preview
            if show_preview:
                elapsed = time.time() - t0
                overlay = f"REC {elapsed:6.1f}s  (ESC to stop)"

                prev_color = color_bgr.copy()
                cv2.putText(prev_color, overlay, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                prev_depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                cv2.putText(prev_depth_color, overlay, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                preview = np.hstack([
                    cv2.resize(prev_color, (640, 480)),
                    cv2.resize(prev_depth_color, (640, 480)),
                ])
                cv2.imshow("RealSense Preview (Color | Depth)", preview)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

    finally:
        # Stop RealSense
        try:
            rs_color_writer.release()
        except Exception:
            pass
        try:
            rs_depth_writer.release()
        except Exception:
            pass
        try:
            rs_pipeline.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()

        # Stop rosbag (SIGINT)
        stop_process(bag_proc, "ros2 bag record")

        # Stop DeckLink recorders (finalize MP4)
        stop_process(deck0_proc, "DeckLink 0 recorder")
        stop_process(deck1_proc, "DeckLink 1 recorder")

        duration = max(time.time() - t0, 1e-6)
        print("🛑 Recording stopped.")
        print(f"   RealSense frames written: {frames_written} (~{frames_written / duration:.2f} fps)")
        print("   Saved session outputs in:", session_dir)
        print("   Files:")
        print(f"    - {deck0_path}")
        print(f"    - {deck1_path}")
        print(f"    - {rs_color_path}")
        print(f"    - {rs_depth_path}")
        print(f"    - {bag_dir}  (rosbag2 directory)")
        print("")
        print("ℹ️ Depth video is a visualization (8-bit). For raw depth, save Z16 frames or record a RealSense .bag.")


if __name__ == "__main__":
    main()
