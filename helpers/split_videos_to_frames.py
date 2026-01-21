#!/usr/bin/env python3
"""
Split 4 videos in a recording session folder into JPG frame sequences.

Assumptions about input naming (from prior recorder):
  <session_dir>/
    decklink0.mp4
    decklink1.mp4
    usb6.mp4
    usb8.mp4

Outputs:
  <session_dir>/frames/
    decklink0/frame_000000.jpg ...
    decklink1/frame_000000.jpg ...
    usb6/frame_000000.jpg ...
    usb8/frame_000000.jpg ...

Also checks (per file):
  - container-reported FPS via OpenCV CAP_PROP_FPS
  - frame count via CAP_PROP_FRAME_COUNT
  - duration estimate = frame_count / fps
  - warns if FPS deviates from 30 beyond tolerance
"""

import argparse
import os
import sys
import math
from typing import Dict, Tuple

import cv2


EXPECTED_FPS = 30.0


def is_close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def probe_video(path: str) -> Dict[str, float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    w = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0)
    h = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0)

    cap.release()

    duration = (frame_count / fps) if (fps and frame_count) else float("nan")

    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": w,
        "height": h,
        "duration_s": duration,
    }


def extract_frames(video_path: str, out_dir: str, jpeg_quality: int = 90) -> Tuple[int, float]:
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        out_path = os.path.join(out_dir, f"frame_{idx:06d}.jpg")
        cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        idx += 1

    cap.release()
    return idx, fps


def find_default_videos(session_dir: str) -> Dict[str, str]:
    candidates = {
        "decklink0": os.path.join(session_dir, "decklink0.mp4"),
        "decklink1": os.path.join(session_dir, "decklink1.mp4"),
        "usb6": os.path.join(session_dir, "usb6.mp4"),
        "usb8": os.path.join(session_dir, "usb8.mp4"),
    }
    existing = {k: v for k, v in candidates.items() if os.path.isfile(v)}
    return existing


def main():
    ap = argparse.ArgumentParser(
        description="Split a 4-camera recording folder into JPG frame sequences and verify FPS."
    )
    ap.add_argument(
        "session_dir",
        help="Path to the recording session folder (contains decklink0.mp4, decklink1.mp4, usb6.mp4, usb8.mp4).",
    )
    ap.add_argument(
        "--out-root",
        default="frames",
        help="Output subfolder name inside session_dir (default: frames).",
    )
    ap.add_argument(
        "--quality",
        type=int,
        default=90,
        help="JPEG quality (0-100), default 90.",
    )
    ap.add_argument(
        "--fps-tol",
        type=float,
        default=0.2,
        help="Allowed deviation from 30 fps before warning (default: 0.2).",
    )
    ap.add_argument(
        "--pattern",
        default="*.mp4",
        help="Unused placeholder; kept for future extension.",
    )
    args = ap.parse_args()

    session_dir = os.path.abspath(args.session_dir)
    if not os.path.isdir(session_dir):
        print(f"ERROR: not a directory: {session_dir}")
        return 2

    videos = find_default_videos(session_dir)
    if not videos:
        print("ERROR: no expected videos found in folder.")
        print("Expected names: decklink0.mp4, decklink1.mp4, usb6.mp4, usb8.mp4")
        return 2

    out_base = os.path.join(session_dir, args.out_root)
    os.makedirs(out_base, exist_ok=True)

    print(f"Session: {session_dir}")
    print(f"Output:  {out_base}")
    print("")

    # Probe first
    for name, vpath in videos.items():
        info = probe_video(vpath)
        fps = info["fps"]
        fc = info["frame_count"]
        dur = info["duration_s"]
        wh = f'{int(info["width"])}x{int(info["height"])}'

        fps_msg = f"{fps:.3f}" if fps and not math.isnan(fps) else "unknown"
        dur_msg = f"{dur:.2f}s" if dur and not math.isnan(dur) else "unknown"

        warn = ""
        if fps and not is_close(fps, EXPECTED_FPS, args.fps_tol):
            warn = f"  <-- WARNING: not ~{EXPECTED_FPS} fps (tol={args.fps_tol})"

        print(f"[{name}] {os.path.basename(vpath)} | {wh} | fps={fps_msg} | frames={int(fc)} | dur={dur_msg}{warn}")

    print("\nExtracting frames...\n")

    for name, vpath in videos.items():
        out_dir = os.path.join(out_base, name)
        frames_written, fps = extract_frames(vpath, out_dir, jpeg_quality=args.quality)

        warn = ""
        if fps and not is_close(fps, EXPECTED_FPS, args.fps_tol):
            warn = f" (WARNING: fps={fps:.3f} not ~{EXPECTED_FPS})"

        print(f"[{name}] wrote {frames_written} frames to {out_dir}{warn}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
