#!/usr/bin/env python3
"""
Record 4 camera streams into VIDEO files at 30 fps:

- DeckLink #0 and #1 via gst-launch-1.0 -> H.264 MP4
- USB cameras #6 and #8 via OpenCV VideoWriter -> MP4 (mp4v)

Output structure:
recordings/<session_id>/
  decklink0.mp4
  decklink1.mp4
  usb6.mp4
  usb8.mp4
"""

import os
import sys
import time
import signal
import subprocess
import threading
from dataclasses import dataclass
from typing import Optional, List

import cv2


# -----------------------------
# DeckLink (gst-launch) MP4 recorder
# -----------------------------
def start_decklink_gst_record_mp4(
    device_number: int,
    out_path: str,
    fps: str = "30/1",
    mode: str = "pal",
    bitrate_kbps: int = 8000,
    key_int_max: int = 30,
):
    """Record DeckLink -> H.264 MP4 using gst-launch (no OpenCV GStreamer needed)."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    cmd = [
        "gst-launch-1.0",
        "-e",
        "decklinkvideosrc", f"mode={mode}", f"device-number={device_number}",
        "!", "videorate",
        "!", f"video/x-raw,framerate={fps}",
        "!", "videoconvert",
        "!", "video/x-raw,format=I420",
        "!", "x264enc",
        "speed-preset=veryfast",
        "tune=zerolatency",
        f"bitrate={bitrate_kbps}",
        f"key-int-max={key_int_max}",
        "!", "h264parse",
        "!", "mp4mux",
        "!", "filesink", f"location={out_path}",
    ]

    print(f"\n[DeckLink{device_number}] START:\n  " + " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid,  # new process group
        text=True,
        bufsize=1,
    )
    return proc


def stop_gst_process(proc: subprocess.Popen, name: str, timeout_s: float = 3.0):
    if proc is None or proc.poll() is not None:
        return

    print(f"[{name}] STOP: sending SIGINT to gst process group...")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
    except ProcessLookupError:
        return

    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            print(f"[{name}] STOP: exited with code {proc.returncode}")
            return
        time.sleep(0.05)

    print(f"[{name}] STOP: SIGINT timeout -> sending SIGKILL...")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass


# -----------------------------
# USB (OpenCV) MP4 recorder thread
# -----------------------------
@dataclass
class UsbVideoRecorderConfig:
    device_index: int
    out_path: str
    target_fps: float = 30.0
    fourcc: str = "mp4v"   # widely supported; change to "avc1" if your OpenCV/FFmpeg supports it
    width: int = 0         # 0 = keep camera default
    height: int = 0        # 0 = keep camera default


class UsbVideoRecorder(threading.Thread):
    def __init__(self, cfg: UsbVideoRecorderConfig, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.stop_event = stop_event
        self.cap: Optional[cv2.VideoCapture] = None
        self.writer: Optional[cv2.VideoWriter] = None
        self.frames_written = 0

    def run(self):
        os.makedirs(os.path.dirname(self.cfg.out_path), exist_ok=True)

        # Prefer V4L2 on Linux
        self.cap = cv2.VideoCapture(self.cfg.device_index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.cfg.device_index)

        if not self.cap.isOpened():
            print(f"[USB{self.cfg.device_index}] ERROR: cannot open camera.")
            return

        # Best effort: request fps + optional resolution
        self.cap.set(cv2.CAP_PROP_FPS, float(self.cfg.target_fps))
        if self.cfg.width > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.cfg.width))
        if self.cfg.height > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.cfg.height))

        # Grab one frame to determine actual size
        ok, frame = self.cap.read()
        if not ok or frame is None:
            print(f"[USB{self.cfg.device_index}] ERROR: cannot read initial frame.")
            self.cap.release()
            return

        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*self.cfg.fourcc)
        self.writer = cv2.VideoWriter(self.cfg.out_path, fourcc, self.cfg.target_fps, (w, h), True)

        if not self.writer.isOpened():
            print(f"[USB{self.cfg.device_index}] ERROR: cannot open VideoWriter at {self.cfg.out_path}")
            self.cap.release()
            return

        print(f"[USB{self.cfg.device_index}] START -> {self.cfg.out_path} ({w}x{h} @ {self.cfg.target_fps} fps)")

        # Write the first frame we already grabbed
        self.writer.write(frame)
        self.frames_written = 1

        req_period = 1.0 / float(self.cfg.target_fps)
        next_t = time.perf_counter() + req_period

        while not self.stop_event.is_set():
            now = time.perf_counter()
            if now < next_t:
                time.sleep(min(0.002, next_t - now))
                continue

            ok, frame = self.cap.read()
            if not ok or frame is None:
                print(f"[USB{self.cfg.device_index}] WARN: frame grab failed.")
                time.sleep(0.05)
                next_t = time.perf_counter() + req_period
                continue

            # If resolution changes mid-stream, drop frames (keeps writer consistent)
            if frame.shape[1] != w or frame.shape[0] != h:
                print(f"[USB{self.cfg.device_index}] WARN: resolution changed; dropping frame.")
                next_t += req_period
                continue

            self.writer.write(frame)
            self.frames_written += 1
            next_t += req_period

        self.writer.release()
        self.cap.release()
        print(f"[USB{self.cfg.device_index}] STOP: wrote {self.frames_written} frames.")


# -----------------------------
# Main
# -----------------------------
def make_session_dir(base_dir: str = "recordings") -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(base_dir, ts)
    os.makedirs(session_dir, exist_ok=True)
    return session_dir


def main():
    decklinks = [0, 1]
    usb_devs = [6, 8]

    fps_str = "30/1"
    target_fps = 30.0

    session_dir = make_session_dir("recordings")
    print(f"\nSession directory: {session_dir}")

    # ---- start decklink recorders ----
    gst_procs: List[subprocess.Popen] = []
    for d in decklinks:
        out_path = os.path.join(session_dir, f"decklink{d}.mp4")
        proc = start_decklink_gst_record_mp4(
            device_number=d,
            out_path=out_path,
            fps=fps_str,
            mode="pal",
            bitrate_kbps=8000,
            key_int_max=30,
        )
        gst_procs.append(proc)

    # ---- start usb recorders ----
    stop_event = threading.Event()
    usb_threads: List[UsbVideoRecorder] = []
    
    for u in usb_devs:
        out_path = os.path.join(session_dir, f"usb{u}.mp4")
        cfg = UsbVideoRecorderConfig(
            device_index=u,
            out_path=out_path,
            target_fps=target_fps,
            fourcc="mp4v",
            width=0,
            height=0,
        )
        th = UsbVideoRecorder(cfg, stop_event)
        th.start()
        usb_threads.append(th)

    def handle_signal(signum, frame):
        print("\nSignal received -> stopping...")
        stop_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print("\nRecording... Press Ctrl+C to stop.\n")
    
    try:
        while not stop_event.is_set():
            time.sleep(0.2)
    finally:
        # Stop USB threads
        stop_event.set()
        for th in usb_threads:
            th.join(timeout=2.0)

        # Stop DeckLink gst pipelines
        for i, proc in enumerate(gst_procs):
            stop_gst_process(proc, name=f"DeckLink{decklinks[i]}")

        # Optional: print gst stderr if failures
        for i, proc in enumerate(gst_procs):
            if proc is not None and proc.poll() is not None and proc.returncode not in (0, None):
                try:
                    err = proc.stderr.read()
                    if err:
                        print(f"\n[DeckLink{decklinks[i]}] gst stderr:\n{err}\n")
                except Exception:
                    pass

        print(f"\nDone. Videos saved under: {session_dir}")


if __name__ == "__main__":
    sys.exit(main())
