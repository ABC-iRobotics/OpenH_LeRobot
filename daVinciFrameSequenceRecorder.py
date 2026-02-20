#!/usr/bin/env python3
"""
Record camera streams into frame sequences at 30 fps:

- DeckLink #0 and #1 via gst-launch-1.0 -> multifilesink (JPGs)
- USB cameras via OpenCV (device #6 and #8) -> cv2.imwrite (JPGs)
- RealSense (color + optional depth) via pyrealsense2 -> JPG/PNG sequences

Hotkeys:
  s -> the NEXT saved frame in recordings/.../decklink0/ gets "_start" suffix AND increments episode counter
  e -> the NEXT saved frame in recordings/.../decklink0/ gets "_end" suffix (no CSV logging, backward compatible)
  p/r/f/d -> the NEXT saved frame in recordings/.../decklink0/ gets "_end" suffix AND logs "<episode>, <key>" into CSV
  q -> stop recording

Episode CSV:
  recordings_episodes/<subtask>.csv
  columns: Episode_Number, Notation
"""

import os
import argparse
import sys
import time
import signal
import subprocess
import threading
from dataclasses import dataclass
from typing import Optional, List, Deque, Tuple
from collections import deque
import re
import select
import termios
import tty
import cv2
import csv

# --- RealSense (optional) ---
try:
    import pyrealsense2 as rs
    _HAVE_REALSENSE = True
except Exception:
    rs = None
    _HAVE_REALSENSE = False


# -----------------------------
# Episode CSV logger (integrated)
# -----------------------------
class EpisodeCsvLogger:
    def __init__(self, episodes_dir: str, subtask: str):
        self.episodes_dir = os.path.abspath(os.path.expanduser(episodes_dir))
        self.subtask = subtask.strip()
        if not self.subtask:
            raise ValueError("subtask must be non-empty")
        os.makedirs(self.episodes_dir, exist_ok=True)
        self.csv_path = os.path.join(self.episodes_dir, f"{self.subtask}.csv")

        self._ensure_header()

    def _ensure_header(self):
        if os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0:
            return
        with open(self.csv_path, mode="w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Episode_Number", "Notation"])

    def get_last_episode_number(self) -> int:
        if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
            return 0

        last_ep = 0
        try:
            with open(self.csv_path, mode="r", newline="") as f:
                r = csv.reader(f)
                header_skipped = False
                for row in r:
                    if not row:
                        continue
                    if not header_skipped:
                        header_skipped = True
                        # expect header in first row; continue
                        continue
                    if len(row) < 1:
                        continue
                    ep_str = (row[0] or "").strip()
                    # expected "episode_###"
                    if ep_str.startswith("episode_"):
                        try:
                            n = int(ep_str.split("_", 1)[1])
                            if n > last_ep:
                                last_ep = n
                        except Exception:
                            pass
        except Exception:
            return 0
        return last_ep

    def append(self, episode_number: int, notation: str):
        with open(self.csv_path, mode="a", newline="") as f:
            w = csv.writer(f)
            w.writerow([f"episode_{episode_number:03d}", notation])


# -----------------------------
# DeckLink (gst-launch) JPG recorder
# -----------------------------
def start_decklink_gst_jpg_sequence(
    device_number: int,
    out_dir: str,
    fps: str = "30/1",
    mode: str = "pal",
    pattern: str = "frame_%06d.jpg",
    preview: bool = True,
    preview_sink: str = "glimagesink",   # or "autovideosink", "xvimagesink"
):
    os.makedirs(out_dir, exist_ok=True)
    location = os.path.join(out_dir, pattern)

    cmd = [
        "gst-launch-1.0",
        "-e",
        "decklinkvideosrc", f"mode={mode}", f"device-number={device_number}",
        "!", "videorate",
        "!", f"video/x-raw,framerate={fps}",
        "!", "videoconvert",
    ]

    if preview:
        cmd += [
            "!", "tee", "name=t",

            "t.", "!", "queue",
            "!", "jpegenc", "quality=90",
            "!", "multifilesink", f"location={location}",
            "post-messages=true",
            "async=false",

            "t.", "!", "queue",
            "!", preview_sink, "sync=false",
        ]
    else:
        cmd += [
            "!", "jpegenc", "quality=90",
            "!", "multifilesink", f"location={location}",
            "post-messages=true",
            "async=false",
        ]

    print(f"\n[DeckLink{device_number}] START:\n  " + " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid,
        text=True,
        bufsize=1,
    )
    return proc


def _get_newest_windows(limit=10, title_substring="OpenGL"):
    # wmctrl -l: WID DESK HOST TITLE...
    time.sleep(0.5)
    out = subprocess.check_output(["wmctrl", "-l"], text=True)
    wids = []
    for line in out.splitlines():
        if title_substring in line:
            wid = line.split()[0]
            wids.append(wid)

    # keep only the last `limit` ones (wmctrl output is usually oldest->newest)
    return wids[-limit:][::-1]


def _place_wid(wid, x, y, w, h):
    subprocess.run(["wmctrl", "-ir", wid, "-e", f"0,{x},{y},{w},{h}"], check=False)
    subprocess.run(["wmctrl", "-ir", wid, "-b", "add,fullscreen"], check=False)


def place_two_latest_preview_windows():
    wids = _get_newest_windows(limit=20)
    print(len(wids))
    if len(wids) < 2:
        print("[wmctrl] Not enough windows found to place previews.")
        return False

    wid1, wid0 = wids[1], wids[0]
    wid_right, wid_left = sorted([wid0, wid1], key=lambda w: int(w, 16))
    wid_right, wid_left = wid0, wid1

    _place_wid(wid_right, 0, 0, 1024, 768)
    _place_wid(wid_left, 1024, 0, 1024, 768)
    print(f"[wmctrl] Placed windows {wid_left} and {wid_right}")
    return True


def stop_gst_process(proc: subprocess.Popen, name: str, timeout_s: float = 3.0):
    if proc is None:
        return
    if proc.poll() is not None:
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
# DeckLink0 marker/renamer (press hotkeys -> rename NEXT frame in decklink0)
# -----------------------------
_FRAME_RE = re.compile(r"^frame_(\d{6})(?:_(start|end))?\.jpg$")


def _get_max_frame_idx(decklink0_dir: str) -> int:
    try:
        names = os.listdir(decklink0_dir)
    except FileNotFoundError:
        return -1

    mx = -1
    for n in names:
        m = _FRAME_RE.match(n)
        if not m:
            continue
        idx = int(m.group(1))
        if idx > mx:
            mx = idx
    return mx


class Decklink0Marker:
    def __init__(self, decklink0_dir: str, stop_event: threading.Event):
        self.dir = decklink0_dir
        self.stop_event = stop_event
        self.q: Deque[Tuple[int, str]] = deque()
        self.lock = threading.Lock()
        self.watcher = threading.Thread(target=self._watch_loop, daemon=True)

    def start(self):
        os.makedirs(self.dir, exist_ok=True)
        self.watcher.start()

    def schedule_next(self, tag: str):
        tag = tag.strip()
        if tag not in ("_start", "_end"):
            return

        with self.lock:
            current_max = _get_max_frame_idx(self.dir)
            queued_max = self.q[-1][0] if self.q else -1
            target = max(current_max, queued_max) + 1
            self.q.append((target, tag))

        print(f"[decklink0] queued {tag} for next frame -> frame_{target:06d}.jpg")

    def _watch_loop(self):
        while not self.stop_event.is_set():
            with self.lock:
                item = self.q[0] if self.q else None

            if item is None:
                time.sleep(0.01)
                continue

            idx, tag = item
            src = os.path.join(self.dir, f"frame_{idx:06d}.jpg")
            if not os.path.exists(src):
                time.sleep(0.005)
                continue

            dst = os.path.join(self.dir, f"frame_{idx:06d}{tag}.jpg")
            if os.path.exists(dst):
                dst = os.path.join(self.dir, f"frame_{idx:06d}{tag}_{int(time.time()*1000)}.jpg")

            try:
                os.rename(src, dst)
            except Exception as ex:
                print(f"[decklink0] WARN: rename failed for {os.path.basename(src)} -> {ex}")

            with self.lock:
                if self.q and self.q[0] == item:
                    self.q.popleft()


def keyboard_listener(
    marker: Decklink0Marker,
    stop_event: threading.Event,
    episode_logger: Optional[EpisodeCsvLogger],
    initial_episode_counter: int,
):
    episode_counter = int(initial_episode_counter)

    if not sys.stdin.isatty():
        print("[keyboard] WARN: stdin is not a TTY; hotkeys disabled.")
        return

    end_keys = {"p", "r", "f", "d"}

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        print("[keyboard] Hotkeys: 's'-> _start (+episode++), 'p/r/f/d'-> _end + CSV log, 'e'-> _end only, 'q'-> stop")
        if episode_logger is not None:
            print(f"[episodes] CSV: {episode_logger.csv_path} (starting from episode_{episode_counter:03d})")
        while not stop_event.is_set():
            r, _, _ = select.select([sys.stdin], [], [], 0.05)
            if not r:
                continue
            ch = sys.stdin.read(1)

            if ch == "s":
                marker.schedule_next("_start")
                episode_counter += 1
                print(f"Episode #{episode_counter:03d} started")
            elif ch in end_keys:
                marker.schedule_next("_end")
                if episode_logger is None:
                    print(f"[episodes] WARN: no CSV logger configured; key '{ch}' not logged.")
                else:
                    if episode_counter <= 0:
                        print(f"[episodes] WARN: got '{ch}' but no episode started yet (press 's' first). Not logging.")
                    else:
                        episode_logger.append(episode_counter, ch)
                        print(f"[episodes] episode_{episode_counter:03d} <- '{ch}'")
            #elif ch == "e":
                # backward-compatible end marker (no notation logging)
                # marker.schedule_next("_end")
            elif ch == "q":
                print("[keyboard] 'q' pressed -> stopping recording...")
                stop_event.set()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# -----------------------------
# USB (OpenCV) JPG recorder thread
# -----------------------------
@dataclass
class UsbRecorderConfig:
    device_index: int
    out_dir: str
    target_fps: float = 30.0
    jpeg_quality: int = 90
    frame_pattern: str = "frame_%06d.jpg"
    width: int = 0
    height: int = 0


class UsbRecorder(threading.Thread):
    def __init__(self, cfg: UsbRecorderConfig, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.stop_event = stop_event
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_idx = 0

    def run(self):
        os.makedirs(self.cfg.out_dir, exist_ok=True)

        self.cap = cv2.VideoCapture(self.cfg.device_index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.cfg.device_index)

        if not self.cap.isOpened():
            print(f"[USB{self.cfg.device_index}] ERROR: cannot open camera.")
            return

        self.cap.set(cv2.CAP_PROP_FPS, float(self.cfg.target_fps))
        if self.cfg.width > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.cfg.width))
        if self.cfg.height > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.cfg.height))

        req_period = 1.0 / float(self.cfg.target_fps)
        next_t = time.perf_counter()

        print(f"[USB{self.cfg.device_index}] START -> {self.cfg.out_dir}")

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

            fname = self.cfg.frame_pattern % self.frame_idx
            out_path = os.path.join(self.cfg.out_dir, fname)

            cv2.imwrite(
                out_path,
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(self.cfg.jpeg_quality)],
            )

            self.frame_idx += 1
            next_t += req_period

        self.cap.release()
        print(f"[USB{self.cfg.device_index}] STOP: saved {self.frame_idx} frames.")


# -----------------------------
# RealSense recorder thread (color + optional depth)
# -----------------------------
@dataclass
class RealSenseRecorderConfig:
    out_dir_color: str
    out_dir_depth: str
    target_fps: float = 30.0
    jpeg_quality: int = 90
    color_width: int = 1280
    color_height: int = 720
    depth_width: int = 1280
    depth_height: int = 720
    serial: str = ""          # optional: force a specific device
    enable_depth: bool = False
    frame_pattern_color: str = "frame_%06d.jpg"
    frame_pattern_depth: str = "frame_%06d.png"  # 16-bit PNG for Z16 depth


class RealSenseRecorder(threading.Thread):
    def __init__(self, cfg: RealSenseRecorderConfig, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.stop_event = stop_event
        self.frame_idx = 0
        self.pipeline = None

    def _device_available(self) -> bool:
        if not _HAVE_REALSENSE:
            return False
        try:
            ctx = rs.context()
            devs = ctx.query_devices()
            return len(devs) > 0
        except Exception:
            return False

    def run(self):
        if not _HAVE_REALSENSE:
            print("[RealSense] WARN: pyrealsense2 not available. Skipping RealSense recording.")
            return
        if not self._device_available():
            print("[RealSense] WARN: No RealSense device detected. Skipping RealSense recording.")
            return

        os.makedirs(self.cfg.out_dir_color, exist_ok=True)
        if self.cfg.enable_depth:
            os.makedirs(self.cfg.out_dir_depth, exist_ok=True)

        print(f"[RealSense] START -> {self.cfg.out_dir_color}" + (" + depth" if self.cfg.enable_depth else ""))

        try:
            self.pipeline = rs.pipeline()
            rs_cfg = rs.config()

            if self.cfg.serial.strip():
                rs_cfg.enable_device(self.cfg.serial.strip())

            rs_cfg.enable_stream(
                rs.stream.color,
                int(self.cfg.color_width),
                int(self.cfg.color_height),
                rs.format.bgr8,
                int(self.cfg.target_fps),
            )

            if self.cfg.enable_depth:
                rs_cfg.enable_stream(
                    rs.stream.depth,
                    int(self.cfg.depth_width),
                    int(self.cfg.depth_height),
                    rs.format.z16,
                    int(self.cfg.target_fps),
                )

            self.pipeline.start(rs_cfg)

            req_period = 1.0 / float(self.cfg.target_fps)
            next_t = time.perf_counter()

            while not self.stop_event.is_set():
                now = time.perf_counter()
                if now < next_t:
                    time.sleep(min(0.002, next_t - now))
                    continue

                try:
                    frames = self.pipeline.wait_for_frames(timeout_ms=200)
                except Exception:
                    next_t = time.perf_counter() + req_period
                    continue

                color_frame = frames.get_color_frame()
                if not color_frame:
                    next_t = time.perf_counter() + req_period
                    continue

                color = color_frame.get_data()
                color_np = None
                try:
                    import numpy as np
                    color_np = np.asanyarray(color)
                except Exception as ex:
                    print(f"[RealSense] ERROR: numpy required for RealSense frames: {ex}")
                    break

                fname_c = self.cfg.frame_pattern_color % self.frame_idx
                out_c = os.path.join(self.cfg.out_dir_color, fname_c)
                cv2.imwrite(out_c, color_np, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.cfg.jpeg_quality)])

                if self.cfg.enable_depth:
                    depth_frame = frames.get_depth_frame()
                    if depth_frame:
                        depth_np = np.asanyarray(depth_frame.get_data())
                        fname_d = self.cfg.frame_pattern_depth % self.frame_idx
                        out_d = os.path.join(self.cfg.out_dir_depth, fname_d)
                        cv2.imwrite(out_d, depth_np)

                self.frame_idx += 1
                next_t += req_period

        except Exception as ex:
            print(f"[RealSense] ERROR: {ex}")
        finally:
            try:
                if self.pipeline is not None:
                    self.pipeline.stop()
            except Exception:
                pass
            print(f"[RealSense] STOP: saved {self.frame_idx} frames.")


# -----------------------------
# Main
# -----------------------------
def make_session_dir(base_dir: str = "recordings", session_dir: Optional[str] = None) -> str:
    if session_dir:
        session_dir = os.path.abspath(os.path.expanduser(session_dir))
        os.makedirs(session_dir, exist_ok=True)
        return session_dir
    ts = time.strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(base_dir, ts)
    os.makedirs(session_dir, exist_ok=True)
    return session_dir


def main():
    parser = argparse.ArgumentParser(description="Record DeckLink + USB + RealSense into frame sequences")
    parser.add_argument("--session-dir", default="", help="Optional pre-created session directory to write into")
    parser.add_argument("--base-dir", default="recordings", help="Base dir for auto-created sessions (default: recordings)")

    # NEW: episode CSV config
    parser.add_argument("--episodes-dir", default="recordings_episodes", help="Directory for episode CSVs")
    parser.add_argument("--subtask", default="", help="Subtask name (CSV will be <subtask>.csv)")

    # --- RealSense options ---
    parser.add_argument("--no-realsense", action="store_true", help="Disable RealSense recording")
    parser.add_argument("--rs-serial", default="", help="Optional RealSense device serial to lock onto")
    parser.add_argument("--no-rs-depth", action="store_true", help="Disable depth frames")
    parser.add_argument("--rs-color-w", type=int, default=1280, help="RealSense color width")
    parser.add_argument("--rs-color-h", type=int, default=720, help="RealSense color height")
    parser.add_argument("--rs-depth-w", type=int, default=1280, help="RealSense depth width")
    parser.add_argument("--rs-depth-h", type=int, default=720, help="RealSense depth height")

    args = parser.parse_args()

    decklinks = [0, 1]
    usb_devs = [6, 8]
    fps_str = "30/1"
    target_fps = 30.0

    session_dir = make_session_dir(args.base_dir, session_dir=(args.session_dir or None))
    print(f"\nSession directory: {session_dir}")

    decklink_dirs = {d: os.path.join(session_dir, f"decklink{d}") for d in decklinks}
    usb_dirs = {u: os.path.join(session_dir, f"usb{u}") for u in usb_devs}

    rs_color_dir = os.path.join(session_dir, "realsense_color")
    rs_depth_dir = os.path.join(session_dir, "realsense_depth")

    # Setup episode logger (only if subtask provided)
    episode_logger: Optional[EpisodeCsvLogger] = None
    initial_episode_counter = 0
    if args.subtask.strip():
        try:
            episode_logger = EpisodeCsvLogger(args.episodes_dir, args.subtask)
            initial_episode_counter = episode_logger.get_last_episode_number()
        except Exception as ex:
            print(f"[episodes] WARN: could not initialize episode CSV logger: {ex}")
            episode_logger = None
            initial_episode_counter = 0
    else:
        print("[episodes] NOTE: --subtask not provided; episode CSV logging disabled.")

    gst_procs: List[subprocess.Popen] = []

    proc0 = start_decklink_gst_jpg_sequence(
        device_number=0,
        out_dir=decklink_dirs[0],
        fps=fps_str,
        mode="pal",
        pattern="frame_%06d.jpg",
    )
    gst_procs.append(proc0)
    time.sleep(0.1)

    proc1 = start_decklink_gst_jpg_sequence(
        device_number=1,
        out_dir=decklink_dirs[1],
        fps=fps_str,
        mode="pal",
        pattern="frame_%06d.jpg",
    )
    gst_procs.append(proc1)
    time.sleep(0.1)

    place_two_latest_preview_windows()

    stop_event = threading.Event()

    # --- start decklink0 marker + keyboard threads ---
    decklink0_marker = Decklink0Marker(decklink_dirs[0], stop_event)
    decklink0_marker.start()
    kb_thread = threading.Thread(
        target=keyboard_listener,
        args=(decklink0_marker, stop_event, episode_logger, initial_episode_counter),
        daemon=True,
    )
    kb_thread.start()

    # --- USB threads ---
    usb_threads: List[UsbRecorder] = []
    for u in usb_devs:
        cfg = UsbRecorderConfig(
            device_index=u,
            out_dir=usb_dirs[u],
            target_fps=target_fps,
            jpeg_quality=90,
            frame_pattern="frame_%06d.jpg",
            width=0,
            height=0,
        )
        th = UsbRecorder(cfg, stop_event)
        th.start()
        usb_threads.append(th)

    # --- RealSense thread (optional) ---
    rs_thread: Optional[RealSenseRecorder] = None
    if not args.no_realsense:
        rs_cfg = RealSenseRecorderConfig(
            out_dir_color=rs_color_dir,
            out_dir_depth=rs_depth_dir,
            target_fps=target_fps,
            jpeg_quality=90,
            color_width=args.rs_color_w,
            color_height=args.rs_color_h,
            depth_width=args.rs_depth_w,
            depth_height=args.rs_depth_h,
            serial=args.rs_serial,
            enable_depth=not args.no_rs_depth,
        )
        rs_thread = RealSenseRecorder(rs_cfg, stop_event)
        rs_thread.start()
    else:
        print("[RealSense] Disabled (--no-realsense).")

    def handle_signal(signum, frame):
        print("\nSignal received -> stopping...")
        stop_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print("\nRecording... Hotkeys: s / p r f d / e / q. Ctrl+C also works.\n")

    try:
        while not stop_event.is_set():
            time.sleep(0.2)
    finally:
        for i, proc in enumerate(gst_procs):
            stop_gst_process(proc, name=f"DeckLink{decklinks[i]}")

        stop_event.set()
        for th in usb_threads:
            th.join(timeout=2.0)

        if rs_thread is not None:
            rs_thread.join(timeout=2.0)

        for i, proc in enumerate(gst_procs):
            if proc is not None and proc.poll() is not None and proc.returncode not in (0, None):
                try:
                    err = proc.stderr.read()
                    if err:
                        print(f"\n[DeckLink{decklinks[i]}] gst stderr:\n{err}\n")
                except Exception:
                    pass

        print(f"\nDone. Frames saved under: {session_dir}")
        if episode_logger is not None:
            print(f"[episodes] Logged to: {episode_logger.csv_path}")


if __name__ == "__main__":
    sys.exit(main())
