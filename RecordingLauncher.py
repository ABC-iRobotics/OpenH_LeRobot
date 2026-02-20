#!/usr/bin/env python3
"""RecordingLauncher.py

Starts:
- daVinciFrameSequenceRecorder.py (VIDEO)  -> recordings/<session_name>/{decklink0,decklink1,usb6,usb8}/frame_*.jpg
- daVinciKinematicsRecorder.py    (ROS)    -> recordings_csv/dvrk_psm_meas_setpoint_<session_name>_30sps.csv

Adds:
- Episode CSV logging handled by VIDEO process (daVinciFrameSequenceRecorder.py):
  recordings_episodes/<subtask>.csv
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def _sigint(proc: subprocess.Popen):
    if proc and proc.poll() is None:
        try:
            proc.send_signal(signal.SIGINT)
        except ProcessLookupError:
            pass


def _sigkill(proc: subprocess.Popen):
    if proc and proc.poll() is None:
        try:
            proc.kill()
        except ProcessLookupError:
            pass


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def main() -> int:

    ##phantom = input("Phantom name (format: string_int): ").strip()
    ##subtask = input("Subtask name (format: int_string): ").strip()
    phantom = "Pork_1"
    subtask = "1_Dissect"
    #phantom = "Test_1"
    #subtask = "1_Test"

    if not phantom or not subtask:
        print("ERROR: phantom and subtask must be non-empty.")
        return 2

    root = _script_dir()
    root_data = Path("/data")
    video_py = root / "daVinciFrameSequenceRecorder.py"
    ros_py = root / "daVinciKinematicsRecorder.py"
    sync_py = root / "Synchronizer.py"

    # One shared session name used by VIDEO folder and ROS CSV filename
    session_name = time.strftime("%Y%m%d_%H%M%S")
    session_dir = root_data / "recordings" / session_name
    session_dir.mkdir(parents=True, exist_ok=True)

    # Ensure recordings_csv exists next to scripts (Synchronizer expects this)
    recordings_csv_dir = root_data / "recordings_csv"
    recordings_csv_dir.mkdir(parents=True, exist_ok=True)

    # NEW: Episode CSV folder next to recordings/ and recordings_csv/
    recordings_episodes_dir = root_data / "recordings_episodes"
    recordings_episodes_dir.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    env = os.environ.copy()

    phantomsubtask=phantom+"_"+subtask

    video_p = subprocess.Popen(
        [
            py,
            str(video_py),
            "--session-dir",
            str(session_dir),
            "--episodes-dir",
            str(recordings_episodes_dir),
            "--subtask",
            phantomsubtask,
        ],
        stdin=None,   # inherit TTY
        stdout=None,
        stderr=None,
        env=env,
    )

    ros_p = subprocess.Popen(
        [py, str(ros_py), "--sps", "30", "--out-dir", str(recordings_csv_dir), "--session-name", session_name],
        stdin=subprocess.DEVNULL,  # don't let ROS read from terminal
        stdout=None,
        stderr=None,
        env=env,
    )

    print(f"\nLauncher started:")
    print(f"  Phantom: {phantom}")
    print(f"  Subtask: {subtask}")
    print(f"  Session: {session_name}")
    print(f"  VIDEO pid={video_p.pid} (hotkeys: s/p/r/f/d/e/q)")
    print(f"  ROS   pid={ros_p.pid} (stdin disabled)")
    print("\nPress 'q' in the VIDEO terminal to finish the recording. Ctrl+C here also stops both.\n")

    try:
        while True:
            time.sleep(0.25)
            if video_p.poll() is not None:
                # video ended by 'q' (or crash) -> stop ROS and proceed
                break
            if ros_p.poll() is not None:
                # if ROS died, also stop video
                break
    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping both (SIGINT)...")
        _sigint(video_p)
        _sigint(ros_p)

        t0 = time.time()
        while time.time() - t0 < 3.0:
            if (video_p.poll() is not None) and (ros_p.poll() is not None):
                break
            time.sleep(0.05)

        if video_p.poll() is None or ros_p.poll() is None:
            print("Some process still running -> SIGKILL")
            _sigkill(video_p)
            _sigkill(ros_p)

    # Prompt for synchronizer
    try:
        ans = input(f"\nRun Synchronizer on session {session_name}? [y/N]: ").strip().lower()
    except EOFError:
        ans = ""

    if ans in ("y", "yes"):
        print("\nRunning Synchronizer...\n")
        cmd = [py, str(sync_py), "-i", str(session_dir), "--phantom", phantom, "--subtask", subtask]
        print("CMD:", " ".join(cmd))
        return subprocess.call(cmd, env=env)

    print("Done (Synchronizer skipped).\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
