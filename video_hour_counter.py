#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path

def get_duration_seconds(video_path: Path) -> float:
    """Return duration in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    print(out)
    return float(out)

def format_hhmmss(total_seconds: float) -> str:
    total_seconds_int = int(round(total_seconds))
    h = total_seconds_int // 3600
    m = (total_seconds_int % 3600) // 60
    s = total_seconds_int % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def main():
    print("SCRIPT STARTED")
    parser = argparse.ArgumentParser(description="Sum total duration of all mp4 videos in a folder.")
    parser.add_argument("folder", help="Path to folder containing .mp4 files")
    args = parser.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        raise SystemExit(f"Error: '{folder}' is not a valid folder.")

    mp4s = sorted(folder.glob("*.mp4"))
    if not mp4s:
        print("No .mp4 files found.")
        return

    total_seconds = 0.0
    failed = []

    for mp4 in mp4s:
        try:
            total_seconds += get_duration_seconds(mp4)
        except Exception:
            failed.append(mp4.name)

    total_hours = total_seconds / 3600.0
    print(f"Folder: {folder}")
    print(f"MP4 files counted: {len(mp4s) - len(failed)} / {len(mp4s)}")
    print(f"Total duration: {format_hhmmss(total_seconds)} (HH:MM:SS)")
    print(f"Total hours: {total_hours:.4f} h")

    if failed:
        print("\nCould not read duration for:")
        for name in failed:
            print(f" - {name}")

if __name__ == "__main__":
    main()
