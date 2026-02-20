#!/usr/bin/env python3
"""
Synchronizer.py — trim + synchronize recordings into dataset structure with MULTIPLE episodes.

MASTER TIMELINE (NEW DEFAULT):
- Kinematics CSV timestamps are the master.
- Cameras (decklink0, decklink1, usb6, usb8) are synchronized to kinematics timestamps.

Episodes:
- Still defined by decklink0 markers:
    frame_XXXXXX_start.jpg ... frame_YYYYYY_end.jpg
- For each episode, we compute a time window from decklink0 marker times,
  then take all CSV rows within that window as the output frame timeline.

Usage:
  python3 Synchronizer.py -i /path/to/recordings/YYYYMMDD_HHMMSS --phantom PHANTOM --subtask SUBTASK

Output:
  <script_dir>/dataset/<phantom>/<subtask>/episode_00X/
    left_img_dir/   (decklink0 synced to kinematics)
    right_img_dir/  (decklink1 synced to kinematics)
    endo_psm1/      (usb6 synced to kinematics)
    endo_psm2/      (usb8 synced to kinematics)
    ee_csv.csv      (kinematics rows used as master)
"""

import argparse
import csv
import os
import re
import shutil
import sys
from bisect import bisect_left
from pathlib import Path
from typing import List, Optional, Tuple


FRAME_RE = re.compile(r"^frame_(\d{6})(?:_(start|end))?\.(jpg|png)$", re.IGNORECASE)
SESSION_RE = re.compile(r"\d{8}_\d{6}")  # YYYYMMDD_HHMMSS

# Keep your old alignment knob: applies when using decklink file times as "episode window"
# and also when picking nearest decklink frames (so decklink can be shifted to align with CSV).
DECKLINK_TIME_OFFSET_NS = -100_000_000  # -100 ms


# ---------------- CLI / paths ----------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="Path to recordings session folder (e.g., .../YYYYMMDD_HHMMSS)")
    ap.add_argument("--phantom", required=True, help="phantom_name")
    ap.add_argument("--subtask", required=True, help="subtask_name")
    ap.add_argument(
        "--master",
        choices=["kinematics", "decklink0"],
        default="kinematics",
        help="Synchronization master timeline (default: kinematics).",
    )
    return ap.parse_args()


def resolve_session_folder(p: str) -> Path:
    src = Path(p).expanduser().resolve()
    if not src.exists() or not src.is_dir():
        raise SystemExit(f"Not a valid folder: {src}")
    return src


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def next_episode_index(base: Path) -> int:
    """
    base = dataset/<phantom>/<subtask>
    episode numbering starts at 001 (your file comment says 000, but code used 001 already).
    """
    base.mkdir(parents=True, exist_ok=True)

    indices = []
    for p in base.iterdir():
        if p.is_dir() and p.name.startswith("episode_"):
            try:
                indices.append(int(p.name.split("_")[1]))
            except ValueError:
                pass

    return max(indices) + 1 if indices else 1


def make_episode_dir(base: Path, idx: int) -> Path:
    ep = base / f"episode_{idx:03d}"
    ep.mkdir(parents=True, exist_ok=False)
    return ep


# ---------------- filesystem timestamp helpers ----------------

def frame_ctime_ns(folder: Path, fname: str) -> int:
    s = os.stat(os.path.join(folder, fname))
    return int(s.st_mtime_ns) #ctime NEM  creation time, hanem az utolsó metadata módosítás. mtime csak a file tartalmának módosításakor változik és a fájlnév változtatás az metadata változásnak számít


def list_frame_files_with_markers(folder: Path) -> List[Tuple[int, str, str]]:
    """
    Returns list of (index_from_name, filename, marker) sorted by index_from_name.
    marker: "", "start", "end"
    """
    out: List[Tuple[int, str, str]] = []
    if not folder.exists():
        return out
    for p in folder.iterdir():
        if not p.is_file():
            continue
        m = FRAME_RE.match(p.name)
        if not m:
            continue
        idx = int(m.group(1))
        marker = (m.group(2) or "").lower()
        out.append((idx, p.name, marker))
    out.sort(key=lambda x: x[0])
    return out


# ---------------- episode segmentation ----------------

def find_episodes_in_decklink0(decklink0: Path) -> List[Tuple[int, int]]:
    """
    Returns list of (start_pos, end_pos) indices into the sorted frames list.
    Each start is paired with the first end AFTER it.
    Episodes are non-overlapping; after closing an episode, search continues after its end.
    """
    frames = list_frame_files_with_markers(decklink0)
    if not frames:
        raise SystemExit(f"No matching frame_XXXXXX(.jpg) files found in: {decklink0}")

    episodes: List[Tuple[int, int]] = []
    i = 0
    while i < len(frames):
        start_pos = None
        while i < len(frames):
            if frames[i][2] == "start":
                start_pos = i
                break
            i += 1
        if start_pos is None:
            break

        end_pos = None
        j = start_pos + 1
        while j < len(frames):
            if frames[j][2] == "end":
                end_pos = j
                break
            j += 1
        if end_pos is None:
            raise SystemExit(f"Found *_start at {frames[start_pos][1]} but no *_end after it in {decklink0}")

        episodes.append((start_pos, end_pos))
        i = end_pos + 1

    if not episodes:
        raise SystemExit(f"No episodes found (no *_start/*_end pairs) in: {decklink0}")
    return episodes


# ---------------- generic nearest sync helpers ----------------

def is_frame_sequence_folder(folder: Path) -> bool:
    if not folder.exists() or not folder.is_dir():
        return False
    for p in folder.iterdir():
        if p.is_file() and FRAME_RE.match(p.name):
            return True
    return False


def build_time_index_for_folder(folder: Path, time_offset_ns: int = 0) -> Tuple[List[int], List[str]]:
    """
    Returns (times_ns_sorted, names_sorted_by_time). Optional time_offset_ns shifts all times.
    """
    items: List[Tuple[int, str]] = []
    for p in folder.iterdir():
        if not p.is_file():
            continue
        if not FRAME_RE.match(p.name):
            continue
        t_ns = frame_ctime_ns(folder, p.name) + time_offset_ns
        items.append((t_ns, p.name))
    items.sort(key=lambda x: x[0])
    return [t for t, _ in items], [n for _, n in items]


def nearest_by_time(times: List[int], names: List[str], target_ns: int) -> Optional[str]:
    if not times:
        return None
    i = bisect_left(times, target_ns)
    if i <= 0:
        return names[0]
    if i >= len(times):
        return names[-1]
    prev_t = times[i - 1]
    next_t = times[i]
    return names[i - 1] if abs(target_ns - prev_t) <= abs(next_t - target_ns) else names[i]


# ---------------- CSV helpers ----------------

def find_recordings_csv_input(session_name: str) -> Path:
    #folder = script_dir() / "recordings_csv"
    folder = Path("/data") / "recordings_csv"
    if not folder.exists() or not folder.is_dir():
        raise SystemExit(f'Missing folder next to script: "{folder}"')

    matches = sorted(
        [p for p in folder.iterdir()
         if p.is_file() and p.suffix.lower() == ".csv" and session_name in p.name]
    )
    if not matches:
        raise SystemExit(f'No CSV found in "{folder}" containing "{session_name}" in filename')

    matches.sort(key=lambda p: p.stat().st_mtime_ns, reverse=True)
    return matches[0]


def detect_timestamp_column(fieldnames: List[str]) -> int:
    lower = [f.strip().lower() for f in fieldnames]
    preferred_exact = ["timestamp", "time", "stamp", "t"]
    for key in preferred_exact:
        if key in lower:
            return lower.index(key)
    for i, f in enumerate(lower):
        if "timestamp" in f or (("time" in f) and ("frame" not in f)) or ("stamp" in f):
            return i
    raise SystemExit(f"Could not detect timestamp column in CSV header: {fieldnames}")


def parse_time_to_ns(v: str) -> int:
    s = v.strip()
    if s == "":
        raise ValueError("empty timestamp")
    ival = int(float(s))
    if ival > 10**14:       # ns
        return int(ival)
    return int(float(s) * 1e9)


def build_csv_time_index(csv_path: Path) -> Tuple[List[int], List[List[str]], List[str]]:
    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise SystemExit(f"Empty CSV: {csv_path}")

        ts_col_idx = detect_timestamp_column(header)

        rows: List[List[str]] = []
        times: List[int] = []
        for r in reader:
            if not r:
                continue
            if ts_col_idx >= len(r):
                continue
            try:
                t_ns = parse_time_to_ns(r[ts_col_idx])
            except Exception:
                continue
            rows.append(r)
            times.append(t_ns)

    if not times:
        raise SystemExit(f"No valid timestamped rows found in CSV: {csv_path}")

    # Ensure sorted by time
    if any(times[i] > times[i + 1] for i in range(len(times) - 1)):
        idx = sorted(range(len(times)), key=lambda i: times[i])
        times = [times[i] for i in idx]
        rows = [rows[i] for i in idx]

    return times, rows, header


def nearest_row_index(times: List[int], target_ns: int) -> int:
    i = bisect_left(times, target_ns)
    if i <= 0:
        return 0
    if i >= len(times):
        return len(times) - 1
    prev_t = times[i - 1]
    next_t = times[i]
    return i - 1 if abs(target_ns - prev_t) <= abs(next_t - target_ns) else i


def csv_rows_in_window(times: List[int], t0: int, t1: int) -> Tuple[int, int]:
    """
    Returns [i0, i1) indices for rows with times in [t0, t1].
    If the window would be empty, we expand to the nearest single row.
    """
    if t1 < t0:
        t0, t1 = t1, t0
        #print(t0, t1)
    i0 = bisect_left(times, t0)
    i1 = bisect_left(times, t1 + 1)  # inclusive end
    if i0 == i1:
        # fallback: include nearest row to midpoint
        mid = (t0 + t1) // 2
        j = nearest_row_index(times, mid)
        return j, j + 1
    return i0, i1


# ---------------- main ----------------

def main() -> int:
    args = parse_args()
    src = resolve_session_folder(args.input)

    # output base
    #ds_base = script_dir() / "dataset" / args.phantom / args.subtask
    ds_base = Path("/data") / "dataset" / args.phantom / args.subtask
    episode_idx = next_episode_index(ds_base)

    # input folders mapping
    in_dl0 = src / "decklink0"
    in_dl1 = src / "decklink1"
    in_usb6 = src / "usb6"
    in_usb8 = src / "usb8"
    in_rs_c = src / "realsense_color"
    in_rs_d = src / "realsense_depth"

    if not in_dl0.exists() or not in_dl0.is_dir():
        raise SystemExit(f"Missing folder: {in_dl0}")

    session_name = src.name
    if not SESSION_RE.search(session_name):
        raise SystemExit(f"Input folder name does not contain YYYYMMDD_HHMMSS: {session_name}")

    csv_in = find_recordings_csv_input(session_name)
    times_csv, rows_csv, header_csv = build_csv_time_index(csv_in)

    # Build camera time indices once
    # Apply DECKLINK_TIME_OFFSET_NS to decklink times so decklink aligns to CSV timebase.
    dl0_times, dl0_names = ([], [])
    dl1_times, dl1_names = ([], [])
    usb6_times, usb6_names = ([], [])
    usb8_times, usb8_names = ([], [])
    rs_c_times, rs_c_names = ([], [])
    rs_d_times, rs_d_names = ([], [])

    if in_dl0.exists() and is_frame_sequence_folder(in_dl0):
        dl0_times, dl0_names = build_time_index_for_folder(in_dl0, time_offset_ns=DECKLINK_TIME_OFFSET_NS)
    if in_dl1.exists() and is_frame_sequence_folder(in_dl1):
        dl1_times, dl1_names = build_time_index_for_folder(in_dl1, time_offset_ns=DECKLINK_TIME_OFFSET_NS)
    if in_usb6.exists() and is_frame_sequence_folder(in_usb6):
        usb6_times, usb6_names = build_time_index_for_folder(in_usb6, time_offset_ns=0)
    if in_usb8.exists() and is_frame_sequence_folder(in_usb8):
        usb8_times, usb8_names = build_time_index_for_folder(in_usb8, time_offset_ns=0)
    if in_rs_c.exists() and is_frame_sequence_folder(in_rs_c):
        rs_c_times, rs_c_names = build_time_index_for_folder(in_rs_c, time_offset_ns=0)
    if in_rs_d.exists() and is_frame_sequence_folder(in_rs_d):
        rs_d_times, rs_d_names = build_time_index_for_folder(in_rs_d, time_offset_ns=0)

    # decklink0 frames + episode segmentation (for markers)
    frames_sorted = list_frame_files_with_markers(in_dl0)
    episodes = find_episodes_in_decklink0(in_dl0)

    print(f"Input recordings: {src}")
    print(f"Found episodes in decklink0: {len(episodes)}")
    print(f"[sync] master = {args.master}")
    print(f"[sync] applying DECKLINK_TIME_OFFSET_NS to decklink time indices: {DECKLINK_TIME_OFFSET_NS/1e6:.1f} ms")

    for ep_local_i, (start_pos, end_pos) in enumerate(episodes):
        ep_dir = make_episode_dir(ds_base, episode_idx + ep_local_i)

        out_left = ep_dir / "left_img_dir"
        out_right = ep_dir / "right_img_dir"
        out_psm1 = ep_dir / "endo_psm1"
        out_psm2 = ep_dir / "endo_psm2"
        out_rs_c = ep_dir / "realsense_color"
        out_rs_d = ep_dir / "realsense_depth"
        out_csv = ep_dir / "ee_csv.csv"

        out_left.mkdir(parents=True, exist_ok=True)
        out_right.mkdir(parents=True, exist_ok=True)
        out_psm1.mkdir(parents=True, exist_ok=True)
        out_psm2.mkdir(parents=True, exist_ok=True)
        out_rs_c.mkdir(parents=True, exist_ok=True)
        out_rs_d.mkdir(parents=True, exist_ok=True)

        # Episode window determined from decklink0 marker file times (shifted to align with CSV)
        start_fname = frames_sorted[start_pos][1]
        end_fname = frames_sorted[end_pos][1]
        t_start = frame_ctime_ns(in_dl0, start_fname) + DECKLINK_TIME_OFFSET_NS
        t_end = frame_ctime_ns(in_dl0, end_fname) + DECKLINK_TIME_OFFSET_NS

        if args.master == "kinematics":
            i0, i1 = csv_rows_in_window(times_csv, t_start, t_end)
            master_times = times_csv[i0:i1]
            master_rows = rows_csv[i0:i1]
            master_row_indices = list(range(i0, i1))
        else:
            # old behavior: master = decklink0 (one output per dl0 frame within start..end)
            # We'll derive master times from the dl0 frames within the marker bounds.
            # (Uses dl0_times already shifted.)
            # Find dl0 frames whose (shifted) times are within [t_start, t_end]
            # Note: dl0_times are sorted by time, not by filename index.
            # We'll just scan and keep those in range.
            master_times = []
            master_rows = []
            master_row_indices = []
            for t in dl0_times:
                if t_start <= t <= t_end:
                    master_times.append(t)
            # match each master time to nearest csv row
            for t in master_times:
                j = nearest_row_index(times_csv, t)
                master_rows.append(rows_csv[j])
                master_row_indices.append(j)

        # Now sync each camera to each master time
        # Also store which source file got used for debugging.
        used_dl0 = []
        used_dl1 = []
        used_u6 = []
        used_u8 = []
        used_rs_c = []
        used_rs_d = []


        # Create a file to save the nearest timestamps
        # log_file = ep_dir / "nearest_timestamps_log.csv"
        # with log_file.open("w") as log:
        #    log.write("Timestamp (ns), Matched decklink0, Matched decklink1, Matched usb6, Matched usb8\n")

        seq = 0
        # Loop through each master time and find the nearest frames for each camera
        for t_ns in master_times:
            m0 = nearest_by_time(dl0_times, dl0_names, t_ns) if dl0_times else None
            m1 = nearest_by_time(dl1_times, dl1_names, t_ns) if dl1_times else None
            m6 = nearest_by_time(usb6_times, usb6_names, t_ns) if usb6_times else None
            m8 = nearest_by_time(usb8_times, usb8_names, t_ns) if usb8_times else None
            mrc = nearest_by_time(rs_c_times, rs_c_names, t_ns) if rs_c_times else None
            mrd = nearest_by_time(rs_d_times, rs_d_names, t_ns) if rs_d_times else None

            # Log the found nearest timestamps
            # log_entry = f"{t_ns}, {m0 or 'None'}, {m1 or 'None'}, {m6 or 'None'}, {m8 or 'None'}\n"
            # with log_file.open("a") as log:
            #    log.write(log_entry)
            
            # Proceed with copying files as usual
            if m0:
                shutil.copy2(in_dl0 / m0, out_left / f"frame_{seq:06d}.jpg")
            if m1:
                shutil.copy2(in_dl1 / m1, out_right / f"frame_{seq:06d}.jpg")
            if m6:
                shutil.copy2(in_usb6 / m6, out_psm1 / f"frame_{seq:06d}.jpg")
            if m8:
                shutil.copy2(in_usb8 / m8, out_psm2 / f"frame_{seq:06d}.jpg")
            if mrc:
                shutil.copy2(in_rs_c / mrc, out_rs_c / f"frame_{seq:06d}.jpg")
            if mrd:
                shutil.copy2(in_rs_d / mrd, out_rs_d / f"frame_{seq:06d}.png")

            used_dl0.append(m0 or "")
            used_dl1.append(m1 or "")
            used_u6.append(m6 or "")
            used_u8.append(m8 or "")
            used_rs_c.append(mrc or "")
            used_rs_d.append(mrd or "")

            seq += 1

        # Write CSV: kinematics rows are the ground truth rows used
        out_header = [
            "frame_index",
            "sync_time_ns",
            "csv_row_index",
            "matched_decklink0_fname",
            "matched_decklink1_fname",
            "matched_usb6_fname",
            "matched_usb8_fname",
            "matched_realsense_color_fname",
            "matched_realsense_depth_fname",
        ] + header_csv

        with out_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(out_header)
            for k in range(len(master_times)):
                w.writerow([
                    k,
                    master_times[k],
                    master_row_indices[k] if k < len(master_row_indices) else "",
                    used_dl0[k],
                    used_dl1[k],
                    used_u6[k],
                    used_u8[k],
                    used_rs_c[k],
                    used_rs_d[k],
                ] + master_rows[k])

        print(
            f"[episode_{episode_idx + ep_local_i:03d}] "
            f"{start_fname} .. {end_fname}  "
            f"(window {min(t_start,t_end)}..{max(t_start,t_end)} ns) "
            f"-> {len(master_times)} frames -> {ep_dir}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
