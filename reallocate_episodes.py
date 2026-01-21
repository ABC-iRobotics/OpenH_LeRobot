#!/usr/bin/env python3
"""
Reallocate / delete episode subfolders based on a CSV (Episode_Number, Notation),
then renumber episodes to start from episode_001 in EACH of:
- base folder
- base_recovery sibling
- base_failure sibling

Rules (Notation column, case-insensitive):
- contains 'r' -> move episode folder to *_recovery
- contains 'f' -> move episode folder to *_failure
- contains 'd' -> delete episode folder
- contains 'p' -> do nothing
Precedence if multiple letters appear: d > r > f > p (d wins, then r, then f, else keep).

Example:
  python3 reallocate_episodes.py recordings/phantom_1/peg_transfer 2_Peg_Transfer_1Peg.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


EP_RE = re.compile(r"^episode_(\d{3,})$")


@dataclass
class Action:
    kind: str  # "keep" | "move_recovery" | "move_failure" | "delete"
    note: str


def parse_episode_index(ep_str: str) -> int:
    """
    Accepts:
      - "episode_001"
      - "001"
      - 1
    Returns int index (1-based).
    """
    s = str(ep_str).strip()
    if not s:
        raise ValueError("Empty Episode_Number")

    m = EP_RE.match(s)
    if m:
        return int(m.group(1))

    # allow plain digits like "001" or "1"
    if s.isdigit():
        return int(s)

    raise ValueError(f"Unrecognized Episode_Number format: {ep_str!r}")


def decide_action(notation: str) -> Action:
    n = (notation or "").strip().lower()

    # precedence: delete > recovery > failure > keep
    if "d" in n:
        return Action("delete", n)
    if "r" in n:
        return Action("move_recovery", n)
    if "f" in n:
        return Action("move_failure", n)
    return Action("keep", n)


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def move_dir(src: Path, dst_dir: Path) -> None:
    safe_mkdir(dst_dir)
    dst = dst_dir / src.name
    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")
    shutil.move(str(src), str(dst))


def delete_dir(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p)


def list_episode_dirs(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    eps = []
    for p in folder.iterdir():
        if p.is_dir() and EP_RE.match(p.name):
            eps.append(p)
    return sorted(eps, key=lambda x: int(EP_RE.match(x.name).group(1)))


def renumber_episodes(folder: Path) -> None:
    """
    Renumber episode_XXX subfolders inside 'folder' to episode_001..N (contiguous).
    Uses a two-phase rename via temporary names to avoid collisions.
    """
    eps = list_episode_dirs(folder)
    if not eps:
        return

    tmp_map: list[tuple[Path, Path]] = []
    final_map: list[tuple[Path, Path]] = []

    # phase 1: rename to temp
    for i, ep in enumerate(eps, start=1):
        tmp_name = f"__tmp_episode__{i:06d}"
        tmp_path = folder / tmp_name
        if tmp_path.exists():
            raise FileExistsError(f"Temp path already exists (unexpected): {tmp_path}")
        tmp_map.append((ep, tmp_path))

    for src, tmp in tmp_map:
        src.rename(tmp)

    # phase 2: rename temp to final episode_XXX
    tmps = sorted(folder.iterdir(), key=lambda p: p.name)
    tmps = [p for p in tmps if p.is_dir() and p.name.startswith("__tmp_episode__")]

    for i, tmp in enumerate(tmps, start=1):
        final = folder / f"episode_{i:03d}"
        if final.exists():
            raise FileExistsError(f"Final path already exists (unexpected): {final}")
        final_map.append((tmp, final))

    for tmp, final in final_map:
        tmp.rename(final)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=str, help="Base folder that contains episode_XXX subfolders")
    ap.add_argument("csv_file", type=str, help="CSV file with Episode_Number, Notation columns")
    ap.add_argument("--dry-run", action="store_true", help="Print actions, do not modify filesystem")
    args = ap.parse_args()

    base = Path(args.folder).expanduser().resolve()
    csv_path = Path(args.csv_file).expanduser().resolve()

    if not base.exists() or not base.is_dir():
        raise FileNotFoundError(f"Base folder not found: {base}")
    if not csv_path.exists() or not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    parent = base.parent
    recovery = parent / f"{base.name}_recovery"
    failure = parent / f"{base.name}_failure"

    # Read CSV mappings
    mapping: dict[int, Action] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header row.")
        # allow some flexibility in header names
        fn = {name.strip().lower(): name for name in reader.fieldnames}
        if "episode_number" not in fn or "notation" not in fn:
            raise ValueError(
                f"CSV must contain headers Episode_Number and Notation. Found: {reader.fieldnames}"
            )

        ep_col = fn["episode_number"]
        note_col = fn["notation"]

        for row in reader:
            ep_idx = parse_episode_index(row.get(ep_col, ""))
            action = decide_action(row.get(note_col, ""))
            mapping[ep_idx] = action

    # Apply actions on base episodes
    moved_r = moved_f = deleted = kept = missing = 0

    for ep_idx, action in sorted(mapping.items(), key=lambda kv: kv[0]):
        ep_name = f"episode_{ep_idx:03d}"
        ep_path = base / ep_name

        if not ep_path.exists():
            missing += 1
            print(f"[WARN] {ep_name} listed in CSV but not found in base folder.")
            continue

        if action.kind == "delete":
            print(f"[DEL ] {ep_path} (notation={action.note!r})")
            if not args.dry_run:
                delete_dir(ep_path)
            deleted += 1

        elif action.kind == "move_recovery":
            print(f"[MOVE] {ep_path} -> {recovery} (notation={action.note!r})")
            if not args.dry_run:
                move_dir(ep_path, recovery)
            moved_r += 1

        elif action.kind == "move_failure":
            print(f"[MOVE] {ep_path} -> {failure} (notation={action.note!r})")
            if not args.dry_run:
                move_dir(ep_path, failure)
            moved_f += 1

        else:
            # keep
            kept += 1

    # Renumber in each folder
    print("\n[RENUMBER] base")
    if not args.dry_run:
        renumber_episodes(base)

    print("[RENUMBER] recovery")
    if not args.dry_run:
        safe_mkdir(recovery)
        renumber_episodes(recovery)

    print("[RENUMBER] failure")
    if not args.dry_run:
        safe_mkdir(failure)
        renumber_episodes(failure)

    print("\n[SUMMARY]")
    print(f"  base:     {base}")
    print(f"  recovery: {recovery}")
    print(f"  failure:  {failure}")
    print(f"  moved to recovery: {moved_r}")
    print(f"  moved to failure:  {moved_f}")
    print(f"  deleted:           {deleted}")
    print(f"  kept (no-op):      {kept}")
    print(f"  missing episodes:  {missing}")
    print(f"  dry-run:           {args.dry_run}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
