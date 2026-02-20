#!/usr/bin/env python3
"""
Copy-based episode reallocator based on CSV (Episode_Number, Notation).

NEW notation rules (case-insensitive, new list ONLY):
  - 't' -> 1_NeedleInsertion
  - 'a' -> 2_SuturePulling
  - 'd' -> 3_DoubleKnot
  - 's' -> 4_SingleKnot
  - 'k' -> ignore that episode (no copy)

Behavior:
  - Keeps original base folder intact (no moves/deletes/renames in base).
  - Copies episode_XXX folders into the target task folders.
  - Optionally renumbers episodes inside each target task folder to episode_001..N
    (without affecting the source).

Usage:
  python3 reallocate_tasks_copy.py /path/to/base_folder labels.csv --dry-run
  python3 reallocate_tasks_copy.py /path/to/base_folder labels.csv --out-root /somewhere/else
  python3 reallocate_tasks_copy.py /path/to/base_folder labels.csv --overwrite

CSV must contain headers: Episode_Number, Notation
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


EP_RE = re.compile(r"^episode_(\d{3,})$")

TASK_FOLDERS = {
    "t": "1_NeedleInsertion",
    "a": "2_SuturePulling",
    "d": "3_DoubleKnot",
    "s": "4_SingleKnot",
    "k": None,  # ignore
}


@dataclass
class Decision:
    key: str | None   # one of t/a/d/s/k or None if unrecognized
    raw: str          # raw notation for logging


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

    if s.isdigit():
        return int(s)

    raise ValueError(f"Unrecognized Episode_Number format: {ep_str!r}")


def decide_task_key(notation: str) -> Decision:
    """
    Extract the FIRST relevant key letter (t/a/d/s/k) that appears in the notation string.
    If none found -> key=None (warn + skip).
    """
    raw = (notation or "")
    n = raw.strip().lower()

    for ch in n:
        if ch in TASK_FOLDERS:
            return Decision(ch, raw)

    return Decision(None, raw)


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


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
        tmp.rename(final)


def copy_episode(src_ep: Path, dst_ep: Path, overwrite: bool) -> None:
    if dst_ep.exists():
        if not overwrite:
            raise FileExistsError(f"Destination already exists: {dst_ep} (use --overwrite)")
        shutil.rmtree(dst_ep)
    shutil.copytree(src_ep, dst_ep)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=str, help="Base folder that contains episode_XXX subfolders (SOURCE; unchanged)")
    ap.add_argument("csv_file", type=str, help="CSV file with Episode_Number, Notation columns")
    ap.add_argument("--out-root", type=str, default=None,
                    help="Where to create the task folders (default: base folder's parent directory)")
    ap.add_argument("--dry-run", action="store_true", help="Print actions, do not modify filesystem")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite already-copied episode folders in targets")
    ap.add_argument("--no-renumber", action="store_true",
                    help="Do not renumber episodes inside each target folder after copying")
    args = ap.parse_args()

    base = Path(args.folder).expanduser().resolve()
    csv_path = Path(args.csv_file).expanduser().resolve()

    if not base.exists() or not base.is_dir():
        raise FileNotFoundError(f"Base folder not found: {base}")
    if not csv_path.exists() or not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    out_root = Path(args.out_root).expanduser().resolve() if args.out_root else base.parent
    safe_mkdir(out_root)

    # Read CSV mappings
    mapping: dict[int, Decision] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header row.")
        fn = {name.strip().lower(): name for name in reader.fieldnames}
        if "episode_number" not in fn or "notation" not in fn:
            raise ValueError(
                f"CSV must contain headers Episode_Number and Notation. Found: {reader.fieldnames}"
            )

        ep_col = fn["episode_number"]
        note_col = fn["notation"]

        for row in reader:
            ep_idx = parse_episode_index(row.get(ep_col, ""))
            dec = decide_task_key(row.get(note_col, ""))
            mapping[ep_idx] = dec

    copied = ignored = missing = unrecognized = 0
    touched_folders: set[Path] = set()

    for ep_idx, dec in sorted(mapping.items(), key=lambda kv: kv[0]):
        ep_name = f"episode_{ep_idx:03d}"
        src = base / ep_name

        if dec.key is None:
            print(f"[WARN] {ep_name}: unrecognized notation {dec.raw!r} -> skip")
            unrecognized += 1
            continue

        if dec.key == "k":
            print(f"[SKIP] {ep_name}: notation {dec.raw!r} -> ignore ('k')")
            ignored += 1
            continue

        if not src.exists():
            print(f"[WARN] {ep_name} listed in CSV but not found in base folder.")
            missing += 1
            continue

        task_folder_name = TASK_FOLDERS[dec.key]
        assert task_folder_name is not None

        dst_dir = out_root / task_folder_name
        dst = dst_dir / ep_name
        touched_folders.add(dst_dir)

        print(f"[COPY] {src} -> {dst_dir} (notation={dec.raw!r})")

        if not args.dry_run:
            safe_mkdir(dst_dir)
            copy_episode(src, dst, overwrite=args.overwrite)

        copied += 1

    # Renumber copied episodes inside each target folder (optional)
    if not args.no_renumber:
        for d in sorted(touched_folders):
            print(f"[RENUMBER] {d}")
            if not args.dry_run:
                renumber_episodes(d)

    print("\n[SUMMARY]")
    print(f"  source base (unchanged): {base}")
    print(f"  output root:             {out_root}")
    print(f"  copied:                  {copied}")
    print(f"  ignored ('k'):           {ignored}")
    print(f"  missing episodes:        {missing}")
    print(f"  unrecognized notation:   {unrecognized}")
    print(f"  overwrite:               {args.overwrite}")
    print(f"  dry-run:                 {args.dry_run}")
    print(f"  renumber targets:        {not args.no_renumber}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
