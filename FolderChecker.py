#!/usr/bin/env python3
"""
List all empty folders recursively under the directory where this script lives.

Empty = contains no files and no subfolders.
(So folders that only contain other empty folders are NOT considered empty.)
"""

from __future__ import annotations
from pathlib import Path
import sys


def find_empty_dirs(root: Path) -> list[Path]:
    empty_dirs: list[Path] = []

    # Walk bottom-up so we can evaluate children before parents.
    for p in sorted(root.rglob("*"), key=lambda x: len(x.parts), reverse=True):
        if not p.is_dir():
            continue
        try:
            # Empty means: no entries at all
            if not any(p.iterdir()):
                empty_dirs.append(p)
        except (PermissionError, FileNotFoundError):
            # Skip folders we can't read or that disappeared mid-scan
            continue

    return sorted(empty_dirs)


def main() -> int:
    script_dir = Path(__file__).resolve().parent

    empty_dirs = find_empty_dirs(script_dir)

    if not empty_dirs:
        print(f"No empty folders found under: {script_dir}")
        return 0

    print(f"Empty folders under: {script_dir}\n")
    for d in empty_dirs:
        # Print relative paths for readability
        try:
            print(d.relative_to(script_dir))
        except ValueError:
            print(d)

    print(f"\nTotal empty folders: {len(empty_dirs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
