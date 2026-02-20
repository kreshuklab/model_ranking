#!/usr/bin/env python3
"""Copy listed files (with .h5 suffix) from a source directory to a destination directory.

Each line in the split file is a basename without the .h5 extension.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy files listed in a split file from a source directory to a destination directory. "
            "Lines are basenames without the .h5 extension."
        )
    )
    _ = parser.add_argument(
        "--split-file",
        type=Path,
        required=True,
        help="Path to finetuning_train_split1.txt (one basename per line, no .h5).",
    )
    _ = parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Directory containing the .h5 files referenced by the split file.",
    )
    _ = parser.add_argument(
        "--dest-dir",
        type=Path,
        required=True,
        help="Destination directory to copy the files into.",
    )
    _ = parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be copied without copying.",
    )
    return parser.parse_args()


def _read_names(split_file: Path) -> list[str]:
    names: list[str] = []
    for line in split_file.read_text().splitlines():
        name = line.strip()
        if not name:
            continue
        names.append(name)
    return names


def main() -> int:
    args = _parse_args()

    if not args.split_file.is_file():
        raise SystemExit(f"Split file not found: {args.split_file}")
    if not args.source_dir.is_dir():
        raise SystemExit(f"Source directory not found: {args.source_dir}")

    args.dest_dir.mkdir(parents=True, exist_ok=True)

    names = _read_names(args.split_file)
    missing: list[Path] = []

    for name in names:
        src = args.source_dir / f"{name}.h5"
        dst = args.dest_dir / f"{name}.h5"
        if not src.is_file():
            missing.append(src)
            continue
        if args.dry_run:
            print(f"DRY RUN: {src} -> {dst}")
        else:
            shutil.copy2(src, dst)

    if missing:
        missing_str = "\n".join(str(p) for p in missing)
        raise SystemExit(f"Missing {len(missing)} file(s):\n{missing_str}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
