"""
Interactively delete trained-model checkpoint directories to free disk space.
Requires manual selection — no automatic dedup logic. This complements
cleanup_old_runs.py, which handles eval result directories.

Usage:
    python scripts/cleanup_models_interactive.py --base outputs/models/hotel_reviews
    python scripts/cleanup_models_interactive.py --base outputs_seq2seq/models/hotel_reviews --sort size
"""

import argparse
import os
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class RunInfo:
    path: Path
    size_bytes: int
    timestamp: datetime | None
    model_key: str | None


def parse_timestamp(dirname: str) -> datetime | None:
    """Parse YYYYMMDD_HHMMSS timestamp from first 15 chars of dirname."""
    try:
        return datetime.strptime(dirname[:15], "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def extract_model_key(dirname: str) -> str | None:
    """Extract model name from dirname using train_model-(...)_lr- pattern."""
    m = re.search(r"train_model-(.*?)_lr-", dirname)
    return m.group(1) if m else None


def dir_size(path: Path) -> int:
    """Recursively compute directory size in bytes, skipping symlinks."""
    total = 0
    try:
        for entry in os.scandir(path):
            try:
                if entry.is_symlink():
                    continue
                if entry.is_dir(follow_symlinks=False):
                    total += dir_size(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    total += entry.stat(follow_symlinks=False).st_size
            except OSError:
                continue
    except OSError:
        pass
    return total


def human_size(num_bytes: float) -> str:
    """Format bytes as human-readable string (e.g., '2.8G')."""
    for unit in ["B", "K", "M", "G", "T"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:.1f}{unit}".rstrip("0").rstrip(".")
        num_bytes /= 1024.0
    return f"{num_bytes:.1f}P".rstrip("0").rstrip(".")


def collect_run_dirs(base: Path) -> list[Path]:
    """Collect all run directories under base, following lang/dataset/seed/run_dir hierarchy."""
    to_collect: list[Path] = []
    for lang_dir in sorted(base.iterdir()):
        if not lang_dir.is_dir():
            continue
        for dataset_dir in sorted(lang_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            for seed_dir in sorted(dataset_dir.iterdir()):
                if not seed_dir.is_dir():
                    continue
                for run_dir in sorted(seed_dir.iterdir()):
                    if not run_dir.is_dir():
                        continue
                    to_collect.append(run_dir)
    return to_collect


def build_run_infos(run_dirs: list[Path]) -> list[RunInfo]:
    """Build RunInfo for each run directory, with size and metadata."""
    print(f"Scanning {len(run_dirs)} run director{'y' if len(run_dirs) == 1 else 'ies'}...")
    infos = []
    for run_dir in run_dirs:
        size_bytes = dir_size(run_dir)
        timestamp = parse_timestamp(run_dir.name)
        model_key = extract_model_key(run_dir.name)
        infos.append(RunInfo(path=run_dir, size_bytes=size_bytes, timestamp=timestamp, model_key=model_key))
    return infos


def print_table(runs: list[RunInfo], sort_by: str) -> list[RunInfo]:
    """Print numbered table of run directories; return sorted list for selection indexing."""
    if sort_by == "size":
        sorted_runs = sorted(runs, key=lambda r: r.size_bytes, reverse=True)
    elif sort_by == "time":
        sorted_runs = sorted(runs, key=lambda r: r.timestamp or datetime.min)
    else:  # default: path
        sorted_runs = sorted(runs, key=lambda r: r.path)

    print("\n" + "─" * 180)
    print(f"{'#':<4} {'Size':<8} {'Timestamp':<20} {'Model':<30} {'Path':<100}")
    print("─" * 180)

    for idx, run in enumerate(sorted_runs, 1):
        size_str = human_size(run.size_bytes)
        ts_str = run.timestamp.strftime("%Y-%m-%d %H:%M") if run.timestamp else "unknown"
        model_str = run.model_key or "unknown"
        path_str = str(run.path)
        print(f"{idx:<4} {size_str:<8} {ts_str:<20} {model_str:<30} {path_str:<100}")

    print("─" * 180 + "\n")
    return sorted_runs


def parse_selection(raw: str, n: int) -> set[int] | None:
    """
    Parse user input for selection.
    Returns None for cancel, set[int] for selected indices (1-based).
    Raises ValueError for invalid input.
    """
    raw = raw.strip().lower()

    # Cancel cases
    if raw in ("", "q", "quit", "none", "cancel"):
        return None

    # All case
    if raw == "all":
        return set(range(1, n + 1))

    # Parse comma-separated indices and ranges
    selected = set()
    tokens = [t.strip() for t in raw.split(",")]

    for token in tokens:
        if "-" in token:
            # Range: "2-4"
            try:
                start_str, end_str = token.split("-", 1)
                start = int(start_str.strip())
                end = int(end_str.strip())
                if start < 1 or end < 1:
                    raise ValueError(f"Index must be >= 1, got {token}")
                if start > end:
                    raise ValueError(f"Invalid range {token} (start > end)")
                if end > n:
                    raise ValueError(f"Index out of range: {token} (max is {n})")
                selected.update(range(start, end + 1))
            except ValueError as e:
                # Re-raise with context if not already a bounds/logic error
                if "Index" not in str(e) and "Invalid range" not in str(e):
                    raise ValueError(f"Invalid range {token}: must be integers")
                raise
        else:
            # Single index
            try:
                idx = int(token)
                if idx < 1:
                    raise ValueError(f"Index must be >= 1, got {idx}")
                if idx > n:
                    raise ValueError(f"Index out of range: {idx} (max is {n})")
                selected.add(idx)
            except ValueError as e:
                if "out of range" not in str(e) and "must be >= 1" not in str(e):
                    raise ValueError(f"Invalid index {token}: must be an integer")
                raise

    return selected


def prompt_selection(n: int) -> set[int] | None:
    """Loop until user provides valid selection or cancels."""
    while True:
        try:
            raw = input("Select indices to delete (e.g. 1,3,5-7), 'all', or blank/'q' to cancel: ")
            return parse_selection(raw, n)
        except ValueError as e:
            print(f"  Error: {e}")


def confirm_deletion(selected: list[RunInfo]) -> bool:
    """Print summary of selected items and require typing DELETE to confirm."""
    print("\n" + "=" * 100)
    print("SELECTED FOR DELETION:")
    print("=" * 100)

    total_size = 0
    for run in selected:
        size_str = human_size(run.size_bytes)
        print(f"  {size_str:<8}  {run.path}")
        total_size += run.size_bytes

    print("=" * 100)
    print(f"Total space to free: {human_size(total_size)}")
    print("=" * 100 + "\n")

    confirmation = input("Type DELETE to confirm deletion (or anything else to abort): ")
    return confirmation == "DELETE"


def delete_dirs(paths: list[Path]) -> tuple[list[Path], list[tuple[Path, OSError]]]:
    """Delete directories; return (successes, failures)."""
    succeeded = []
    failed = []

    for path in paths:
        try:
            shutil.rmtree(path)
            succeeded.append(path)
        except OSError as e:
            failed.append((path, e))

    return succeeded, failed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactively select and delete trained-model checkpoint directories."
    )
    parser.add_argument("--base", required=True, help="Base model directory (e.g., outputs/models/hotel_reviews)")
    parser.add_argument(
        "--sort",
        choices=["path", "size", "time"],
        default="path",
        help="Sort table by: path (default, groups by lang/dataset/seed), size (largest first), or time (oldest first)",
    )
    args = parser.parse_args()

    base = Path(args.base)
    if not base.is_dir():
        raise SystemExit(f"Directory not found: {base}")

    run_dirs = collect_run_dirs(base)
    if not run_dirs:
        print(f"No run directories found under {base}")
        return

    infos = build_run_infos(run_dirs)
    sorted_infos = print_table(infos, args.sort)

    selection = prompt_selection(len(sorted_infos))
    if selection is None:
        print("Cancelled. Nothing deleted.")
        return

    selected = [sorted_infos[i - 1] for i in sorted(selection)]

    if not confirm_deletion(selected):
        print("Aborted. Nothing deleted.")
        return

    print("Deleting...")
    succeeded, failed = delete_dirs([r.path for r in selected])

    total_freed = sum(r.size_bytes for r in selected if r.path in succeeded)
    print(f"\nSuccessfully deleted {len(succeeded)} director{'y' if len(succeeded) == 1 else 'ies'}.")
    print(f"Space freed: {human_size(total_freed)}")

    if failed:
        print(f"\nFailed to delete {len(failed)} director{'y' if len(failed) == 1 else 'ies'}:")
        for path, err in failed:
            print(f"  {path}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
