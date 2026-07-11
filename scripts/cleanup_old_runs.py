"""
Remove duplicate training run directories, keeping only the latest per
(lang, dataset_folder, seed, model_basename) group.

Usage:
    python scripts/cleanup_old_runs.py --base outputs/evals/hotel_reviews --dry-run
    python scripts/cleanup_old_runs.py --base outputs/evals/hotel_reviews --delete
"""

import argparse
import re
import shutil
from collections import defaultdict
from pathlib import Path


def extract_model_key(dirname: str) -> str | None:
    m = re.search(r"train_model-(.*?)_lr-", dirname)
    return m.group(1) if m else None


def collect_deletions(base: Path) -> list[Path]:
    to_delete: list[Path] = []
    for lang_dir in sorted(base.iterdir()):
        if not lang_dir.is_dir():
            continue
        for dataset_dir in sorted(lang_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            for seed_dir in sorted(dataset_dir.iterdir()):
                if not seed_dir.is_dir():
                    continue
                by_model: dict[str, list[Path]] = defaultdict(list)
                for run_dir in seed_dir.iterdir():
                    if not run_dir.is_dir():
                        continue
                    key = extract_model_key(run_dir.name)
                    if key:
                        by_model[key].append(run_dir)
                for runs in by_model.values():
                    runs.sort(key=lambda p: p.name[:15])
                    to_delete.extend(runs[:-1])
    return sorted(to_delete)


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove old duplicate run directories.")
    parser.add_argument("--base", required=True, help="Base eval directory (e.g. outputs/evals/hotel_reviews)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", dest="dry_run", action="store_true", default=True, help="Print what would be deleted (default)")
    mode.add_argument("--delete", dest="dry_run", action="store_false", help="Actually delete directories")
    args = parser.parse_args()

    base = Path(args.base)
    if not base.is_dir():
        raise SystemExit(f"Directory not found: {base}")

    to_delete = collect_deletions(base)

    if not to_delete:
        print("Nothing to delete — all groups already have exactly one run.")
        return

    label = "[DRY RUN] Would delete" if args.dry_run else "Deleting"
    print(f"{label} {len(to_delete)} run director{'y' if len(to_delete) == 1 else 'ies'}:\n")
    for p in to_delete:
        print(f"  {p}")

    if not args.dry_run:
        for p in to_delete:
            shutil.rmtree(p)
        print(f"\nDone. Deleted {len(to_delete)} director{'y' if len(to_delete) == 1 else 'ies'}.")
    else:
        print(f"\nRe-run with --delete to remove these directories.")


if __name__ == "__main__":
    main()
