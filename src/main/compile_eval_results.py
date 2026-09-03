import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# outputs/evals/{dataset_type}/{lang}/{dataset_folder}/seed_{seed}/
#   {timestamp}_train_model-{model}_lr-{lr}_bs-{bs}_epochs-{epochs}/
#   checkpoint-{checkpoint}/{decoding_strategy}/{filename}
PATH_PATTERN = re.compile(
    r"(?:outputs|outputs_seq2seq)/evals/"
    r"(?P<dataset_type>[^/]+)/"
    r"(?P<lang>[^/]+)/"
    r"(?P<dataset_folder>[^/]+)/"
    r"seed_(?P<seed>[^/]+)/"
    r"(?P<timestamp>\d+_\d+)_train_model-(?P<model_name>.+?)"
    r"_lr-(?P<lr>[^_/]+)_bs-(?P<bs>[^_/]+)_epochs-(?P<epochs>[^_/]+)/"
    r"checkpoint-(?P<checkpoint>[^/]+)/"
    r"(?P<decoding_strategy>[^/]+)/"
)

EVAL_TYPES: Dict[str, str] = {
    "exact_match.json": "exact_match_results.csv",
    "instruct_absa.json": "instruct_absa_results.csv",
    "semantic_metrics.json": "semantic_results.csv",
}


def parse_run_metadata(path: Path) -> Dict[str, Any]:
    normalized_path = path.as_posix()
    match = PATH_PATTERN.search(normalized_path)

    metadata: Dict[str, Any] = {
        "dataset_type": None,
        "lang": None,
        "dataset_folder": None,
        "seed": None,
        "timestamp": None,
        "model_name": None,
        "lr": None,
        "bs": None,
        "epochs": None,
        "checkpoint": None,
        "decoding_strategy": None,
    }

    if match:
        metadata.update(match.groupdict())

    return metadata


def load_summary(path: Path) -> Optional[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    row = parse_run_metadata(path)
    row.update(
        {
            "precision": data.get("precision"),
            "recall": data.get("recall"),
            "f1": data.get("f1"),
            "use_constrained_decoding": data.get("use_constrained_decoding"),
            "source_path": normalize_relpath(path),
        }
    )
    return row


def normalize_relpath(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


SORT_KEYS = ["dataset_type", "lang", "dataset_folder", "seed", "checkpoint", "decoding_strategy"]

COLUMN_ORDER = [
    "dataset_type",
    "lang",
    "dataset_folder",
    "seed",
    "model_name",
    "lr",
    "bs",
    "epochs",
    "checkpoint",
    "decoding_strategy",
    "use_constrained_decoding",
    "precision",
    "recall",
    "f1",
    "timestamp",
    "source_path",
]


def compile_eval_type(evals_root: Path, filename: str) -> List[Dict[str, Any]]:
    rows = [load_summary(path) for path in sorted(evals_root.rglob(filename))]
    return rows


def main(args: argparse.Namespace) -> None:
    evals_root = Path(args.evals_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not evals_root.is_dir():
        raise FileNotFoundError(f"Evals root not found: {evals_root}")

    for filename, csv_name in EVAL_TYPES.items():
        rows = compile_eval_type(evals_root, filename)
        if not rows:
            print(f"No '{filename}' files found under {evals_root}; skipping {csv_name}")
            continue

        df = pd.DataFrame(rows)
        df = df.reindex(columns=COLUMN_ORDER)
        df = df.sort_values(by=SORT_KEYS, na_position="last").reset_index(drop=True)

        csv_path = output_dir / csv_name
        df.to_csv(csv_path, index=False)
        print(f"Wrote {len(df)} row(s) to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compile exact-match, instruct-ABSA, and semantic eval results under outputs/evals into CSVs"
    )
    parser.add_argument(
        "--evals_root",
        type=str,
        default="outputs/evals",
        help="Root directory to search for eval result JSON files (default: outputs/evals)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/evals",
        help="Directory to write the compiled CSV files to (default: outputs/evals)",
    )

    parsed_args = parser.parse_args()
    main(parsed_args)
