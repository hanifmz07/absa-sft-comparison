"""
Extract evaluation JSON files from outputs_seq2seq/evals/hotel_reviews/<lang> into combined CSV files.

All languages are merged into a single CSV per metric, written to outputs_seq2seq/evals/hotel_reviews/.
Languages are auto-discovered from subdirectories; use --lang to restrict to one.

Extracts instruct_absa.json, exact_match.json, and semantic_metrics.json from outputs_seq2seq/.../<lang>/mvp_aos/.

Decoding filter (--decoding):
  all           -- include both constrained and unconstrained rows (default)
  constrained   -- keep only rows where use_constrained_decoding is True
  unconstrained -- keep only rows where use_constrained_decoding is False

Usage:
  python scripts/extract_seq2seq_results.py
  python scripts/extract_seq2seq_results.py --lang indo
  python scripts/extract_seq2seq_results.py --decoding unconstrained
  python scripts/extract_seq2seq_results.py --lang eng --decoding constrained
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
EVALS_BASE = BASE_DIR / "outputs_seq2seq" / "evals" / "hotel_reviews"

TARGETS = {
    "instruct_absa": "instruct_absa.csv",
    "exact_match": "exact_match.csv",
    "semantic_metrics": "semantic_metrics.csv",
}

FIELDNAMES = [
    "lang",
    "seed",
    "model",
    "checkpoint",
    "dataset_type",
    "dataset_folder",
    "use_constrained_decoding",
    "precision",
    "recall",
    "f1",
    "inference_time",
    "path",
]

import re

def filter_latest_models(rows: list[dict]) -> list[dict]:
    grouped = {}
    for row in rows:
        model_folder = row.get("model", "")
        match = re.match(r"^(\d{8}_\d{6})_(.+)$", model_folder)
        if match:
            timestamp = match.group(1)
            rest = match.group(2)
            m2 = re.search(r"train_model-(.*?)_lr-", rest)
            base_model = m2.group(1) if m2 else rest
        else:
            timestamp = ""
            base_model = model_folder
            
        key = (
            row.get("lang", ""),
            row.get("dataset_folder", ""),
            row.get("seed", ""),
            base_model,
            row.get("use_constrained_decoding", "")
        )
        
        if key not in grouped:
            grouped[key] = (timestamp, row)
        else:
            if timestamp > grouped[key][0]:
                grouped[key] = (timestamp, row)
                
    return [item[1] for item in grouped.values()]


def collect(search_dir: Path, lang: str, filename: str, decoding: str) -> list[dict]:
    rows = []
    for json_file in sorted(search_dir.rglob(f"{filename}.json")):
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)
        if decoding == "constrained" and not data.get("use_constrained_decoding"):
            continue
        if decoding == "unconstrained" and data.get("use_constrained_decoding"):
            continue
        row = {field: data.get(field, "") for field in FIELDNAMES if field not in ["inference_time", "model", "checkpoint"]}
        row["lang"] = lang
        rel = json_file.relative_to(search_dir)
        # rel.parts: (seed_XXX, <model_folder>, <checkpoint>, <decoding>, filename)
        row["model"] = rel.parts[1] if len(rel.parts) > 1 else ""
        row["checkpoint"] = rel.parts[2] if len(rel.parts) > 2 else ""

        inference_time = ""
        eval_res_path = json_file.with_name("evaluation_results.json")
        if eval_res_path.exists():
            with open(eval_res_path, encoding="utf-8") as ef:
                try:
                    eval_data = json.load(ef)
                    if "inference_duration" in eval_data:
                        inference_time = eval_data["inference_duration"]
                except json.JSONDecodeError:
                    pass

        if inference_time == "":
            inf_res_path = json_file.with_name("inference_results.json")
            if inf_res_path.exists():
                with open(inf_res_path, encoding="utf-8") as inf_f:
                    try:
                        inf_data = json.load(inf_f)
                        if len(inf_data) > 0 and "inference_time" in inf_data[0]:
                            inference_time = inf_data[0]["inference_time"]
                    except json.JSONDecodeError:
                        pass

        row["inference_time"] = inference_time
        rows.append(row)
    return rows


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Written {len(rows)} rows -> {out_path}")


def discover_langs() -> list[str]:
    return sorted(p.name for p in EVALS_BASE.iterdir() if p.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract seq2seq eval JSON results to CSV.")
    parser.add_argument(
        "--lang",
        default=None,
        help="Language subdirectory to process (default: all discovered languages).",
    )
    parser.add_argument(
        "--decoding",
        choices=["all", "constrained", "unconstrained"],
        default="all",
        help="Filter rows by decoding type (default: all).",
    )
    args = parser.parse_args()

    langs = [args.lang] if args.lang else discover_langs()

    combined: dict[str, list[dict]] = defaultdict(list)
    for lang in langs:
        search_dir = EVALS_BASE / lang / "mvp_aos"
        if not search_dir.exists():
            print(f"[skip] {search_dir} does not exist")
            continue
        print(f"[{lang}] Searching in: {search_dir}")
        for filename, csv_name in TARGETS.items():
            rows = collect(search_dir, lang, filename, args.decoding)
            if not rows:
                print(f"  No files found for '{filename}'")
                continue
            combined[csv_name].extend(rows)

    for csv_name, rows in combined.items():
        filtered_rows = filter_latest_models(rows)
        write_csv(filtered_rows, EVALS_BASE / csv_name)


if __name__ == "__main__":
    main()
