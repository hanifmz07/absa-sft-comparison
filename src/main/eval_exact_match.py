import argparse
import glob
import json
import os
from typing import Any, Dict, List

from ..utils.eval_utils import calculate_metrics


def resolve_inference_paths(inference_path: str) -> List[str]:
    if os.path.isfile(inference_path):
        return [inference_path]

    matched_paths = sorted(glob.glob(inference_path, recursive=True))
    return [path for path in matched_paths if os.path.isfile(path)]


def extract_metadata(path: str) -> Dict[str, Any]:
    normalized_path = path.replace("\\", "/")
    parts = [part for part in normalized_path.split("/") if part]
    metadata: Dict[str, Any] = {
        "path": path,
        "lang": None,
        "seed": None,
        "dataset_type": None,
        "dataset_folder": None,
        "use_constrained_decoding": None,
    }

    # Resolve both relative and absolute paths for:
    # outputs/evals/{dataset_type}/{lang}/{dataset_folder}/seed_{seed}/...
    # outputs_seq2seq/evals/{dataset_type}/{lang}/{dataset_folder}/seed_{seed}/...
    eval_root_idx = None
    for idx in range(1, len(parts)):
        if parts[idx] == "evals" and parts[idx - 1] in {"outputs", "outputs_seq2seq"}:
            eval_root_idx = idx
            break

    if eval_root_idx is not None and len(parts) > eval_root_idx + 3:
        metadata["dataset_type"] = parts[eval_root_idx + 1]
        metadata["lang"] = parts[eval_root_idx + 2]
        metadata["dataset_folder"] = parts[eval_root_idx + 3]

    seed_part = next((part for part in parts if part.startswith("seed_")), None)
    if seed_part and len(seed_part) > len("seed_"):
        metadata["seed"] = seed_part[len("seed_"):]

    if len(parts) >= 2:
        metadata["use_constrained_decoding"] = parts[-2] == "constrained_decoding"

    return metadata


def evaluate_file(inference_path: str) -> Dict[str, Any]:
    with open(inference_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    target_lists = [item["target_list"] for item in data]
    prediction_lists = [item["prediction_list"] for item in data]
    scores = calculate_metrics(prediction_lists, target_lists, task="exact_match")

    result = extract_metadata(inference_path)
    result.update(
        {
            "precision": scores["precision_exact_match"] * 100,
            "recall": scores["recall_exact_match"] * 100,
            "f1": scores["f1_exact_match"] * 100,
        }
    )
    return result


def evaluate_file_details(inference_path: str) -> List[Dict[str, Any]]:
    with open(inference_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    details: List[Dict[str, Any]] = []
    for item in data:
        target_list = item.get("target_list", [])
        prediction_list = item.get("prediction_list", [])
        score = calculate_metrics([prediction_list], [target_list], task="exact_match")
        false_positive = [pred for pred in prediction_list if pred not in target_list]
        false_negative = [target for target in target_list if target not in prediction_list]

        details.append(
            {
                "sentence_id": item.get("sentence_id"),
                "task_elements": item.get("task_elements"),
                "element_order": item.get("element_order"),
                "target_list": target_list,
                "prediction_list": prediction_list,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "precision": score["precision_exact_match"] * 100,
                "recall": score["recall_exact_match"] * 100,
                "f1": score["f1_exact_match"] * 100,
            }
        )

    return details


def main(args: argparse.Namespace) -> None:
    inference_paths = resolve_inference_paths(args.inference_path)
    if not inference_paths:
        raise FileNotFoundError(f"No inference files found from: {args.inference_path}")

    results = []
    for inference_path in inference_paths:
        print(f"Processing: {inference_path}")
        summary_result = evaluate_file(inference_path)
        detail_result = evaluate_file_details(inference_path)
        results.append(summary_result)

        output_dir = os.path.dirname(inference_path)
        input_filename = os.path.basename(inference_path)
        if input_filename == "voting_results.json":
            summary_filename = "voting_exact_match.json"
            detail_filename = "voting_exact_match_detail.json"
        else:
            summary_filename = "exact_match.json"
            detail_filename = "exact_match_detail.json"

        summary_path = os.path.join(output_dir, summary_filename)
        detail_path = os.path.join(output_dir, detail_filename)

        with open(summary_path, "w", encoding="utf-8") as file:
            json.dump(summary_result, file, indent=2, ensure_ascii=False)
        with open(detail_path, "w", encoding="utf-8") as file:
            json.dump(detail_result, file, indent=2, ensure_ascii=False)

        print(f"Saved summary metrics to: {summary_path}")
        print(f"Saved detail metrics to: {detail_path}")

    if len(results) == 1:
        print("Exact match metrics:")
        print(json.dumps(results[0], indent=2, ensure_ascii=False))
    else:
        print("Exact match metrics summary:")
        print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate exact-match ABSA metrics from inference results")
    parser.add_argument(
        "--inference_path",
        type=str,
        required=True,
        help="Path or glob pattern to inference_results.json or voting_results.json",
    )

    parsed_args = parser.parse_args()
    main(parsed_args)
