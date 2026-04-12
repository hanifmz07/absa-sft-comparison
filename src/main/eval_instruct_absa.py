import argparse
import glob
import json
import os
from typing import Any, Dict, List

from ..utils.eval_utils import parse_absa_string


def resolve_inference_paths(inference_path: str) -> List[str]:
    if os.path.isfile(inference_path):
        return [inference_path]

    matched_paths = sorted(glob.glob(inference_path, recursive=True))
    return [path for path in matched_paths if os.path.isfile(path)]


def extract_metadata(path: str) -> Dict[str, Any]:
    parts = path.split("/")
    metadata: Dict[str, Any] = {
        "path": path,
        "lang": None,
        "seed": None,
        "dataset_type": None,
        "dataset_folder": None,
        "use_constrained_decoding": None,
    }

    # Expected pattern:
    # outputs/evals/{dataset_type}/{lang}/{dataset_folder}/seed_{seed}/.../{constrained_decoding|unconstrained_decoding}/{inference_results|voting_results}.json
    try:
        metadata["dataset_type"] = parts[2]
        metadata["lang"] = parts[3]
        metadata["dataset_folder"] = parts[4]
        metadata["seed"] = parts[5].split("_")[1] if parts[5].startswith("seed_") else None
        metadata["use_constrained_decoding"] = parts[-2] == "constrained_decoding"
    except IndexError:
        pass

    return metadata


def form_aoste_triplet(triplets: List[Dict[str, str]]) -> str:
    aoste_triplets: List[str] = []
    for triplet in triplets:
        aspect = triplet.get("A", "")
        opinion = triplet.get("O", "")
        sentiment = triplet.get("S", "")
        aoste_triplets.append(f"{aspect}:{opinion}:{sentiment}")
    return ", ".join(aoste_triplets)


def to_aoste_text(text: str) -> str:
    return form_aoste_triplet(parse_absa_string(text))


def split_aoste_triplets(aoste_text: str) -> List[str]:
    return [part.strip() for part in aoste_text.split(",") if part.strip()]


def instruct_absa_fp_fn(target_aoste: str, prediction_aoste: str) -> Dict[str, List[str]]:
    target_triplets = split_aoste_triplets(target_aoste)
    prediction_triplets = split_aoste_triplets(prediction_aoste)

    false_negative = [
        gt
        for gt in target_triplets
        if not any(pred in gt or gt in pred for pred in prediction_triplets)
    ]
    false_positive = [
        pred
        for pred in prediction_triplets
        if not any(pred in gt or gt in pred for gt in target_triplets)
    ]
    return {
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def safe_metrics_instructabsa(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    total_pred = 0
    total_gt = 0
    tp = 0

    for gt, pred in zip(y_true, y_pred):
        gt_list = [x.strip() for x in gt.split(",") if x.strip()]
        pred_list = [x.strip() for x in pred.split(",") if x.strip()]

        total_pred += len(pred_list)
        total_gt += len(gt_list)

        for gt_val in gt_list:
            for pred_val in pred_list:
                if pred_val in gt_val or gt_val in pred_val:
                    tp += 1
                    break

    precision = tp / total_pred if total_pred > 0 else 0.0
    recall = tp / total_gt if total_gt > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_file(inference_path: str) -> Dict[str, Any]:
    with open(inference_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    target_aoste = [to_aoste_text(item.get("target", "")) for item in data]
    prediction_aoste = [to_aoste_text(item.get("prediction", "")) for item in data]

    scores = safe_metrics_instructabsa(target_aoste, prediction_aoste)

    result = extract_metadata(inference_path)
    result.update(
        {
            "precision": scores["precision"] * 100,
            "recall": scores["recall"] * 100,
            "f1": scores["f1"] * 100,
        }
    )
    return result


def evaluate_file_details(inference_path: str) -> List[Dict[str, Any]]:
    with open(inference_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    details: List[Dict[str, Any]] = []
    for item in data:
        target_text = item.get("target", "")
        prediction_text = item.get("prediction", "")
        target_aoste = to_aoste_text(target_text)
        prediction_aoste = to_aoste_text(prediction_text)
        scores = safe_metrics_instructabsa([target_aoste], [prediction_aoste])
        mismatch = instruct_absa_fp_fn(target_aoste, prediction_aoste)

        details.append(
            {
                "sentence_id": item.get("sentence_id"),
                "task_elements": item.get("task_elements"),
                "element_order": item.get("element_order"),
                "target": target_text,
                "prediction": prediction_text,
                "target_aoste": target_aoste,
                "prediction_aoste": prediction_aoste,
                "false_positive": mismatch["false_positive"],
                "false_negative": mismatch["false_negative"],
                "precision": scores["precision"] * 100,
                "recall": scores["recall"] * 100,
                "f1": scores["f1"] * 100,
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
            summary_filename = "voting_instruct_absa.json"
            detail_filename = "voting_instruct_absa_detail.json"
        else:
            summary_filename = "instruct_absa.json"
            detail_filename = "instruct_absa_detail.json"

        summary_path = os.path.join(output_dir, summary_filename)
        detail_path = os.path.join(output_dir, detail_filename)

        with open(summary_path, "w", encoding="utf-8") as file:
            json.dump(summary_result, file, indent=2, ensure_ascii=False)
        with open(detail_path, "w", encoding="utf-8") as file:
            json.dump(detail_result, file, indent=2, ensure_ascii=False)

        print(f"Saved summary metrics to: {summary_path}")
        print(f"Saved detail metrics to: {detail_path}")

    if len(results) == 1:
        print("InstructABSA metrics:")
        print(json.dumps(results[0], indent=2, ensure_ascii=False))
    else:
        print("InstructABSA metrics summary:")
        print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate InstructABSA metrics from inference results")
    parser.add_argument(
        "--inference_path",
        type=str,
        required=True,
        help="Path or glob pattern to inference_results.json or voting_results.json",
    )

    parsed_args = parser.parse_args()
    main(parsed_args)
