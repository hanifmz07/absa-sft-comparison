#!/bin/bash

source .venv/bin/activate

OUTPUT_DIR="$1"
DATASET_TYPE="$2"
LANG="$3"
DATASET_FOLDER="$4"

if [ -z "$OUTPUT_DIR" ]; then
    echo "Error: output_dir is required."
    echo "Usage: bash scripts/eval_instruct_absa.sh <output_dir> <dataset_type> <lang> <dataset_folder>"
    exit 1
fi

if [ -z "$DATASET_TYPE" ]; then
    echo "Error: dataset_type is required."
    echo "Usage: bash scripts/eval_instruct_absa.sh <output_dir> <dataset_type> <lang> <dataset_folder>"
    exit 1
fi

if [ -z "$LANG" ]; then
    echo "Error: lang is required."
    echo "Usage: bash scripts/eval_instruct_absa.sh <output_dir> <dataset_type> <lang> <dataset_folder>"
    exit 1
fi

if [ -z "$DATASET_FOLDER" ]; then
    echo "Error: dataset_folder is required."
    echo "Usage: bash scripts/eval_instruct_absa.sh <output_dir> <dataset_type> <lang> <dataset_folder>"
    exit 1
fi

SEARCH_ROOT="$OUTPUT_DIR/evals/$DATASET_TYPE/$LANG/$DATASET_FOLDER"

if [ ! -d "$SEARCH_ROOT" ]; then
    echo "Error: directory not found: $SEARCH_ROOT"
    exit 1
fi

if [ "$DATASET_FOLDER" = "mvp" ]; then
    mapfile -t INFERENCE_PATHS < <(find "$SEARCH_ROOT" -type f \( -name "inference_results.json" -o -name "voting_results.json" \) | sort)
else
    mapfile -t INFERENCE_PATHS < <(find "$SEARCH_ROOT" -type f -name "inference_results.json" | sort)
fi

if [ ${#INFERENCE_PATHS[@]} -eq 0 ]; then
    echo "No result files found under: $SEARCH_ROOT"
    exit 0
fi

echo "Found ${#INFERENCE_PATHS[@]} result file(s) under: $SEARCH_ROOT"

for INFERENCE_PATH in "${INFERENCE_PATHS[@]}"; do
    echo "Processing: $INFERENCE_PATH"
    python -m src.main.eval_instruct_absa \
        --inference_path "$INFERENCE_PATH"
done
