#!/bin/bash

OUTPUT_DIR="$1"

if [ -z "$OUTPUT_DIR" ]; then
    echo "Error: output_dir is required."
    echo "Usage: bash scripts/eval_all_exact_match.sh <output_dir>"
    exit 1
fi

LANGS=(eng indo jav mad min sun)
DATASET_TYPE="hotel_reviews"
DATASET_FOLDERS=(mvp_aos mvp)

for LANG in "${LANGS[@]}"; do
    for DATASET_FOLDER in "${DATASET_FOLDERS[@]}"; do
        echo "========================================================"
        echo "Running exact-match eval for: dataset_type=$DATASET_TYPE lang=$LANG dataset_folder=$DATASET_FOLDER"
        echo "========================================================"

        bash scripts/eval_exact_match.sh \
            "$OUTPUT_DIR" \
            "$DATASET_TYPE" \
            "$LANG" \
            "$DATASET_FOLDER"
    done
done
