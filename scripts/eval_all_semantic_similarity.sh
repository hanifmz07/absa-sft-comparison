#!/bin/bash

source .venv/bin/activate

OUTPUT_DIR="$1"
EMBEDDING_MODEL_NAME="$2"

if [ -z "$OUTPUT_DIR" ]; then
    echo "Error: output_dir is required."
    echo "Usage: bash scripts/eval_all_semantic_similarity.sh <output_dir> <embedding_model_name>"
    exit 1
fi

if [ -z "$EMBEDDING_MODEL_NAME" ]; then
    echo "Error: embedding_model_name is required."
    echo "Usage: bash scripts/eval_all_semantic_similarity.sh <output_dir> <embedding_model_name>"
    exit 1
fi

LANGS=(eng indo jav mad min sun)
DATASET_TYPE="hotel_reviews"
DATASET_FOLDERS=(mvp_aos mvp)

for LANG in "${LANGS[@]}"; do
    for DATASET_FOLDER in "${DATASET_FOLDERS[@]}"; do
        echo "========================================================"
        echo "Running semantic eval for: dataset_type=$DATASET_TYPE lang=$LANG dataset_folder=$DATASET_FOLDER"
        echo "========================================================"

        bash scripts/eval_semantic_similarity.sh \
            "$OUTPUT_DIR" \
            "$DATASET_TYPE" \
            "$LANG" \
            "$DATASET_FOLDER" \
            "$EMBEDDING_MODEL_NAME"
    done
done
