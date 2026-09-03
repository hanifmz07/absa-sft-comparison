#!/bin/bash

set -u

source .venv/bin/activate

# Usage:
#   bash scripts/eval_all_semantic_similarity.sh <output_dir> <embedding_model_name> [gpu_ids]
# Example:
#   bash scripts/eval_all_semantic_similarity.sh outputs my-embedding-model 0,1
# gpu_ids is a comma-separated list of CUDA device indices to run on in parallel
# (default "0,1"). Languages are pulled from a shared work queue by GPU workers,
# so a GPU that finishes early (or skips a language with no data) immediately
# picks up the next pending language instead of waiting on the other GPU(s).

OUTPUT_DIR="${1:-}"
EMBEDDING_MODEL_NAME="${2:-}"
GPU_IDS_RAW="${3:-0,1}"

if [ -z "$OUTPUT_DIR" ]; then
    echo "Error: output_dir is required."
    echo "Usage: bash scripts/eval_all_semantic_similarity.sh <output_dir> <embedding_model_name> [gpu_ids]"
    exit 1
fi

if [ -z "$EMBEDDING_MODEL_NAME" ]; then
    echo "Error: embedding_model_name is required."
    echo "Usage: bash scripts/eval_all_semantic_similarity.sh <output_dir> <embedding_model_name> [gpu_ids]"
    exit 1
fi

IFS=',' read -r -a GPU_IDS <<< "$GPU_IDS_RAW"
NUM_GPUS=${#GPU_IDS[@]}

LANGS=(eng indo jav mad min sun)
DATASET_TYPE="hotel_reviews"
DATASET_FOLDERS=(mvp_aos mvp)

QUEUE_DIR="$(mktemp -d)"
mkdir -p "$QUEUE_DIR/pending" "$QUEUE_DIR/claimed" "$QUEUE_DIR/failed"
trap 'rm -rf "$QUEUE_DIR"' EXIT

for LANG in "${LANGS[@]}"; do
    : > "$QUEUE_DIR/pending/$LANG"
done

echo "Starting semantic similarity eval run"
echo "output_dir=${OUTPUT_DIR}, embedding_model_name=${EMBEDDING_MODEL_NAME}"
echo "languages=${LANGS[*]}"
echo "gpu_ids=${GPU_IDS[*]}"
echo "========================================================"

worker() {
    local GPU_ID="$1"
    local LANG FOUND_ANY_DATA HAD_ERROR SEARCH_ROOT CANDIDATE REMAINING

    while true; do
        LANG=""
        for CANDIDATE in "$QUEUE_DIR/pending"/*; do
            [ -e "$CANDIDATE" ] || continue
            if mv "$CANDIDATE" "$QUEUE_DIR/claimed/$(basename "$CANDIDATE")" 2>/dev/null; then
                LANG="$(basename "$CANDIDATE")"
                break
            fi
        done

        if [ -z "$LANG" ]; then
            REMAINING=("$QUEUE_DIR/pending"/*)
            [ -e "${REMAINING[0]}" ] && continue
            break
        fi

        echo ""
        echo "[RUN] GPU ${GPU_ID}: lang=${LANG}"

        FOUND_ANY_DATA=0
        HAD_ERROR=0
        for DATASET_FOLDER in "${DATASET_FOLDERS[@]}"; do
            SEARCH_ROOT="$OUTPUT_DIR/evals/$DATASET_TYPE/$LANG/$DATASET_FOLDER"
            if [ ! -d "$SEARCH_ROOT" ]; then
                echo "[SKIP] GPU ${GPU_ID}: no data for lang=$LANG dataset_folder=$DATASET_FOLDER"
                continue
            fi
            FOUND_ANY_DATA=1

            echo "========================================================"
            echo "Running semantic eval for: dataset_type=$DATASET_TYPE lang=$LANG dataset_folder=$DATASET_FOLDER (GPU ${GPU_ID})"
            echo "========================================================"

            CUDA_VISIBLE_DEVICES="${GPU_ID}" bash scripts/eval_semantic_similarity.sh \
                "$OUTPUT_DIR" \
                "$DATASET_TYPE" \
                "$LANG" \
                "$DATASET_FOLDER" \
                "$EMBEDDING_MODEL_NAME" || HAD_ERROR=1
        done

        if [ "$HAD_ERROR" -eq 1 ]; then
            echo "[FAIL] ${LANG} failed"
            : > "$QUEUE_DIR/failed/$LANG"
        elif [ "$FOUND_ANY_DATA" -eq 0 ]; then
            echo "[SKIP] ${LANG} has no data under any dataset folder, skipping"
        else
            echo "[OK] ${LANG} completed"
        fi
    done
}

WORKER_PIDS=()
for GPU_ID in "${GPU_IDS[@]}"; do
    worker "$GPU_ID" &
    WORKER_PIDS+=("$!")
done

for PID in "${WORKER_PIDS[@]}"; do
    wait "$PID"
done

FAILED=("$QUEUE_DIR"/failed/*)
echo ""
echo "========================================================"
if [ ! -e "${FAILED[0]}" ]; then
    echo "All language evaluations completed successfully."
    exit 0
fi

FAILED_NAMES=()
for F in "${FAILED[@]}"; do
    FAILED_NAMES+=("$(basename "$F")")
done
echo "Completed with failures for language(s): ${FAILED_NAMES[*]}"
exit 1
