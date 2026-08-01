#!/bin/bash

set -u

# Usage:
#   ./scripts/sft_all_loop_langs.sh [dataset_type] [dataset_folder] [batch_size] [gpu_ids]
# Example (same as your current command, but looped over languages):
#   ./scripts/sft_all_loop_langs.sh hotel_reviews mvp 4
# gpu_ids is a comma-separated list of CUDA device indices to run on in parallel
# (default "0,1"). Languages are dispatched in chunks of size len(gpu_ids), one
# language per GPU per chunk, e.g. with 5 languages and gpu_ids=0,1:
#   chunk1: [eng->GPU0, jav->GPU1] (wait) -> chunk2: [mad->GPU0, min->GPU1] (wait) -> chunk3: [sun->GPU0]

DATASET_TYPE="${1:-hotel_reviews}"
DATASET_FOLDER="${2:-mvp}"
BATCH_SIZE="${3:-4}"
GPU_IDS_RAW="${4:-0,1}"
IFS=',' read -r -a GPU_IDS <<< "$GPU_IDS_RAW"
NUM_GPUS=${#GPU_IDS[@]}

LANGUAGES=(eng jav mad min sun)

FAILED=()

echo "Starting looped SFT run"
echo "dataset_type=${DATASET_TYPE}, dataset_folder=${DATASET_FOLDER}, batch_size=${BATCH_SIZE}"
echo "languages=${LANGUAGES[*]}"
echo "gpu_ids=${GPU_IDS[*]}"
echo "========================================================"

for ((i=0; i<${#LANGUAGES[@]}; i+=NUM_GPUS)); do
    PIDS=()
    CHUNK_LANGS=()

    for ((j=0; j<NUM_GPUS && i+j<${#LANGUAGES[@]}; j++)); do
        LANGUAGE="${LANGUAGES[$((i+j))]}"
        GPU_ID="${GPU_IDS[$j]}"
        echo ""
        echo "[RUN] GPU ${GPU_ID}: ./scripts/sft_all.sh ${LANGUAGE} ${DATASET_TYPE} ${DATASET_FOLDER} ${BATCH_SIZE}"

        CUDA_VISIBLE_DEVICES="${GPU_ID}" ./scripts/sft_all.sh "${LANGUAGE}" "${DATASET_TYPE}" "${DATASET_FOLDER}" "${BATCH_SIZE}" &
        PIDS+=("$!")
        CHUNK_LANGS+=("${LANGUAGE}")
    done

    for ((k=0; k<${#PIDS[@]}; k++)); do
        if wait "${PIDS[$k]}"; then
            echo "[OK] ${CHUNK_LANGS[$k]} completed"
        else
            echo "[FAIL] ${CHUNK_LANGS[$k]} failed"
            FAILED+=("${CHUNK_LANGS[$k]}")
        fi
    done
done

echo ""
echo "========================================================"
if [ "${#FAILED[@]}" -eq 0 ]; then
    echo "All language runs completed successfully."
    exit 0
fi

echo "Completed with failures for language(s): ${FAILED[*]}"
exit 1
