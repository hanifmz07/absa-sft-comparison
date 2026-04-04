#!/bin/bash

set -u

# Usage:
#   ./scripts/sft_all_loop_langs.sh [dataset_type] [dataset_folder] [batch_size]
# Example (same as your current command, but looped over languages):
#   ./scripts/sft_all_loop_langs.sh hotel_reviews mvp 4

DATASET_TYPE="${1:-hotel_reviews}"
DATASET_FOLDER="${2:-mvp}"
BATCH_SIZE="${3:-4}"

LANGUAGES=(eng jav mad min sun)

FAILED=()

echo "Starting looped SFT run"
echo "dataset_type=${DATASET_TYPE}, dataset_folder=${DATASET_FOLDER}, batch_size=${BATCH_SIZE}"
echo "languages=${LANGUAGES[*]}"
echo "========================================================"

for LANGUAGE in "${LANGUAGES[@]}"; do
    echo ""
    echo "[RUN] ./scripts/sft_all.sh ${LANGUAGE} ${DATASET_TYPE} ${DATASET_FOLDER} ${BATCH_SIZE}"

    if ./scripts/sft_all.sh "${LANGUAGE}" "${DATASET_TYPE}" "${DATASET_FOLDER}" "${BATCH_SIZE}"; then
        echo "[OK] ${LANGUAGE} completed"
    else
        echo "[FAIL] ${LANGUAGE} failed"
        FAILED+=("${LANGUAGE}")
    fi
done

echo ""
echo "========================================================"
if [ "${#FAILED[@]}" -eq 0 ]; then
    echo "All language runs completed successfully."
    exit 0
fi

echo "Completed with failures for language(s): ${FAILED[*]}"
exit 1
