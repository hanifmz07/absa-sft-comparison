#!/bin/bash

set -u

# Usage:
#   ./scripts/eval_all_seq2seq_loop_langs.sh [dataset_type] [dataset_folder] [batch_size] [use_constrained_decoding] [languages...]
# Example (defaults):
#   ./scripts/eval_all_seq2seq_loop_langs.sh hotel_reviews mvp 4 false
# Example with specified languages:
#   ./scripts/eval_all_seq2seq_loop_langs.sh hotel_reviews mvp 4 false eng jav mad

DATASET_TYPE="${1:-hotel_reviews}"
DATASET_FOLDER="${2:-mvp}"
BATCH_SIZE="${3:-4}"
USE_CONSTRAINED_DECODING="${4:-false}"

if [[ "$USE_CONSTRAINED_DECODING" != "true" && "$USE_CONSTRAINED_DECODING" != "false" ]]; then
    echo "Error: 4th arg USE_CONSTRAINED_DECODING must be 'true' or 'false'."
    exit 1
fi

if [ "$#" -gt 4 ]; then
    LANGUAGES=("${@:5}")
else
    # LANGUAGES=(eng sun jav mad min)
    LANGUAGES=(indo eng sun jav mad min)
fi

FAILED=()

echo "Starting looped Seq2Seq eval run"
echo "dataset_type=${DATASET_TYPE}, dataset_folder=${DATASET_FOLDER}, batch_size=${BATCH_SIZE}"
echo "use_constrained_decoding=${USE_CONSTRAINED_DECODING}"
echo "languages=${LANGUAGES[*]}"
echo "========================================================"

for LANGUAGE in "${LANGUAGES[@]}"; do
    echo ""
    echo "[RUN] ./scripts/eval_all_seq2seq.sh ${LANGUAGE} ${DATASET_TYPE} ${DATASET_FOLDER} ${BATCH_SIZE} ${USE_CONSTRAINED_DECODING}"

    if ./scripts/eval_all_seq2seq.sh "${LANGUAGE}" "${DATASET_TYPE}" "${DATASET_FOLDER}" "${BATCH_SIZE}" "${USE_CONSTRAINED_DECODING}"; then
        echo "[OK] ${LANGUAGE} completed"
    else
        echo "[FAIL] ${LANGUAGE} failed"
        FAILED+=("${LANGUAGE}")
    fi
done

echo ""
echo "========================================================"
if [ "${#FAILED[@]}" -eq 0 ]; then
    echo "All language evaluations completed successfully."
    exit 0
fi

echo "Completed with failures for language(s): ${FAILED[*]}"
exit 1
