#!/bin/bash

set -u

# Usage:
#   ./scripts/eval_all_noconstraint_loop_langs.sh [dataset_type] [dataset_folder] [batch_size] [languages...]
# Example with defaults:
#   ./scripts/eval_all_noconstraint_loop_langs.sh hotel_reviews mvp 4
# Example with specified languages:
#   ./scripts/eval_all_noconstraint_loop_langs.sh hotel_reviews mvp 4 eng jav mad

DATASET_TYPE="${1:-hotel_reviews}"
DATASET_FOLDER="${2:-mvp}"
BATCH_SIZE="${3:-4}"

# Default language set follows eval_all_noconstraint.sh validation.
if [ "$#" -gt 3 ]; then
    LANGUAGES=("${@:4}")
else
    LANGUAGES=(eng sun jav mad min)
    # LANGUAGES=(indo eng sun jav mad min)
fi

FAILED=()

echo "Starting looped no-constraint eval run"
echo "dataset_type=${DATASET_TYPE}, dataset_folder=${DATASET_FOLDER}, batch_size=${BATCH_SIZE}"
echo "languages=${LANGUAGES[*]}"
echo "========================================================"

for LANGUAGE in "${LANGUAGES[@]}"; do
    echo ""
    echo "[RUN] ./scripts/eval_all_noconstraint.sh ${LANGUAGE} ${DATASET_TYPE} ${DATASET_FOLDER} ${BATCH_SIZE}"

    if ./scripts/eval_all_noconstraint.sh "${LANGUAGE}" "${DATASET_TYPE}" "${DATASET_FOLDER}" "${BATCH_SIZE}"; then
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
