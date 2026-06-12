#!/bin/bash
#SBATCH --job-name=eval-noconstraint-array
#SBATCH --partition=mi250x
#SBATCH --ntasks=1
#SBATCH --gres=gpu:MI250:1
#SBATCH --mem=32GB
#SBATCH --time=08:00:00
#SBATCH --array=0-11%10
#SBATCH --output=logs/out/eval-noconstraint-array-%A_%a.out
#SBATCH --error=logs/err/eval-noconstraint-array-%A_%a.err

set -euo pipefail

cd ~/absa-sft-comparison

# This array version splits the language and dataset folder loop into separate Slurm array tasks.
DATASET_TYPE="${DATASET_TYPE:-hotel_reviews}"
BATCH_SIZE="${BATCH_SIZE:-4}"

LANGUAGES=(
    "eng"
    "indo"
    "sun"
    "jav"
    "mad"
    "min"
)

DATASET_FOLDERS=(
    "mvp_aos"
    "mvp"
)

NUM_LANGUAGES="${#LANGUAGES[@]}"
NUM_DATASET_FOLDERS="${#DATASET_FOLDERS[@]}"
TOTAL_JOBS=$((NUM_LANGUAGES * NUM_DATASET_FOLDERS))

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not set. Submit this script with sbatch as a job array.}"

if [ "$TASK_ID" -ge "$TOTAL_JOBS" ]; then
    echo "Task ${TASK_ID} is outside the configured grid size (${TOTAL_JOBS}); exiting."
    exit 0
fi

idx="$TASK_ID"
dataset_folder_idx=$((idx % NUM_DATASET_FOLDERS))
idx=$((idx / NUM_DATASET_FOLDERS))
language_idx=$((idx % NUM_LANGUAGES))

LANGUAGE="${LANGUAGES[$language_idx]}"
DATASET_FOLDER="${DATASET_FOLDERS[$dataset_folder_idx]}"

echo "Resolved no-constraint eval array task:"
echo "  task_id=${TASK_ID}/${TOTAL_JOBS}"
echo "  language=${LANGUAGE}"
echo "  dataset_type=${DATASET_TYPE}"
echo "  dataset_folder=${DATASET_FOLDER}"
echo "  batch_size=${BATCH_SIZE}"

bash scripts/eval_all_noconstraint_loop_langs.sh \
    "$DATASET_TYPE" \
    "$DATASET_FOLDER" \
    "$BATCH_SIZE" \
    "$LANGUAGE"
