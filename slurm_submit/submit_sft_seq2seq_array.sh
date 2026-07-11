#!/bin/bash
#SBATCH --job-name=sft-seq2seq-array
#SBATCH --partition=mi250x
#SBATCH --ntasks=1
#SBATCH --gres=gpu:MI250:1
#SBATCH --mem=32GB
#SBATCH --time=08:00:00
#SBATCH --array=0-29%5
#SBATCH --output=logs/out/sft-seq2seq-array-%A_%a.out
#SBATCH --error=logs/err/sft-seq2seq-array-%A_%a.err

set -euo pipefail

cd ~/absa-sft-comparison

LANGUAGES=(
    "eng"
    "jav"
    "indo"
    "mad"
    "min"
    "sun"
)

SEEDS=(
    "9584"
    "123"
    "2024"
    "31415"
    "777"
)

DATASET_TYPE="hotel_reviews"
DATASET_FOLDER="mvp_aos"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-2e-4}"
NUM_EPOCHS="${NUM_EPOCHS:-20}"
EVAL_STRATEGY="${EVAL_STRATEGY:-epoch}"
SAVE_STRATEGY="${SAVE_STRATEGY:-best}"

NUM_LANGUAGES="${#LANGUAGES[@]}"
NUM_SEEDS="${#SEEDS[@]}"
TOTAL_JOBS=$((NUM_LANGUAGES * NUM_SEEDS))

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not set. Submit this script with sbatch as a job array.}"

if [ "$TASK_ID" -ge "$TOTAL_JOBS" ]; then
    echo "Task ${TASK_ID} is outside the configured grid size (${TOTAL_JOBS}); exiting."
    exit 0
fi

idx="$TASK_ID"

seed_idx=$((idx % NUM_SEEDS))
idx=$((idx / NUM_SEEDS))

language_idx=$((idx % NUM_LANGUAGES))

LANGUAGE="${LANGUAGES[$language_idx]}"
SEED="${SEEDS[$seed_idx]}"

echo "Resolved seq2seq SFT array task:"
echo "  task_id=${TASK_ID}/${TOTAL_JOBS}"
echo "  language=${LANGUAGE}"
echo "  dataset_type=${DATASET_TYPE}"
echo "  dataset_folder=${DATASET_FOLDER}"
echo "  seed=${SEED}"
echo "  batch_size=${BATCH_SIZE}"
echo "  lr=${LR}"
echo "  num_epochs=${NUM_EPOCHS}"
echo "  eval_strategy=${EVAL_STRATEGY}"
echo "  save_strategy=${SAVE_STRATEGY}"

bash scripts/sft_one_seq2seq.sh \
    "$LANGUAGE" \
    "$DATASET_TYPE" \
    "$DATASET_FOLDER" \
    "$SEED" \
    "$BATCH_SIZE" \
    "$LR" \
    "$NUM_EPOCHS" \
    "$EVAL_STRATEGY" \
    "$SAVE_STRATEGY"
