#!/bin/bash
#SBATCH --job-name=sft-array
#SBATCH --partition=mi250x
#SBATCH --ntasks=1
#SBATCH --gres=gpu:MI250:1
#SBATCH --mem=32GB
#SBATCH --time=08:00:00
#SBATCH --array=0-29%5
#SBATCH --output=logs/out/sft-array-%A_%a.out
#SBATCH --error=logs/err/sft-array-%A_%a.err

set -euo pipefail

cd ~/absa-sft-comparison

# Edit these arrays to define the experiment grid.
# Total jobs = models * languages * dataset_folders * seeds.
MODELS=(
    # "google/gemma-3-270m"
    "Qwen/Qwen2.5-0.5B"
)

LANGUAGES=(
    "eng"
    # "jav"
    # "indo"
    # "mad"
    # "min"
    # "sun"
)

DATASET_TYPE="hotel_reviews"
DATASET_FOLDERS=(
    "mvp_aos"
)

SEEDS=(
    "9584"
    "123"
    "2024"
    "31415"
    "777"
)

BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-1e-5}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
EVAL_STRATEGY="${EVAL_STRATEGY:-epoch}"
SAVE_STRATEGY="${SAVE_STRATEGY:-epoch}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"

NUM_MODELS="${#MODELS[@]}"
NUM_LANGUAGES="${#LANGUAGES[@]}"
NUM_DATASET_FOLDERS="${#DATASET_FOLDERS[@]}"
NUM_SEEDS="${#SEEDS[@]}"
TOTAL_JOBS=$((NUM_MODELS * NUM_LANGUAGES * NUM_DATASET_FOLDERS * NUM_SEEDS))

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not set. Submit this script with sbatch as a job array.}"

if [ "$TASK_ID" -ge "$TOTAL_JOBS" ]; then
    echo "Task ${TASK_ID} is outside the configured grid size (${TOTAL_JOBS}); exiting."
    exit 0
fi

idx="$TASK_ID"

seed_idx=$((idx % NUM_SEEDS))
idx=$((idx / NUM_SEEDS))

dataset_folder_idx=$((idx % NUM_DATASET_FOLDERS))
idx=$((idx / NUM_DATASET_FOLDERS))

language_idx=$((idx % NUM_LANGUAGES))
idx=$((idx / NUM_LANGUAGES))

model_idx=$((idx % NUM_MODELS))

MODEL_NAME="${MODELS[$model_idx]}"
LANGUAGE="${LANGUAGES[$language_idx]}"
DATASET_FOLDER="${DATASET_FOLDERS[$dataset_folder_idx]}"
SEED="${SEEDS[$seed_idx]}"

echo "Resolved SFT array task:"
echo "  task_id=${TASK_ID}/${TOTAL_JOBS}"
echo "  model=${MODEL_NAME}"
echo "  language=${LANGUAGE}"
echo "  dataset_type=${DATASET_TYPE}"
echo "  dataset_folder=${DATASET_FOLDER}"
echo "  seed=${SEED}"
echo "  batch_size=${BATCH_SIZE}"
echo "  lr=${LR}"
echo "  num_epochs=${NUM_EPOCHS}"
echo "  gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}"
echo "  eval_strategy=${EVAL_STRATEGY}"
echo "  save_strategy=${SAVE_STRATEGY}"
echo "  max_grad_norm=${MAX_GRAD_NORM}"
echo "  warmup_ratio=${WARMUP_RATIO}"

bash scripts/sft_one.sh \
    "$MODEL_NAME" \
    "$LANGUAGE" \
    "$DATASET_TYPE" \
    "$DATASET_FOLDER" \
    "$SEED" \
    "$BATCH_SIZE" \
    "$LR" \
    "$NUM_EPOCHS" \
    "$GRADIENT_ACCUMULATION_STEPS" \
    "$EVAL_STRATEGY" \
    "$SAVE_STRATEGY" \
    "$MAX_GRAD_NORM" \
    "$WARMUP_RATIO"
