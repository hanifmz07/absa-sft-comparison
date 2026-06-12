#!/bin/bash

set -euo pipefail

# Run one SFT configuration. This is intended to be called by Slurm array jobs.
#
# Usage:
#   bash scripts/sft_one.sh MODEL_NAME LANGUAGE DATASET_TYPE DATASET_FOLDER SEED BATCH_SIZE [LR] [EPOCHS] [GRAD_ACCUM] [EVAL_STRATEGY] [SAVE_STRATEGY]

MODEL_NAME="${1:?Error: model name must be specified.}"
LANGUAGE="${2:?Error: language must be specified.}"
DATASET_TYPE="${3:?Error: dataset type must be specified.}"
DATASET_FOLDER="${4:?Error: dataset folder must be specified.}"
SEED="${5:?Error: seed must be specified.}"
BATCH_SIZE="${6:?Error: batch size must be specified.}"
LR="${7:-5e-5}"
NUM_EPOCHS="${8:-10}"
GRADIENT_ACCUMULATION_STEPS="${9:-4}"
EVAL_STRATEGY="${10:-no}"
SAVE_STRATEGY="${11:-epoch}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"

case "$LANGUAGE" in
    indo|eng|sunda|jav|mad|sun|min) ;;
    *)
        echo "Error: invalid language '$LANGUAGE'."
        exit 1
        ;;
esac

case "$DATASET_FOLDER" in
    mvp|mvp_aos|mvp_aos_augment)
        PROMPT_TYPE="mvp"
        ;;
    gas)
        PROMPT_TYPE="gas"
        ;;
    legoabsa_multitask|legoabsa_tasktransfer|indolegoabsa_multitask)
        PROMPT_TYPE="legoabsa"
        ;;
    *)
        echo "Error: unknown dataset folder '$DATASET_FOLDER'."
        exit 1
        ;;
esac

TRAIN_JSON_PATH="dataset/${DATASET_TYPE}/${LANGUAGE}/${DATASET_FOLDER}/train.json"
VAL_JSON_PATH="dataset/${DATASET_TYPE}/${LANGUAGE}/${DATASET_FOLDER}/dev.json"
OUTPUT_DIR="outputs/models/${DATASET_TYPE}/${LANGUAGE}/${DATASET_FOLDER}/seed_${SEED}/"

if [ ! -f "$TRAIN_JSON_PATH" ]; then
    echo "Error: training dataset not found: $TRAIN_JSON_PATH"
    exit 1
fi

source .venv/bin/activate

MODEL_SLUG="$(echo "$MODEL_NAME" | tr '/:' '__')"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/sft_array/${DATASET_TYPE}/${LANGUAGE}/${DATASET_FOLDER}/seed_${SEED}/${MODEL_SLUG}"
mkdir -p "$LOG_DIR"

STDOUT_LOG="${LOG_DIR}/${RUN_STAMP}_job-${SLURM_JOB_ID:-local}_task-${SLURM_ARRAY_TASK_ID:-0}.log"
STDERR_LOG="${LOG_DIR}/${RUN_STAMP}_job-${SLURM_JOB_ID:-local}_task-${SLURM_ARRAY_TASK_ID:-0}.err"

{
    echo "========================================================"
    echo "Starting SFT single run at: $(date)"
    echo "Job ID: ${SLURM_JOB_ID:-local}"
    echo "Array Task ID: ${SLURM_ARRAY_TASK_ID:-0}"
    echo "Model: $MODEL_NAME"
    echo "Language: $LANGUAGE"
    echo "Dataset Type: $DATASET_TYPE"
    echo "Dataset Folder: $DATASET_FOLDER"
    echo "Prompt Type: $PROMPT_TYPE"
    echo "Seed: $SEED"
    echo "Batch Size: $BATCH_SIZE"
    echo "Learning Rate: $LR"
    echo "Epochs: $NUM_EPOCHS"
    echo "Gradient Accumulation Steps: $GRADIENT_ACCUMULATION_STEPS"
    echo "Eval Strategy: $EVAL_STRATEGY"
    echo "Save Strategy: $SAVE_STRATEGY"
    echo "========================================================"

    CMD=(
        python -m src.main.train
        --train_json_path "$TRAIN_JSON_PATH"
        --model_name "$MODEL_NAME"
        --output_dir "$OUTPUT_DIR"
        --prompt_type "$PROMPT_TYPE"
        --save_strategy "$SAVE_STRATEGY"
        --num_epochs "$NUM_EPOCHS"
        --lr "$LR"
        --optimizer "adamw_torch"
        --seed "$SEED"
        --batch_size "$BATCH_SIZE"
        --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
        --eval_strategy "$EVAL_STRATEGY"
    )

    if [ "$EVAL_STRATEGY" != "no" ]; then
        if [ ! -f "$VAL_JSON_PATH" ]; then
            echo "Error: validation dataset not found: $VAL_JSON_PATH"
            exit 1
        fi
        CMD+=(--val_json_path "$VAL_JSON_PATH" --val_batch_size "$VAL_BATCH_SIZE")
    fi

    printf 'Command:'
    printf ' %q' "${CMD[@]}"
    printf '\n'

    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "DRY_RUN=1, skipping training command."
        exit 0
    fi

    "${CMD[@]}"

    echo "========================================================"
    echo "Completed SFT single run at: $(date)"
    echo "========================================================"
} > >(tee "$STDOUT_LOG") 2> >(tee "$STDERR_LOG" >&2)
