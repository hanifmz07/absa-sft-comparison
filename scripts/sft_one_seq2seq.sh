#!/bin/bash

set -euo pipefail

# Run one seq2seq SFT configuration. Intended to be called by Slurm array jobs.
#
# Usage:
#   bash scripts/sft_one_seq2seq.sh LANGUAGE DATASET_TYPE DATASET_FOLDER SEED BATCH_SIZE [LR] [EPOCHS] [EVAL_STRATEGY] [SAVE_STRATEGY]

LANGUAGE="${1:?Error: language must be specified.}"
DATASET_TYPE="${2:?Error: dataset type must be specified.}"
DATASET_FOLDER="${3:?Error: dataset folder must be specified.}"
SEED="${4:?Error: seed must be specified.}"
BATCH_SIZE="${5:?Error: batch size must be specified.}"
LR="${6:-2e-4}"
NUM_EPOCHS="${7:-20}"
EVAL_STRATEGY="${8:-no}"
SAVE_STRATEGY="${9:-epoch}"
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
OUTPUT_DIR="outputs_seq2seq/models/${DATASET_TYPE}/${LANGUAGE}/${DATASET_FOLDER}/seed_${SEED}/"

if [ ! -f "$TRAIN_JSON_PATH" ]; then
    echo "Error: training dataset not found: $TRAIN_JSON_PATH"
    exit 1
fi

source .venv/bin/activate

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/sft_seq2seq_array/${DATASET_TYPE}/${LANGUAGE}/${DATASET_FOLDER}/seed_${SEED}"
mkdir -p "$LOG_DIR"

STDOUT_LOG="${LOG_DIR}/${RUN_STAMP}_job-${SLURM_JOB_ID:-local}_task-${SLURM_ARRAY_TASK_ID:-0}.log"
STDERR_LOG="${LOG_DIR}/${RUN_STAMP}_job-${SLURM_JOB_ID:-local}_task-${SLURM_ARRAY_TASK_ID:-0}.err"

{
    echo "========================================================"
    echo "Starting seq2seq SFT single run at: $(date)"
    echo "Job ID: ${SLURM_JOB_ID:-local}"
    echo "Array Task ID: ${SLURM_ARRAY_TASK_ID:-0}"
    echo "Language: $LANGUAGE"
    echo "Dataset Type: $DATASET_TYPE"
    echo "Dataset Folder: $DATASET_FOLDER"
    echo "Prompt Type: $PROMPT_TYPE"
    echo "Seed: $SEED"
    echo "Batch Size: $BATCH_SIZE"
    echo "Learning Rate: $LR"
    echo "Epochs: $NUM_EPOCHS"
    echo "Eval Strategy: $EVAL_STRATEGY"
    echo "Save Strategy: $SAVE_STRATEGY"
    echo "========================================================"

    CMD=(
        python -m src.main.train_seq2seq
        --train_json_path "$TRAIN_JSON_PATH"
        --model_name "google/mt5-base"
        --output_dir "$OUTPUT_DIR"
        --prompt_type "$PROMPT_TYPE"
        --save_strategy "$SAVE_STRATEGY"
        --num_epochs "$NUM_EPOCHS"
        --lr "$LR"
        --optimizer "adamw_torch"
        --seed "$SEED"
        --batch_size "$BATCH_SIZE"
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
    echo "Completed seq2seq SFT single run at: $(date)"
    echo "========================================================"
} > >(tee "$STDOUT_LOG") 2> >(tee "$STDERR_LOG" >&2)
