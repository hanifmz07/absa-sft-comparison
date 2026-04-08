#!/bin/bash

set -u

source .venv/bin/activate

# Specify CUDA device if needed
export CUDA_VISIBLE_DEVICES=0

LANGUAGE="$1"
if [ -z "$LANGUAGE" ]; then
    echo "Error: No language specified."
    exit 1
fi

# Validate the language argument
if [[ "$LANGUAGE" != "indo" && "$LANGUAGE" != "eng" && "$LANGUAGE" != "sunda" && "$LANGUAGE" != "sun" && "$LANGUAGE" != "jav" && "$LANGUAGE" != "mad" && "$LANGUAGE" != "min" ]]; then
    echo "Error: Invalid language specified. Use 'indo', 'eng', 'sunda', 'sun', 'jav', 'mad', or 'min'."
    exit 1
fi

DATASET_TYPE="$2"
if [ -z "$DATASET_TYPE" ]; then
    echo "Error: Dataset type must be specified."
    exit 1
fi

DATASET_FOLDER="$3"
if [ -z "$DATASET_FOLDER" ]; then
    echo "Error: Dataset folder must be specified."
    exit 1
fi

BATCH_SIZE="$4"
if [ -z "$BATCH_SIZE" ]; then
    echo "Error: Batch size must be specified."
    exit 1
fi

# Constrained decoding is optional (default: false)
USE_CONSTRAINED_DECODING="${5:-false}"
if [[ "$USE_CONSTRAINED_DECODING" != "true" && "$USE_CONSTRAINED_DECODING" != "false" ]]; then
    echo "Error: 5th arg USE_CONSTRAINED_DECODING must be 'true' or 'false'."
    exit 1
fi

SEEDS=(9584 123 2024 31415 777)

if [ "$DATASET_FOLDER" == "mvp_aos" ] || [ "$DATASET_FOLDER" == "mvp_aos_augment" ] || [ "$DATASET_FOLDER" == "mvp" ]; then
    PROMPT_TYPE="mvp"
elif [ "$DATASET_FOLDER" == "gas" ]; then
    PROMPT_TYPE="gas"
elif [ "$DATASET_FOLDER" == "legoabsa_multitask" ] || [ "$DATASET_FOLDER" == "legoabsa_tasktransfer" ] || [ "$DATASET_FOLDER" == "indolegoabsa_multitask" ]; then
    PROMPT_TYPE="legoabsa"
else
    echo "Error: Unexpected dataset folder '$DATASET_FOLDER'. Cannot determine prompt type."
    exit 1
fi

LOG_BASE_NAME="eval_seq2seq"
LOG_DIR="logs"
PID=$$
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_LOG_DIR="${LOG_DIR}/${LOG_BASE_NAME}/${DATASET_TYPE}/${LANGUAGE}/${DATASET_FOLDER}"
mkdir -p "$RUN_LOG_DIR"

STDOUT_LOG="${RUN_LOG_DIR}/${PID}_${TIMESTAMP}.log"
STDERR_LOG="${RUN_LOG_DIR}/${PID}_${TIMESTAMP}.err"

{
  echo "========================================================"
  echo "Starting Seq2Seq evaluation script run at: $(date)"
  echo "dataset_type=${DATASET_TYPE}, language=${LANGUAGE}, dataset_folder=${DATASET_FOLDER}, batch_size=${BATCH_SIZE}"
  echo "prompt_type=${PROMPT_TYPE}, use_constrained_decoding=${USE_CONSTRAINED_DECODING}"
  echo "========================================================"

  for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "Processing seed: $SEED"

    TEST_JSON="dataset/${DATASET_TYPE}/${LANGUAGE}/${DATASET_FOLDER}/test.json"
    MODEL_DIR="outputs_seq2seq/models/${DATASET_TYPE}/${LANGUAGE}/${DATASET_FOLDER}/seed_${SEED}"
    OUTPUT_DIR="outputs_seq2seq/evals/${DATASET_TYPE}/${LANGUAGE}/${DATASET_FOLDER}"

    if [ ! -d "$MODEL_DIR" ]; then
      echo "Skipping seed ${SEED}: model directory not found at ${MODEL_DIR}"
      continue
    fi

    # Seq2Seq training stores checkpoints under an intermediate run folder, e.g.
    # seed_x/<timestamp>_train_.../checkpoint-xxxx. Discover recursively and pick latest by mtime.
    LATEST_CHECKPOINT=$(find "$MODEL_DIR" -maxdepth 4 -type d -name "checkpoint-*" -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)

    if [ -z "$LATEST_CHECKPOINT" ]; then
      echo "Skipping seed ${SEED}: no checkpoint-* found under ${MODEL_DIR} (including nested run folders)"
      continue
    fi

    PARENT_DIR_NAME=$(basename "$(dirname "$LATEST_CHECKPOINT")")
    CHECKPOINT_NAME=$(basename "$LATEST_CHECKPOINT")
    EVAL_RESULT_DIR="${OUTPUT_DIR}/${PARENT_DIR_NAME}/${CHECKPOINT_NAME}"

    if [ "$USE_CONSTRAINED_DECODING" == "true" ]; then
      RESULT_SUBDIR="constrained_decoding"
    else
      RESULT_SUBDIR="unconstrained_decoding"
    fi

    if [ -d "${EVAL_RESULT_DIR}/${RESULT_SUBDIR}" ]; then
      echo "Skipping seed ${SEED} (${CHECKPOINT_NAME}) - already evaluated."
      continue
    fi

    echo "-----------------------------------------------------------"
    echo "Evaluating Seq2Seq checkpoint: ${LATEST_CHECKPOINT}"
    echo "-----------------------------------------------------------"

    CMD=(
      python -m src.main.eval_seq2seq
      --test_json_path "$TEST_JSON"
      --model_path "$LATEST_CHECKPOINT"
      --prompt_type "$PROMPT_TYPE"
      --output_dir "$OUTPUT_DIR"
      --batch_size "$BATCH_SIZE"
      --save_predictions
    )

    if [ "$USE_CONSTRAINED_DECODING" == "true" ]; then
      CMD+=(--use_constrained_decoding)
    fi

    "${CMD[@]}"
  done

  echo ""
  echo "========================================================"
  echo "All Seq2Seq evaluations completed at: $(date)"
  echo "========================================================"

} 2> >(tee "$STDERR_LOG") | tee "$STDOUT_LOG"
