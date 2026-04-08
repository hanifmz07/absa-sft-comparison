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
if [[ "$LANGUAGE" != "indo" && "$LANGUAGE" != "eng" && "$LANGUAGE" != "sunda" && "$LANGUAGE" != "jav" && "$LANGUAGE" != "mad" && "$LANGUAGE" != "sun" && "$LANGUAGE" != "min" ]]; then
    echo "Error: Invalid language specified. Use 'indo', 'eng', 'sunda', 'jav', 'mad', 'sun', or 'min'."
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

# Seeds for the SFT process
SEEDS=(9584 123 2024 31415 777)

# Determine prompt type from dataset folder
if [ "$DATASET_FOLDER" == "mvp_aos" ] || [ "$DATASET_FOLDER" == "mvp_aos_augment" ] || [ "$DATASET_FOLDER" == "mvp" ]; then
    PROMPT_TYPE="mvp"
elif [ "$DATASET_FOLDER" == "gas" ]; then
    PROMPT_TYPE="gas"
elif [ "$DATASET_FOLDER" == "legoabsa_multitask" ] || [ "$DATASET_FOLDER" == "legoabsa_tasktransfer" ] || [ "$DATASET_FOLDER" == "indolegoabsa_multitask" ]; then
    PROMPT_TYPE="legoabsa"
else
    echo "Unknown dataset folder: $DATASET_FOLDER"
    exit 1
fi

LOG_BASE_NAME="sft_seq2seq_full"
LOG_DIR="logs"
PID=$$
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

RUN_LOG_DIR="${LOG_DIR}/${LOG_BASE_NAME}/${DATASET_TYPE}/${LANGUAGE}/${DATASET_FOLDER}"
mkdir -p "$RUN_LOG_DIR"

STDOUT_LOG="${RUN_LOG_DIR}/${PID}_${TIMESTAMP}.log"
STDERR_LOG="${RUN_LOG_DIR}/${PID}_${TIMESTAMP}.err"

echo "========================================================" > "$STDOUT_LOG"
echo "Starting Seq2Seq SFT script run at: $(date)" >> "$STDOUT_LOG"
echo "========================================================" >> "$STDOUT_LOG"

>"$STDERR_LOG"

{
    echo "Running ABSA Seq2Seq SFT (mT5)"
    echo "dataset_type=${DATASET_TYPE}, language=${LANGUAGE}, dataset_folder=${DATASET_FOLDER}, batch_size=${BATCH_SIZE}"
    echo "prompt_type=${PROMPT_TYPE}"

    for SEED in "${SEEDS[@]}"; do
        echo ""
        echo "--------------------------------------------------------"
        echo "Processing seed: $SEED"
        echo "--------------------------------------------------------"

        python -m src.main.train_seq2seq \
            --train_json_path "dataset/${DATASET_TYPE}/${LANGUAGE}/${DATASET_FOLDER}/train.json" \
            --model_name "google/mt5-base" \
            --output_dir "outputs_seq2seq/models/${DATASET_TYPE}/${LANGUAGE}/${DATASET_FOLDER}/seed_${SEED}/" \
            --prompt_type "$PROMPT_TYPE" \
            --save_strategy "epoch" \
            --num_epochs 20 \
            --lr 2e-4 \
            --optimizer "adamw_torch" \
            --seed "$SEED" \
            --batch_size "$BATCH_SIZE" \
            --eval_strategy "no"
            # --val_json_path "dataset/${DATASET_TYPE}/${LANGUAGE}/${DATASET_FOLDER}/dev.json" \
            # --val_batch_size 16
    done

    echo ""
    echo "========================================================"
    echo "All seeds completed at: $(date)"
    echo "========================================================"

} 2> >(tee "$STDERR_LOG") | tee "$STDOUT_LOG"
