#!/bin/bash
#SBATCH --job-name=eval-semantic-seq2seq-array
#SBATCH --partition=mi250x
#SBATCH --ntasks=1
#SBATCH --gres=gpu:MI250:1
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --array=0-5%6
#SBATCH --output=logs/out/eval-semantic-seq2seq-array-%A_%a.out
#SBATCH --error=logs/err/eval-semantic-seq2seq-array-%A_%a.err

set -euo pipefail

cd ~/absa-sft-comparison

DATASET_TYPE="${DATASET_TYPE:-hotel_reviews}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs_seq2seq}"
EMBEDDING_MODEL_NAME="${EMBEDDING_MODEL_NAME:-Qwen/Qwen3-Embedding-8B}"

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

echo "Resolved semantic similarity eval array task (seq2seq):"
echo "  task_id=${TASK_ID}/${TOTAL_JOBS}"
echo "  language=${LANGUAGE}"
echo "  dataset_type=${DATASET_TYPE}"
echo "  dataset_folder=${DATASET_FOLDER}"
echo "  output_dir=${OUTPUT_DIR}"
echo "  embedding_model=${EMBEDDING_MODEL_NAME}"

bash scripts/eval_semantic_similarity.sh \
    "$OUTPUT_DIR" \
    "$DATASET_TYPE" \
    "$LANGUAGE" \
    "$DATASET_FOLDER" \
    "$EMBEDDING_MODEL_NAME"
