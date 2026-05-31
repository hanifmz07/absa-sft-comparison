#!/bin/bash
#SBATCH --job-name=eval-noconstraint-array
#SBATCH --partition=mi250x
#SBATCH --ntasks=1
#SBATCH --gres=gpu:MI250:1
#SBATCH --mem=32GB
#SBATCH --time=08:00:00
#SBATCH --array=0-5%5
#SBATCH --output=logs/out/eval-noconstraint-array-%A_%a.out
#SBATCH --error=logs/err/eval-noconstraint-array-%A_%a.err

set -euo pipefail

cd ~/absa-sft-comparison

# This array version is equivalent to:
#   ./scripts/eval_all_noconstraint_loop_langs.sh hotel_reviews mvp_aos 4
# but splits the language loop into separate Slurm array tasks.
DATASET_TYPE="${DATASET_TYPE:-hotel_reviews}"
DATASET_FOLDER="${DATASET_FOLDER:-mvp_aos}"
BATCH_SIZE="${BATCH_SIZE:-4}"

LANGUAGES=(
    "eng"
    "indo"
    "sun"
    "jav"
    "mad"
    "min"
)

NUM_LANGUAGES="${#LANGUAGES[@]}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not set. Submit this script with sbatch as a job array.}"

if [ "$TASK_ID" -ge "$NUM_LANGUAGES" ]; then
    echo "Task ${TASK_ID} is outside the configured language count (${NUM_LANGUAGES}); exiting."
    exit 0
fi

LANGUAGE="${LANGUAGES[$TASK_ID]}"

echo "Resolved no-constraint eval array task:"
echo "  task_id=${TASK_ID}/${NUM_LANGUAGES}"
echo "  language=${LANGUAGE}"
echo "  dataset_type=${DATASET_TYPE}"
echo "  dataset_folder=${DATASET_FOLDER}"
echo "  batch_size=${BATCH_SIZE}"

bash scripts/eval_all_noconstraint_loop_langs.sh \
    "$DATASET_TYPE" \
    "$DATASET_FOLDER" \
    "$BATCH_SIZE" \
    "$LANGUAGE"
