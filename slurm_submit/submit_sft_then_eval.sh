#!/bin/bash
#SBATCH --job-name=sft-eval-submit
#SBATCH --partition=mi250x
#SBATCH --ntasks=1
#SBATCH --mem=512MB
#SBATCH --time=00:05:00
#SBATCH --output=logs/out/sft-eval-submit-%j.out
#SBATCH --error=logs/err/sft-eval-submit-%j.err
#
# Meta-job: submits the SFT array, then submits the eval array with an
# afterok dependency so eval only starts after every SFT task succeeds.
#
# The eval script discovers trained models by scanning the same
# outputs/models/ directories that SFT writes into, so it automatically
# evaluates the models produced by this session.
#
# Usage:
#   sbatch slurm_submit/submit_sft_then_eval.sh
#
# Override any hyperparameter via --export:
#   sbatch --export=ALL,BATCH_SIZE=8,LR=1e-4 slurm_submit/submit_sft_then_eval.sh

set -euo pipefail

cd ~/absa-sft-comparison
mkdir -p logs/out logs/err

# ── SFT hyperparams ───────────────────────────────────────────────────────────
export BATCH_SIZE="${BATCH_SIZE:-4}"
export LR="${LR:-1e-5}"
export NUM_EPOCHS="${NUM_EPOCHS:-10}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
export EVAL_STRATEGY="${EVAL_STRATEGY:-epoch}"
export SAVE_STRATEGY="${SAVE_STRATEGY:-epoch}"
export MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.03}"

# ── Eval params ───────────────────────────────────────────────────────────────
export DATASET_TYPE="${DATASET_TYPE:-hotel_reviews}"
export DATASET_FOLDER="${DATASET_FOLDER:-mvp_aos}"
# BATCH_SIZE is reused for eval as well

# ── Submit SFT ────────────────────────────────────────────────────────────────
SFT_JOB_ID=$(sbatch --parsable slurm_submit/submit_sft_array.sh)
echo "Submitted SFT array job: ${SFT_JOB_ID}"

# ── Submit eval (waits for every SFT task to succeed) ────────────────────────
EVAL_JOB_ID=$(sbatch --parsable \
    --dependency=afterok:"${SFT_JOB_ID}" \
    slurm_submit/submit_eval_noconstraint_array.sh)
echo "Submitted eval array job: ${EVAL_JOB_ID} (depends on SFT job ${SFT_JOB_ID})"

echo ""
echo "Monitor with:  squeue -j ${SFT_JOB_ID},${EVAL_JOB_ID}"
echo "Cancel all:    scancel ${SFT_JOB_ID} ${EVAL_JOB_ID}"
