#!/bin/bash
#SBATCH --job-name=sft-eval-submit-eng-qwen
#SBATCH --partition=mi250x
#SBATCH --mem=512MB
#SBATCH --time=00:05:00
#SBATCH --output=logs/out/sft-eval-submit-eng-qwen-%j.out
#SBATCH --error=logs/err/sft-eval-submit-eng-qwen-%j.err
#
# Meta-job: submits the English/Qwen SFT job, then submits the eval job
# with an afterok dependency so eval only starts once training succeeds.
#
# Usage:
#   sbatch slurm_submit/submit_sft_then_eval_eng_qwen.sh

set -euo pipefail
cd /share/work/raflyh/absa-sft-comparison
mkdir -p logs/out logs/err

SFT_JOB_ID=$(sbatch --parsable slurm_submit/submit_sft_eng_qwen.sh)
echo "Submitted SFT job: ${SFT_JOB_ID}"

EVAL_JOB_ID=$(sbatch --parsable \
    --dependency=afterok:"${SFT_JOB_ID}" \
    slurm_submit/submit_eval_eng_qwen.sh)
echo "Submitted eval job: ${EVAL_JOB_ID} (depends on SFT job ${SFT_JOB_ID})"

echo ""
echo "Monitor with:  squeue -j ${SFT_JOB_ID},${EVAL_JOB_ID}"
echo "Cancel all:    scancel ${SFT_JOB_ID} ${EVAL_JOB_ID}"
