#!/bin/bash
#SBATCH --job-name=eval-eng-qwen
#SBATCH --partition=mi250x
#SBATCH --gres=gpu:MI250:1
#SBATCH --mem=32GB
#SBATCH --time=06:00:00
#SBATCH --output=logs/out/eval-eng-qwen-%j.out
#SBATCH --error=logs/err/eval-eng-qwen-%j.err

set -euo pipefail
cd /share/work/raflyh/absa-sft-comparison
mkdir -p logs/out logs/err

bash scripts/eval_all.sh eng hotel_reviews mvp 4
bash scripts/eval_all_noconstraint.sh eng hotel_reviews mvp 4
