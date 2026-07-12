#!/bin/bash
#SBATCH --job-name=sft-eng-qwen
#SBATCH --partition=mi250x
#SBATCH --gres=gpu:MI250:1
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --output=logs/out/sft-eng-qwen-%j.out
#SBATCH --error=logs/err/sft-eng-qwen-%j.err

set -euo pipefail
cd /share/work/raflyh/absa-sft-comparison
mkdir -p logs/out logs/err

bash scripts/sft_all.sh eng hotel_reviews mvp 4
