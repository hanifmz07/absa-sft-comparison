#!/bin/bash
#SBATCH --job-name=sft-gemma
#SBATCH --partition=mi250x
#SBATCH --ntasks=2
#SBATCH --gres=gpu:MI250:1
#SBATCH --mem=32GB
#SBATCH --time=08:00:00
#SBATCH --output=logs/out-%j.txt
#SBATCH --error=logs/err-%j.txt

cd ~/absa-sft-comparison

bash scripts/sft_all_loop_langs.sh
