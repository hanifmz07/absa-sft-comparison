#!/bin/bash
set -euo pipefail

source .venv/bin/activate
export WANDB_MODE=offline

python -m src.main.train \
    --train_json_path dataset/hotel_reviews/eng/mvp_aos/train.json \
    --model_name google/gemma-3-270m \
    --output_dir outputs/smoke/models/hotel_reviews/eng/mvp_aos/seed_42/ \
    --prompt_type mvp \
    --save_strategy no \
    --num_epochs 1 \
    --lr 5e-5 \
    --optimizer adamw_torch \
    --seed 42 \
    --batch_size 1 \
    --gradient_accumulation_steps 1 \
    --eval_strategy no \
    --sample_size 1
