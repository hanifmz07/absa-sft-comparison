#!/bin/bash

source .venv/bin/activate
export WANDB_MODE=offline

python -m src.main.train_seq2seq \
  --train_json_path dataset/hotel_reviews/eng/mvp_aos/train.json \
  --model_name google/mt5-small \
  --output_dir outputs_seq2seq/smoke/models/hotel_reviews/eng/mvp_aos/seed_42/ \
  --prompt_type mvp \
  --save_strategy no \
  --num_epochs 1 \
  --lr 2e-4 \
  --optimizer adamw_torch \
  --seed 42 \
  --batch_size 1 \
  --eval_strategy no \
  --sample_size 8