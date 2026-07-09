#!/bin/bash
#SBATCH --job-name=eval-dtype-test
#SBATCH --partition=mi250x
#SBATCH --ntasks=1
#SBATCH --gres=gpu:MI250:1
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/out/eval-dtype-test-%A_%a.out
#SBATCH --error=logs/err/eval-dtype-test-%A_%a.err

# One-off diagnostic: does eval-time load dtype (fp32 vs bf16) explain part of the
# A100-vs-MI250 F1 gap in TODO.md section 3? The pre-2026-06-03 eval.py never passed
# torch_dtype to from_pretrained, which upcasts a bf16-saved checkpoint to fp32 on load.
# Every eval since commit 9b83ce1 explicitly loads in the checkpoint's native bf16.
# array=0 -> torch_dtype=auto (bf16, current default, reproduces the existing 65.76% baseline)
# array=1 -> torch_dtype=float32 (mimics pre-9b83ce1 load behavior)

set -euo pipefail

cd ~/absa-sft-comparison

source .venv/bin/activate

CHECKPOINT="${CHECKPOINT:-outputs/models/hotel_reviews/eng/mvp/seed_123/20260706_065303_train_model-Qwen2.5-0.5B_lr-5e-05_bs-4_epochs-10/checkpoint-7760}"
TEST_JSON="${TEST_JSON:-dataset/hotel_reviews/eng/mvp/test_aug.json}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-300}"
# Set to "sdpa" to isolate the attention-implementation axis (task 0 becomes bf16+sdpa, task 1 fp32+sdpa).
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    DTYPE="auto"
else
    DTYPE="float32"
fi

OUTPUT_DIR="outputs/evals/dtype_test/${DTYPE}_${ATTN_IMPLEMENTATION}"

echo "Checkpoint: $CHECKPOINT"
echo "torch_dtype: $DTYPE"
echo "attn_implementation: $ATTN_IMPLEMENTATION"
echo "Output dir: $OUTPUT_DIR"

python -m src.main.eval \
    --test_json_path "$TEST_JSON" \
    --model_path "$CHECKPOINT" \
    --prompt_type mvp \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --torch_dtype "$DTYPE" \
    --attn_implementation "$ATTN_IMPLEMENTATION" \
    --save_predictions
