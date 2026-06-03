# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## GPU Access

**There is no GPU in the login shell.** Any command that loads a model or runs
inference will fail locally. All training and evaluation must be submitted to
Slurm (partition `mi250x`, AMD MI250 GPUs, ROCm). Never suggest running
`src.main.train` or `src.main.eval` directly in the shell; always use `sbatch`.

## Environment

```bash
uv sync
source .venv/bin/activate
```

This installs dependencies from `pyproject.toml`. PyTorch must then be installed separately to match the hardware:

```bash
# AMD ROCm 6.2 (Slurm cluster — MI250)
pip install -r requirements/torch-rocm6.2.txt && pip install -e . --no-deps

# Default CUDA / CPU
pip install -r requirements/torch-default.txt && pip install -e . --no-deps
```

Copy `.env.example` → `.env` and set `WANDB_API_KEY` and `HF_CACHE_DIR`.

## Common Commands

**Smoke test (1 sample, 1 epoch, no W&B):**
```bash
bash scripts/sft_test.sh
```

**Single-language training (local, all 5 seeds):**
```bash
bash scripts/sft_all.sh eng hotel_reviews mvp_aos 4
```

**Single-language eval (unconstrained, local):**
```bash
bash scripts/eval_all_noconstraint.sh eng hotel_reviews mvp_aos 4
```

**Debug eval with generation inspection:**
```bash
LIMIT_SAMPLES=20 DEBUG_GENERATIONS=1 MAX_NEW_TOKENS=80 FORCE_RERUN=1 \
  bash scripts/eval_all_noconstraint.sh eng hotel_reviews mvp_aos 4
```

**Submit full Slurm training array (30 jobs = 6 langs × 5 seeds):**
```bash
sbatch slurm_submit/submit_sft_array.sh
```

**Submit Slurm eval array (6 jobs, 1 per language):**
```bash
sbatch slurm_submit/submit_eval_noconstraint_array.sh
```

**Run modules directly:**
```bash
python -m src.main.train --help
python -m src.main.eval --help
python -m src.main.train_seq2seq --help
python -m src.main.eval_seq2seq --help
```

## Architecture

### Two model families

| Path | Model | Trainer |
|---|---|---|
| `src/main/train.py` / `eval.py` | Causal LM (Gemma-3, Qwen) | TRL `SFTTrainer` |
| `src/main/train_seq2seq.py` / `eval_seq2seq.py` | Seq2Seq (mT5) | HF `Seq2SeqTrainer` |

### Prompt format → dataset folder mapping

All scripts infer prompt type from the dataset folder name:

| Folder | Prompt type | Format |
|---|---|---|
| `mvp_aos`, `mvp` | `mvp` | `<input> [A] [O] [S] => [A] aspect [O] opinion [S] sentiment` |
| `gas` | `gas` | `<input> => (aspect, opinion, sentiment); ...` |
| `legoabsa_*`, `indolegoabsa_*` | `legoabsa` | Uses `<\|aspect\|>`, `<\|opinion\|>`, `<\|sentiment\|>` special tokens |

### Training data flow (causal LM)

`train.py` splits each example into:
```python
"prompt":     instance['input'] + " =>"
"completion": " " + instance['target'] + tokenizer.eos_token
```

`SFTConfig` is set with `completion_only_loss=True` — loss is computed **only** on completion tokens. Changing this to `False` dilutes the gradient with a language-modelling objective over the prompt and causes training to fail (see `issue/completion-only-loss-regression.md`).

### Eval output structure

```
outputs/evals/{dataset_type}/{language}/{dataset_folder}/seed_{n}/
  {run_stamp}_train_model-{model}_lr-{lr}_bs-{bs}_epochs-{e}/
    checkpoint-{step}/
      unconstrained_decoding/
        evaluation_results.json   # precision/recall/f1 stored ×100 (percent scale)
        inference_results.json    # per-sample predictions
        raw_inference_results.json
```

`evaluation_results.json` stores metrics **already multiplied by 100**, so `f1_aos: 38.5` means 38.5% F1.

### Constrained decoding

`src/utils/constrained_decoding.py` implements a `LogitsProcessor` subclass (`BaseConstrainedDecoder`) that restricts generation to tokens present in the input sequence plus sentiment words (`positive`, `negative`, `null`) and special tokens. `MVPConstrainedDecoder`, `GASConstrainedDecoder`, and `LegoABSAConstrainedDecoder` extend it.

### Post-hoc metric scripts

Three additional metrics can be computed from already-generated `inference_results.json` files without re-running inference:

- `eval_exact_match.py` / `scripts/eval_all_exact_match.sh` — strict string match
- `eval_semantic_similarity.py` / `scripts/eval_all_semantic_similarity.sh` — embedding cosine similarity fallback (threshold 0.9)
- `eval_instruct_absa.py` / `scripts/eval_all_instruct_absa.sh` — substring overlap matching

### Slurm array indexing

`slurm_submit/submit_sft_array.sh` encodes the experiment grid (models × languages × dataset_folders × seeds) into a flat `SLURM_ARRAY_TASK_ID`. The decoding order is seeds → dataset_folders → languages → models (seed varies fastest).

## Key Invariants

- **Seeds:** always `9584 123 2024 31415 777`
- **Standard hyperparameters:** `lr=1e-5`, `batch_size=4`, `gradient_accum=4`, `epochs=10`, `max_grad_norm=1.0`
- **`completion_only_loss=True`** must stay set in `SFTConfig` inside `train.py` — see above.
- **EOS in completions:** `tokenizer.eos_token` is appended to every training completion so the model learns a stop condition.
- **BF16 on ROCm:** ~50% of training steps produce `grad_norm: inf` (BF16 overflow in the 262k-token lm_head); these steps are zeroed by gradient clipping and are wasted but harmless. Training still converges.
- **Tokenizer size vs embedding size mismatch:** the checkpoint tokenizer reports `len=262145` (includes `<image_soft_token>` at id 262144) while the model embedding matrix has 262144 rows. `eval.py` detects this and resizes embeddings before inference.
- **Eval script skipping logic:** `eval_all_noconstraint.sh` skips a model if `unconstrained_decoding/` already exists for it. Use `FORCE_RERUN=1` to override.
- **Invalid checkpoint batches:** `20260509_*` (lr=5e-05, no EOS), `20260518_150*` and `20260518_16*` (completion_only_loss=False) are known-bad. See `issue/` for details.
