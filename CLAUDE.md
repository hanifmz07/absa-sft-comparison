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

**Note:** `requirements/torch-rocm6.2.txt` and `requirements/torch-default.txt` are not committed to the repo (the `requirements/` directory is empty in a fresh checkout) — obtain or create the appropriate torch pin for the target machine before running the install command above.

## Common Commands

**Smoke test (1 sample, 1 epoch, no W&B):**
```bash
bash scripts/sft_test.sh
```

**Single-language training (local, all 5 seeds):**
```bash
bash scripts/sft_all.sh eng hotel_reviews mvp_aos 4
bash scripts/sft_all_seq2seq.sh eng hotel_reviews mvp_aos 16   # seq2seq variant
```

**Single-language eval (unconstrained only):**
```bash
bash scripts/eval_all_noconstraint.sh eng hotel_reviews mvp_aos 4
bash scripts/eval_all_seq2seq.sh eng hotel_reviews mvp_aos 16  # seq2seq variant
```

**Full eval pipeline (constrained + unconstrained + voting + all post-hoc metrics):**
```bash
bash scripts/eval_all.sh eng hotel_reviews mvp 4
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

**Submit training then chain eval automatically (eval starts only after all SFT tasks succeed):**
```bash
sbatch slurm_submit/submit_sft_then_eval.sh
```

**Submit Slurm eval array (6 jobs, 1 per language):**
```bash
sbatch slurm_submit/submit_eval_noconstraint_array.sh
```

**Submit semantic similarity eval (runs on already-generated inference_results.json):**
```bash
sbatch slurm_submit/submit_semantic_similarity_array.sh          # causal LM (outputs/)
sbatch slurm_submit/submit_semantic_similarity_seq2seq_array.sh  # mt5-base (outputs_seq2seq/)
```

**Aggregate results to CSV after all evals finish:**
```bash
python scripts/extract_mvp_voting_results.py --mode mvp_aos --decoding unconstrained
python scripts/extract_mvp_voting_results.py --mode mvp
python scripts/extract_seq2seq_results.py
```

**Remove duplicate run directories (keep latest per lang/seed/model):**
```bash
python scripts/cleanup_old_runs.py --base outputs/evals/hotel_reviews --dry-run
python scripts/cleanup_old_runs.py --base outputs/evals/hotel_reviews --delete
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

| Path | Model | Trainer | Output dir |
|---|---|---|---|
| `src/main/train.py` / `eval.py` | Causal LM (Gemma-3, Qwen) | TRL `SFTTrainer` | `outputs/` |
| `src/main/train_seq2seq.py` / `eval_seq2seq.py` | Seq2Seq (mT5) | HF `Seq2SeqTrainer` | `outputs_seq2seq/` |

### Prompt format → dataset folder mapping

All scripts infer prompt type from the dataset folder name:

| Folder | Prompt type | Format |
|---|---|---|
| `mvp_aos`, `mvp` | `mvp` | `<input> [A] [O] [S] => [A] aspect [O] opinion [S] sentiment` |
| `gas` | `gas` | `<input> => (aspect, opinion, sentiment); ...` |
| `legoabsa_*`, `indolegoabsa_*` | `legoabsa` | Uses `<\|aspect\|>`, `<\|opinion\|>`, `<\|sentiment\|>` special tokens |

### mvp_aos vs mvp eval pipelines

These two dataset folders differ in how inference is evaluated:

- **`mvp_aos`**: Runs unconstrained decoding only → `inference_results.json` → post-hoc metrics applied directly.
- **`mvp`**: Runs both constrained and unconstrained decoding → `eval_voting.py` aggregates them into `voting_results.json` (majority-vote ensemble) → post-hoc metrics applied to the voting output (`voting_exact_match.json`, `voting_semantic_metrics.json`, etc.).

`eval_all.sh` runs the full mvp pipeline (both decode passes + voting). `eval_all_noconstraint.sh` runs only unconstrained decoding and is sufficient for mvp_aos.

### Training data flow (causal LM)

`train.py` splits each example into:
```python
"prompt":     instance['input'] + " =>"
"completion": " " + instance['target'] + tokenizer.eos_token
```

`SFTConfig` is set with `completion_only_loss=True` — loss is computed **only** on completion tokens. Changing this to `False` dilutes the gradient with a language-modelling objective over the prompt and causes training to fail (see `issue/completion-only-loss-regression.md`).

### Eval output structure

**Causal LM (`outputs/`):**
```
outputs/evals/{dataset_type}/{language}/{dataset_folder}/seed_{n}/
  {run_stamp}_train_model-{model}_lr-{lr}_bs-{bs}_epochs-{e}/
    checkpoint-{step}/
      unconstrained_decoding/
        evaluation_results.json        # precision/recall/f1 ×100
        inference_results.json         # per-sample predictions
        raw_inference_results.json
        exact_match.json               # post-hoc: strict string match
        semantic_metrics.json          # post-hoc: embedding similarity (threshold 0.9)
        instruct_absa.json             # post-hoc: substring overlap
        voting_results.json            # mvp only: ensemble of constrained+unconstrained
        voting_exact_match.json        # post-hoc on voting output
        voting_semantic_metrics.json
```

**Seq2Seq (`outputs_seq2seq/`):** same structure but rooted at `outputs_seq2seq/`.

`evaluation_results.json` stores metrics **already multiplied by 100**, so `f1_aos: 38.5` means 38.5% F1. All post-hoc metric JSON files use the same 0–100 scale.

### Constrained decoding

`src/utils/constrained_decoding.py` implements a `LogitsProcessor` subclass (`BaseConstrainedDecoder`) that restricts generation to tokens present in the input sequence plus sentiment words (`positive`, `negative`, `null`) and special tokens. `MVPConstrainedDecoder`, `GASConstrainedDecoder`, and `LegoABSAConstrainedDecoder` extend it.

### Post-hoc metric scripts

Three metrics computed from existing `inference_results.json` (or `voting_results.json`) without re-running inference:

| Module | Shell wrapper | Output file | Method |
|---|---|---|---|
| `eval_exact_match.py` | `eval_all_exact_match.sh` | `exact_match.json` | Strict string equality |
| `eval_semantic_similarity.py` | `eval_all_semantic_similarity.sh` | `semantic_metrics.json` | Cosine similarity ≥ 0.9 (Qwen3-Embedding-8B) |
| `eval_instruct_absa.py` | `eval_all_instruct_absa.sh` | `instruct_absa.json` | Substring overlap (`pred in gt` or `gt in pred`) |

`src/utils/eval_utils.py` provides the shared `calculate_metrics()` and `calculate_metrics_semantic()` functions used by all three.

### Result aggregation

After all eval jobs finish, two scripts consolidate per-run JSON files into flat CSVs:

- **`scripts/extract_mvp_voting_results.py`** — causal LM results → `outputs/evals/hotel_reviews/`
  - `--mode mvp_aos` → `semantic_metrics_aos.csv`, `exact_match_aos.csv`, `instruct_absa_aos.csv`
  - `--mode mvp` → `semantic_metrics.csv`, `voting_exact_match.csv`, `voting_instruct_absa.csv`
  - `--decoding unconstrained|constrained|all` filters decode mode; auto-picks latest run per model/seed
- **`scripts/extract_seq2seq_results.py`** — seq2seq results → `outputs_seq2seq/evals/hotel_reviews/`

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
- **Duplicate run directories:** multiple Slurm retries produce multiple run-stamp dirs per (lang, seed, model). `scripts/cleanup_old_runs.py` removes all but the latest before running post-hoc metrics or aggregation.
- **Invalid checkpoint batches:** `20260509_*` (lr=5e-05, no EOS), `20260518_150*` and `20260518_16*` (completion_only_loss=False) are known-bad. See `issue/` for details.
- **Checkpoint selection:** `slurm_submit/submit_sft_array.sh` defaults `SAVE_STRATEGY=best`, which keeps only the checkpoint with the lowest `eval_loss`. Combined with `save_total_limit=1`, this ensures the best model is retained across all epochs. The invariant is: the saved checkpoint is selected by validation loss, not epoch number. Pass `SAVE_STRATEGY=epoch` to revert to last-epoch selection (not recommended; observed overfitting: `eval_loss` rose from 0.15 at epoch 1 to 0.44 at epoch 10 while train loss collapsed to ~0.001–0.007).
- **Warmup and grad norm:** `train.py` reads `WARMUP_RATIO` and `MAX_GRAD_NORM` via `os.getenv()` with defaults 0.0 and 1.0 respectively. These must be `export`ed by the calling shell script (not passed as CLI flags) to reach the training process. Both `slurm_submit/submit_sft_array.sh` and `submit_sft_then_eval.sh` export them.
- **Active work tracking:** see `TODO.md` at the repo root for in-progress experiments and pending follow-ups (not just historical bugs like `issue/`).
