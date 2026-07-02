# TODO

---

## 1. Retrain mT5-base

New Slurm array scripts created — 30 jobs (6 langs × 5 seeds), 5 concurrent:

```bash
sbatch slurm_submit/submit_sft_seq2seq_array.sh
```

After training finishes, run inference then semantic eval then aggregate:

```bash
# Re-run semantic similarity on new checkpoints
sbatch slurm_submit/submit_semantic_similarity_seq2seq_array.sh # [CURRENTLY DOING]

# Aggregate
python scripts/extract_seq2seq_results.py # [TODO]
```

---

## 2. Regenerate Semantic Metrics CSVs

### Why

Two Slurm array jobs were submitted to rerun semantic similarity evaluation using `Qwen/Qwen3-Embedding-8B` embeddings:

- **Job A** (`submit_semantic_similarity_array.sh`) — processes gemma-3-270m and Qwen2.5-0.5B across all 6 languages × {mvp_aos, mvp} in `outputs/`. This also fixes the missing "eng" language in `semantic_metrics_aos.csv` (eng/mvp_aos had inference results but semantic eval had never been run for it).
- **Job B** (`submit_semantic_similarity_seq2seq_array.sh`) — processes mt5-base across all 6 languages × mvp_aos in `outputs_seq2seq/`.

Before submitting, old duplicate run directories were removed using `scripts/cleanup_old_runs.py --delete` so that only the latest run per (lang, seed, model) was kept. The CSVs therefore need to be regenerated from scratch to reflect the cleaned-up, updated results.

### When

After both Slurm jobs finish. Check with:

```bash
squeue -u $USER
```

### Commands to Run

```bash
# 1. Causal LM results (gemma-3-270m + Qwen2.5-0.5B)
#    → outputs/evals/hotel_reviews/semantic_metrics_aos.csv
python scripts/extract_mvp_voting_results.py --mode mvp_aos --decoding unconstrained # [TODO]

# 2. Seq2seq results (mt5-base)
#    → outputs_seq2seq/evals/hotel_reviews/semantic_metrics.csv
python scripts/extract_seq2seq_results.py # [TODO]
```

### Verify

```bash
# eng should now appear in semantic_metrics_aos.csv
grep "^eng" outputs/evals/hotel_reviews/semantic_metrics_aos.csv | head

# Check eng/mvp_aos has semantic_metrics.json files
find outputs/evals/hotel_reviews/eng/mvp_aos -name semantic_metrics.json
```

---

## 3. Diagnose ABSA eval regression (A100 vs MI250)

### Why

Results from before April 2026 (A100) scored much higher than the current
MI250/ROCm rerun for the same config (`eng/mvp/seed_123/Qwen2.5-0.5B`,
unconstrained decoding): F1_aos **87.44%** (old, `checkpoint-7760`,
`lr=5e-05`) vs **60.14%** (current, `checkpoint-7760`, `lr=1e-05`).

Investigation found this is **not primarily a GPU-architecture effect**:

- The comparison is confounded — `lr` changed from `5e-05` (old, considered a
  stale/buggy value) to `1e-05` (current standard) between the two runs.
- Strongest lead: severe overfitting. `trainer_state.json` for the current
  run shows `eval_loss` rising monotonically from **0.150 at epoch 1 to
  0.438 at epoch 10** while train loss collapses to ~0.001–0.007. But
  `SAVE_STRATEGY` defaults to `epoch` (not `best`) in
  `slurm_submit/submit_sft_array.sh`, and `save_total_limit=1` means the
  pipeline always keeps and evaluates the **last, most overfit** epoch
  checkpoint instead of the best one. This would happen on any hardware.
- Ruled out: dtype mismatch between train/eval (checkpoint weights verified
  as real `bfloat16` on disk; `eval.py`'s `torch_dtype="auto"` loads them
  correctly) and ROCm bf16-detection failure (the saved weights prove
  `bf16=True` was correctly selected during training on the MI250).
- Two real code/doc mismatches were found and already fixed in
  `src/main/train.py`, `src/main/eval.py`, `src/main/eval_seq2seq.py`:
  training completions weren't appending `tokenizer.eos_token` (violated a
  documented CLAUDE.md invariant), and eval scripts never called
  `resize_token_embeddings` despite CLAUDE.md claiming they do. Neither was
  visibly breaking Qwen generations today (no empty/looping predictions
  found), but both are now fixed for correctness.

Full writeup: `/share/work/raflyh/.claude/plans/currently-existing-result-before-vectorized-pudding.md`

### Status

Code fixes are done (EOS append, embedding resize ×2). What's left is an
empirical controlled re-run to confirm the overfitting/checkpoint-selection
hypothesis. This is a separate code path (causal LM `train.py`) from the
mT5 seq2seq retrain and semantic-similarity jobs above — **no need to cancel
those**, this can be submitted alongside them.

### Step 1 — Controlled re-run with best-checkpoint selection

Targets just the one cell under investigation (`eng`, `mvp`,
`Qwen2.5-0.5B`, `seed_123` = array index 31 of 90: model_idx 1 × 30 +
language_idx 0 × 5 + seed_idx 1, given `MODELS=(gemma-3-270m,
Qwen2.5-0.5B, mt5-base)`, `LANGUAGES=(eng, jav, indo, mad, min, sun)`,
`SEEDS=(9584, 123, 2024, 31415, 777)` in `slurm_submit/submit_sft_array.sh`).
Keep `lr=1e-05` (current standard) for now — this isolates checkpoint
selection as a variable on its own:

```bash
SAVE_STRATEGY=best sbatch --array=31 slurm_submit/submit_sft_array.sh # [ON PROGRESS]
```

Check progress: `squeue -u $USER`

### Step 2 — Eval and compare

```bash
FORCE_RERUN=1 bash scripts/eval_all_noconstraint.sh eng hotel_reviews mvp 4 # [TODO]
```

Then check:

```bash
cat outputs/evals/hotel_reviews/eng/mvp/seed_123/<new_run_stamp>/checkpoint-*/unconstrained_decoding/evaluation_results.json
```

Compare `f1_aos` against **87.44** (old A100) and **60.14** (current
last-epoch MI250 run). Also confirm best-epoch selection actually happened:

```bash
python3 -c "import json; d=json.load(open('outputs/models/hotel_reviews/eng/mvp/seed_123/<new_run_stamp>/checkpoint-*/trainer_state.json')); print(d['best_model_checkpoint'], d['best_metric'])"
```
(fill in the actual checkpoint path/glob — should point to an early epoch,
likely epoch 1 given the eval_loss curve above, not epoch 10.)

### Step 3 — Interpret and decide follow-up

- **If F1 recovers close to ~87%:** the regression was overfitting /
  checkpoint selection, not hardware. Recommend changing the default
  `SAVE_STRATEGY` to `best` for the full training grid (all 90 jobs) going
  forward, then rerun evals + re-aggregate CSVs (see sections 1–2 above).
- **If F1 improves but a large gap remains:** the remainder is likely the
  `lr` change and/or a residual hardware effect. Run the same cell at the
  old `lr=5e-05` on MI250 for a true apples-to-apples comparison:
  ```bash
  SAVE_STRATEGY=best LR=5e-5 sbatch --array=31 slurm_submit/submit_sft_array.sh
  ```
  Then repeat Step 2's eval/compare for this run too.

### Verify

- `best_model_checkpoint` in `trainer_state.json` points to an epoch earlier
  than the last one (not `checkpoint-7760`).
- `f1_aos` in the new `evaluation_results.json` compared directly against
  87.44 (old) and 60.14 (current last-epoch).
- Spot-check generations look coherent:
  ```bash
  LIMIT_SAMPLES=20 DEBUG_GENERATIONS=1 MAX_NEW_TOKENS=80 FORCE_RERUN=1 \
    bash scripts/eval_all_noconstraint.sh eng hotel_reviews mvp_aos 4
  ```

### Post-training checklist (after array job completes)

**Note:** `SAVE_STRATEGY` default has been fixed to `best` in commit `57f997c`.
The rerun should now save the best checkpoint instead of the overfit epoch-10.

**After training finishes:**

1. **Verify checkpoint selection:**
   ```bash
   find outputs/models/hotel_reviews/eng/mvp/seed_123 -name trainer_state.json -newer outputs/models/hotel_reviews/eng/mvp/seed_123/20260402_091332* | xargs -I {} sh -c 'echo "File: {}"; jq "{best_checkpoint: .best_model_checkpoint, best_epoch: .best_epoch, best_loss: .best_metric}" {}'
   ```
   Should show a checkpoint from epoch 1–3, NOT `checkpoint-7760`.

2. **Run eval on the new checkpoint:**
   ```bash
   FORCE_RERUN=1 bash scripts/eval_all_noconstraint.sh eng hotel_reviews mvp 4
   ```

3. **Compare F1 scores:**
   ```bash
   # Find and display the new run's results
   find outputs/evals/hotel_reviews/eng/mvp/seed_123 -name evaluation_results.json -newer outputs/evals/hotel_reviews/eng/mvp/seed_123/20260402* | xargs -I {} sh -c 'echo "{}:"; jq .f1_aos {}'
   ```
   Expected comparison:
   - Old A100 (epoch unknown, presumably good): **87.44%**
   - Old MI250 (epoch 10, overfit): **60.14%**
   - New MI250 (epoch 1–3, best): **should recover toward 87%**

4. **Decide next steps:**
   - If `≥85%`: checkpoint selection was the cause. Proceed to full-grid retrain (all 90 jobs).
   - If `70–84%`: partial recovery; likely a combination of checkpoint selection + `lr` change. Run at old `lr=5e-05` for comparison.
   - If `<70%`: something else; investigate further before committing to full retrain.

5. **If proceeding with full-grid retrain:**
   ```bash
   sbatch slurm_submit/submit_sft_array.sh  # Uses SAVE_STRATEGY=best now
   ```
   Then after all jobs finish, regenerate eval CSVs:
   ```bash
   python scripts/extract_mvp_voting_results.py --mode mvp_aos --decoding unconstrained
   python scripts/extract_mvp_voting_results.py --mode mvp
   ```

### Does this affect the mT5 retrain (section 1)? Not yet — sequencing decision

Checked whether the mT5/seq2seq pipeline (section 1's `job 2706`, already
queued) has the same checkpoint-selection blind spot. Short answer: it's
actually a **worse** version of the same issue, but we're deliberately not
touching it until the Qwen result (`job 2707_[31]`) comes back.

Findings:
- `scripts/sft_one_seq2seq.sh:80,86` **hardcodes** `--save_strategy "epoch"
  --eval_strategy "no"` — there's no `SAVE_STRATEGY`/`EVAL_STRATEGY` env-var
  override for seq2seq at all (unlike `submit_sft_array.sh` for causal LM).
- Because `eval_strategy="no"`, `train_seq2seq.py` never builds an eval
  dataset (`train_seq2seq.py:87`) and never computes `eval_loss` — so today
  there's **no way to tell** if mT5 checkpoints are overfit, and
  `load_best_model_at_end` (`train_seq2seq.py:134`) could never trigger even
  if `save_strategy=best` were passed, since there's no metric to select on.
- mT5 trains for **20 epochs** (`submit_sft_seq2seq_array.sh:36`, double
  Qwen's 10) with zero validation signal — if the same overfitting pattern
  found for Qwen applies here, this blind spot is bigger, not smaller.
- Existing pre-retrain mT5 baseline (`outputs_seq2seq/evals/hotel_reviews/eng/mvp_aos/seed_*/20260408_..._checkpoint-3120`,
  F1_aos ≈ 63.0–63.8% across 5 seeds) has no A100-vs-MI250 pair the way Qwen
  does, so there's no known "should be ~X%" target to regress against here —
  this is more "is this run trustworthy?" than "did it regress?".

**Decision: don't modify or cancel the mT5 retrain (`job 2706`) yet.** It's
still `PD` (queued, not consuming GPU time), so there's no cost to leaving it
running. Fixing it properly (adding `SAVE_STRATEGY`/`EVAL_STRATEGY`
passthrough to `submit_sft_seq2seq_array.sh` / `sft_one_seq2seq.sh`,
mirroring the causal LM scripts) is a small, mechanical change — but doing it
now, before the Qwen diagnostic confirms the hypothesis, risks
canceling/redoing a 30-job × up to 8h array for nothing.

**Follow-up (only after `job 2707_[31]`'s result confirms checkpoint
selection was the cause):**
1. Add `SAVE_STRATEGY`/`EVAL_STRATEGY` env-var passthrough to
   `slurm_submit/submit_sft_seq2seq_array.sh` and `scripts/sft_one_seq2seq.sh`.
2. Confirm a `dev.json` exists per language/dataset-folder for seq2seq
   (needed for `eval_strategy=epoch` to have something to validate against).
3. Resubmit the mT5 retrain with `SAVE_STRATEGY=best EVAL_STRATEGY=epoch`
   before trusting its numbers in the final comparison.
