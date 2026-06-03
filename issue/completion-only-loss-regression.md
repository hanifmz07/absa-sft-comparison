# Completion-Only Loss Regression

## Symptom

All `20260518_150*` and `20260518_16*` Slurm checkpoints produce mostly empty predictions
or very low F1 despite training running to completion without explicit errors.

| Checkpoint batch | `completion_only_loss` | Step-1 loss | Final loss | Inference |
|---|---|---|---|---|
| `20260515_*` | `True` | ~2.75 | ~0.52 | partial (some non-empty) |
| `20260518_150*` | `False` | ~4.09 | ~2.65 | mostly empty |
| `20260518_16*` | `False` | ~4.09 | ~2.65 | mostly empty |

`20260518_150*` and `20260518_16*` were both trained with the broken flag.

## Root Cause

Between the May 15 runs and the May 18 runs, `completion_only_loss` in
`src/main/train.py` was changed from `True` to `False`:

```python
# Before (20260515_* — working)
completion_only_loss=True,

# After (20260518_* — broken)
# completion_only_loss=True,
completion_only_loss=False,
```

With `completion_only_loss=False`, the loss is computed over **every token** in the
sequence — including the full input prompt such as
`"the service is very friendly . [A] [O] [S] =>"`. The model spends gradient budget
trying to predict the review text from scratch (a language modelling objective) on top
of the ABSA completion objective. This dilutes the training signal and prevents
convergence on the part that matters.

Evidence from `trainer_state.json` (sun, seed_31415):

```
20260515  step 1: loss=2.75  →  step 1560: loss=0.52   ✓ converged
20260518  step 1: loss=4.09  →  step 1560: loss=2.65   ✗ barely moved
```

The `inf` grad norms visible throughout Slurm runs are pre-clip values and are present
in both the working and broken batches; gradient clipping to `max_grad_norm=1.0` keeps
the actual updates bounded in both cases. The high grad norms are not the primary
cause of empty predictions — the `completion_only_loss=False` setting is.

### Why the local CPU debug run still worked

The local validation run (50 samples, 3 epochs, CPU FP32) overfits a tiny sample, so
even with `completion_only_loss=False` it converged well enough to produce reasonable
outputs on 20 test samples. It does **not** represent a reliable signal for full Slurm
runs.

### Secondary issue: `sft_all.sh` had stale LR

`scripts/sft_all.sh` hardcoded `--lr 5e-5`, but all Slurm runs since May 15 have
used `lr=1e-5`. The script was updated to match.

## What Was Fixed

**`src/main/train.py`** — restored `completion_only_loss=True`:

```python
# Before fix
# completion_only_loss=True,
completion_only_loss=False,

# After fix
completion_only_loss=True,
```

**`scripts/sft_all.sh`** — corrected the learning rate:

```bash
# Before
--lr 5e-5 \

# After
--lr 1e-5 \
```

## What Remains

- All checkpoints trained with `completion_only_loss=False` are degraded.
  This covers all `20260518_150*` and `20260518_16*` model directories.
  Do not use them.
- The `20260515_*` checkpoints were trained with the correct flag but without
  EOS in the completion. They produce partial predictions (no reliable stop) and
  serve only as a lower-bound baseline.
- Retrain the full Slurm array using the corrected `train.py`.
- After retraining, verify on one language/seed that:
  - Step-1 loss is in the 2.5–3.0 range (completion tokens only)
  - Final-step loss converges below 1.0
  - Debug generations show `[A]` as first token and EOS stopping before
    `max_new_tokens`

## Validation Checklist

There is no GPU in the login shell. All inference must go through Slurm
(`partition=mi250x`, AMD MI250).

**Step 1 — retrain one seed first (eng, seed 9584):**

```bash
sbatch \
  --array=0-0 \
  --export=ALL,LR=1e-5,NUM_EPOCHS=10,BATCH_SIZE=4,GRADIENT_ACCUMULATION_STEPS=4 \
  slurm_submit/submit_sft_array.sh
```

Array task 0 maps to `eng / mvp_aos / seed_9584`. Check the training log at
`logs/out/sft-array-{jobid}_0.out` and verify:
- Step-1 loss is in the 2.5–3.0 range (completion tokens only, not full sequence)
- Final-step loss converges below 1.0
- W&B project `absa-sft` shows a monotonically decreasing loss curve

**Step 2 — run a debug eval on that one seed:**

```bash
sbatch \
  --array=0-0 \
  --export=ALL,LIMIT_SAMPLES=50,DEBUG_GENERATIONS=1,MAX_NEW_TOKENS=80,FORCE_RERUN=1 \
  slurm_submit/submit_eval_noconstraint_array.sh
```

Array task 0 maps to `eng`. Output lands in
`logs/out/eval-noconstraint-array-{jobid}_0.out`. Good signs:
- First generated token is `[A]` on every sample
- Output stops with EOS well before the 80-token budget
- No `]S]S]S...` loops

**Step 3 — once step 2 looks good, launch the full arrays:**

```bash
sbatch slurm_submit/submit_sft_then_eval.sh
```

This submits all 30 SFT tasks (6 langs × 5 seeds) and chains the 6-task eval array
to start automatically after all training jobs succeed.
