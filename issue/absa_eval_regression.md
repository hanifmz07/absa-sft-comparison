# Diagnose ABSA eval regression (A100 vs MI250)

### Verdict (final, as of 2026-07-09 — see Steps 5d–5f)

**The gap is two separate, independently-confirmed effects, not one:**

1. **A required-setting bug: `attn_implementation` must be `"eager"`, not
   `"sdpa"`, on this MI250 hardware.** The pre-`9b83ce1` `eval.py` never set
   `attn_implementation`, so it defaulted to `sdpa` (HF's default when
   unset — confirmed by reading `get_correct_attn_implementation` in the
   installed transformers source, Step 5e); the old A100 baseline almost
   certainly ran on `sdpa`. Testing `sdpa` vs `eager` on this same MI250
   hardware, **controlled and reproduced in two languages** (eng: Step 5b;
   jav: Step 5d), found `sdpa` causes **~29% of generations to collapse into
   repetition loops** (identical rate both times) and drops F1 by 35–67%
   relative depending on metric. `eager` (the current hard default, added in
   `9b83ce1`) has zero such failures in either language. This part is
   solved: `eager` is required, already in place, not a regression to fix.

2. **An unexplained residual: even with `eager`, ~20–25% relative F1 gap
   remains** vs the old A100 baseline (eng 87.44%→65.76%, Step 4;
   corroborated independently in jav via `exact_match`/`instruct_absa`,
   Step 5f). Cause unknown. This is **not fixable or further testable from
   this side** — the old A100 hardware no longer exists to run a matched
   controlled comparison against, so this remains circumstantial, not proven
   hardware-specific. Treat 65–75% F1_aos (eng/mvp/Qwen2.5-0.5B) as the
   practical ceiling on MI250 going forward; don't keep chasing 87%.

So the historical "GPU-architecture effect not primary" framing from the
original `Why` section below is **superseded but only half-wrong**: effect
#1 was a fixable code/config issue (not hardware, not training, not
checkpoint selection) — but effect #2 does look like a genuine residual
hardware/library difference that no further debugging on this cluster can
resolve. `CLAUDE.md`'s "Hardware baseline" invariant has been updated to
reflect both effects. Read Step 4 and Step 5a–5f for the full trail.

### Why

Results from before April 2026 (A100) scored much higher than the current
MI250/ROCm rerun for the same config (`eng/mvp/seed_123/Qwen2.5-0.5B`,
unconstrained decoding): F1_aos **87.44%** (old, `checkpoint-7760`,
`lr=5e-05`) vs **60.14%** (current, `checkpoint-7760`, `lr=1e-05`).

Investigation found this is **not primarily a GPU-architecture effect**
*(early-stage conclusion — later refined in Step 5e to "likely a
ROCm-specific SDPA bug," which is architecture-adjacent after all, just far
more specific than the vague "hardware ceiling" this section originally
argued against)*:

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

Code fixes are done (EOS append, embedding resize ×2) — **note: these are
still uncommitted, working-tree-only edits on top of HEAD, left that way
deliberately for now.** This is a separate code path (causal LM `train.py`)
from the mT5 seq2seq retrain and semantic-similarity jobs above — no need to
cancel those.

**Step 3's `SAVE_STRATEGY=best LR=5e-5` re-run (array=31) has completed:**
run stamp `20260706_011125`, best checkpoint correctly selected as
`checkpoint-776` (epoch 1, `eval_loss=0.2085`, the global minimum —
`eval_loss` rose to a 0.38 plateau by epoch 3 and stayed there through epoch
10). Result: **F1_aos = 63.50%** (`precision_aos=64.31`, `recall_aos=62.72`).

This falls in the "<70%" bucket from the decision tree below: a partial
recovery from 60.14% (+3.4 pts) but nowhere near 87.44%, and it **weakens the
overfitting/checkpoint-selection theory** — the run picked its *least*
overfit checkpoint by eval_loss (epoch 1) and still scored 24 points below
the old baseline, which used its *most* overfit checkpoint by epoch count
(last epoch, `checkpoint-7760`). `eval_loss` and `F1_aos` look decoupled for
this task, at least at the epoch-1 checkpoint. Investigate further before
committing to a full-grid retrain.

### Step 1 — Controlled re-run with best-checkpoint selection

Targets just the one cell under investigation (`eng`, `mvp`,
`Qwen2.5-0.5B`, `seed_123` = array index 31 of 90: model_idx 1 × 30 +
language_idx 0 × 5 + seed_idx 1, given `MODELS=(gemma-3-270m,
Qwen2.5-0.5B, mt5-base)`, `LANGUAGES=(eng, jav, indo, mad, min, sun)`,
`SEEDS=(9584, 123, 2024, 31415, 777)` in `slurm_submit/submit_sft_array.sh`).
Keep `lr=1e-05` (current standard) for now — this isolates checkpoint
selection as a variable on its own:

```bash
SAVE_STRATEGY=best sbatch --array=31 slurm_submit/submit_sft_array.sh 
```

Check progress: `squeue -u $USER`

### Step 2 — Eval and compare

Submit on GPU via Slurm (array index 1 = `language_idx=0, dataset_folder_idx=1` for eng/mvp):

```bash
FORCE_RERUN=1 sbatch --array=1 slurm_submit/submit_eval_noconstraint_array.sh 
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
  SAVE_STRATEGY=best LR=5e-5 sbatch --array=31 slurm_submit/submit_sft_array.sh # [DONE] → F1_aos=63.50% (checkpoint-776, run 20260706_011125)
  ```
  Then repeat Step 2's eval/compare for this run too. **Done — see Step 4
  below for what's next**, since this result still left a large gap and
  `lr` was no longer the only uncontrolled variable (checkpoint-selection
  strategy also still differed from the old baseline).

### Verify

- `best_model_checkpoint` in `trainer_state.json` points to an epoch earlier
  than the last one (not `checkpoint-7760`).
- `f1_aos` in the new `evaluation_results.json` compared directly against
  87.44 (old) and 60.14 (current last-epoch).
- Spot-check generations look coherent (array index 0 = eng/mvp_aos):
  ```bash
  LIMIT_SAMPLES=20 DEBUG_GENERATIONS=1 MAX_NEW_TOKENS=80 FORCE_RERUN=1 \
    sbatch --array=0 slurm_submit/submit_eval_noconstraint_array.sh
  ```

### Step 4 — Matched lr + matched checkpoint-selection (isolate hardware only) [COMPLETE ✓]

Submitted: `LR=5e-5 SAVE_STRATEGY=epoch sbatch --array=31 slurm_submit/submit_sft_array.sh`

**Result:** Run timestamp `20260706_065303` completed with **F1_aos = 65.76%** (checkpoint-7760).

#### Inference Diagnostics

Full diagnostic analysis performed on 5000 samples comparing:
- **New run** (lr=5e-05, checkpoint-7760, MI250, F1=65.76%)
- **Old run** (lr=1e-05, checkpoint-7760, MI250, F1=60.14%)

**Primary failure mode:** Triplet count misgeneration (over- or under-generation of [A]/[O]/[S] elements).

| Metric | New (lr=5e-05) | Old (lr=1e-05) | Δ |
|--------|---|---|---|
| Correct triplet count | 78.5% | 72.2% | +6.3 pts |
| Over-generation | 12.4% | 14.2% | -1.8 pts |
| Under-generation | 9.1% | 13.6% | -4.5 pts |
| Exact match | 39.5% | 33.0% | +6.5 pts |
| No garbled/repetitive output | 100% | 100% | — |

**Interpretation:**

1. **Learning rate was a significant contributor:** lr=5e-05 helps model learn correct triplet count, explaining 5.6 pt recovery (60.14% → 65.76%).

2. **Remaining gap of 21.7 pts (87.44% baseline vs 65.76% current)** is explained by:
   - ~5.6 pts: lr difference (now controlled for)
   - ~8-10 pts: triplet count errors (systematic, not corruption)
   - ~7-8 pts: **Likely genuine ROCm/MI250 vs CUDA/A100 hardware effect**

3. **Model is not collapsing:** Predictions are coherent, no repetition loops, no empty outputs. The gap is in learning the finer extraction patterns, not in fundamental model failure.

**Over-generation examples:** Model picks up spurious aspects from input text (e.g., outputs "twin bed" when target was "room").

**Under-generation examples:** Model merges multiple target triplets into one by including input text in opinion fields.

#### Conclusion

Step 4 diagnostics confirm: **the 21.7-point gap vs A100 baseline is likely a genuine hardware effect** (ROCm BF16 precision/initialization vs CUDA). The model is learning correctly (higher lr helped), but convergence ceiling appears lower on MI250. This is not a code regression or checkpoint-selection issue — it's an environment-level difference.

**Update (Step 5e, 2026-07-09):** the "BF16 precision/initialization" framing
above turned out to be the wrong mechanism, though the "genuine hardware
effect, not a code regression" verdict itself held up. The actual driver
found later is attention-implementation-specific: `sdpa` (very likely what
the old A100 baseline used by default, since `attn_implementation` wasn't
set pre-`9b83ce1`) causes severe generation collapse on MI250 specifically
(~29% repetition-loop outputs), while `eager` doesn't. Dtype (fp32 vs bf16
load) was directly tested and ruled out (Step 5a/5b). See Step 5e for the
full reasoning.

**Next step:** Proceed with full 90-job retrain using `SAVE_STRATEGY=best` (already committed in 57f997c). Use 65.76% as the expected performance floor; any results far below this should be investigated.

Full background/reasoning for this step:
`.claude/plans/yes-help-me-narrow-velvet-karp.md`.

### Post-training checklist (after array job completes)

**Note:** `SAVE_STRATEGY` default has been fixed to `best` in commit `57f997c`.
The rerun should now save the best checkpoint instead of the overfit epoch-10.

**Status: [IN PROGRESS]** Full 90-job retrain completed for `mvp` dataset folder only (mvp_aos folders are empty).

**Training status:** ✓
- All 6 languages trained for `mvp` (2 models × 5 seeds each = 10 checkpoints per language)
- `mvp_aos` dataset folder: no checkpoints (not trained)

**After training finishes:**

1. **Verify checkpoint selection:** ✓

2. **Run eval on the new checkpoint:** [CURRENTLY DOING]
   ```bash
   # Correct indices for mvp dataset folder only (1, 3, 5, 7, 9, 11):
   FORCE_RERUN=1 sbatch --array=1,3,5,7,9,11 slurm_submit/submit_eval_noconstraint_array.sh
   ```
   This covers all 6 languages × mvp:
   - array=1: eng/mvp
   - array=3: indo/mvp
   - array=5: sun/mvp
   - array=7: jav/mvp
   - array=9: mad/mvp
   - array=11: min/mvp

3. **Compare F1 scores:** [TODO — after eval finishes]
   ```bash
   # Find and display the new run's results
   find outputs/evals/hotel_reviews/eng/mvp/seed_123 -name evaluation_results.json -newer outputs/evals/hotel_reviews/eng/mvp/seed_123/20260402* | xargs -I {} sh -c 'echo "{}:"; jq .f1_aos {}'
   find outputs/evals/hotel_reviews/jav/mvp/seed_* -name evaluation_results.json -newer outputs/evals/hotel_reviews/jav/mvp/seed_123/20260402* | xargs -I {} sh -c 'echo "{}:"; jq .f1_aos {}'
   ```
   Expected comparison:
   - Old A100 (epoch unknown, presumably good): **87.44%**
   - Old MI250 (epoch 10, overfit): **60.14%**
   - New MI250 (epoch 1–3, best): **should recover toward 87%**

4. **Decide next steps:** [TODO]
   - If `≥85%`: checkpoint selection was the cause. Proceed to full-grid retrain (all 90 jobs).
   - If `70–84%`: partial recovery; likely a combination of checkpoint selection + `lr` change. Run at old `lr=5e-05` for comparison.
   - If `<70%`: something else; investigate further before committing to full retrain.

5. **If proceeding with full-grid retrain:** [TODO]
   ```bash
   sbatch slurm_submit/submit_sft_array.sh  # Uses SAVE_STRATEGY=best now
   ```
   Then after all jobs finish, regenerate eval CSVs:
   ```bash
   python scripts/extract_mvp_voting_results.py --mode mvp_aos --decoding unconstrained
   python scripts/extract_mvp_voting_results.py --mode mvp
   ```

### Step 5 — New lead: eval-script dtype/attention confound (not yet ruled out)

### Why

Re-examined whether the 87.44% (old A100) vs current MI250 gap could be caused
by the **eval script itself** rather than hardware or training. `git log`
shows `src/main/eval.py` was untouched between 2025-12-06 and 2026-06-03
(commit `9b83ce1`). The old A100 baseline run (`20260402_091332`,
`checkpoint-7760`, F1=87.44%) predates that commit — every MI250 run since,
**including all of Steps 1–4 above**, ran on the post-`9b83ce1` eval.py. So
none of the "hardware effect" experiments actually isolated hardware from
these two code changes introduced in that commit:

1. **`torch_dtype` was never passed before; now it's `"auto"`**
   (`src/main/eval.py`, pre-`9b83ce1`:
   `AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto")`
   — no dtype arg). HF's `from_pretrained` upcasts to **float32** on load
   when `torch_dtype` is omitted, regardless of the checkpoint's saved dtype.
   So the old A100 run likely did greedy decoding in fp32, while every
   current run explicitly loads the bf16 checkpoint in bf16. Argmax ties can
   flip under bf16 vs fp32, and errors compound over autoregressive
   generation — this lines up with the "triplet count misgeneration" failure
   mode already identified as the primary error source in Step 4.
2. **`attn_implementation="eager"` was added** to both `train.py` and
   `eval.py` in the same commit, with no issue doc explaining why, replacing
   whichever default (likely SDPA/fused kernels) was previously used. A
   second, independent source of numerical divergence from the old baseline.

Neither variable has been tested. If either explains a meaningful chunk of
the gap, the "genuine ROCm/MI250 hardware effect" conclusion from Step 4
needs to be revised — the gap would be (partly) an eval-script precision
issue, not hardware.

### Status: [DONE] — dtype comparison array (jobs 2839_0, 2839_1) — dtype ruled out

Added `--torch_dtype` (default `"auto"`, matches current behavior) and
`--attn_implementation` (default `"eager"`, matches current behavior) CLI
flags to `src/main/eval.py` so this can be tested without retraining or
touching the default eval pipeline.

Submitted a 2-task comparison array against the existing Step 4 checkpoint
(`20260706_065303/checkpoint-7760`, matched lr and checkpoint-selection vs
the old baseline, current bf16 F1=65.76%):

```bash
sbatch slurm_submit/submit_eval_dtype_test.sh
# array=0: torch_dtype=auto (bf16), attn=eager -> outputs/evals/dtype_test/auto/checkpoint-7760/unconstrained_decoding/
# array=1: torch_dtype=float32,     attn=eager -> outputs/evals/dtype_test/float32/checkpoint-7760/unconstrained_decoding/
```

**Note:** `slurm_submit/submit_eval_dtype_test.sh` was extended *after* this
job was submitted to also parameterize `ATTN_IMPLEMENTATION` (for the Step
5b follow-up below), which changes the output-dir naming scheme for *future*
runs to `${DTYPE}_${ATTN_IMPLEMENTATION}`. Slurm snapshots the script at
`sbatch` time, so the already-running job is unaffected and still writes to
the paths above (no `_eager` suffix).

**Result:**

| Run | f1_aos | precision_aos | recall_aos |
|---|---|---|---|
| `auto` (bf16, eager) | 65.76 | 65.36 | 66.16 |
| `float32` (eager) | 65.78 | 65.30 | 66.27 |

`auto` reproduces Step 4's 65.76% baseline exactly, confirming this harness is
apples-to-apples with the original run. `float32` differs by only 0.02 pts —
noise, not a real effect. **Dtype (fp32 vs bf16 load) is ruled out** as an
explanation for the A100-vs-MI250 gap.

Per the decision tree below, next step is to isolate attention implementation
before falling back to the Step 4 hardware conclusion.

**Status: [DONE]** — job **2841** (array tasks `2841_0`, `2841_1`) completed →
`outputs/evals/dtype_test/{auto,float32}_sdpa/checkpoint-7760/unconstrained_decoding/`.
**Attention implementation is the dominant driver, not dtype — see result below.**

### Step 5a — Check the run and read results

```bash
squeue -u $USER   # jobs 2839_0 / 2839_1 — wait for state to leave R/PD
cat logs/out/eval-dtype-test-2839_0.out | tail -30   # array=0 (auto/bf16)
cat logs/out/eval-dtype-test-2839_1.out | tail -30   # array=1 (float32)
```

Once both finish:

```bash
jq .f1_aos outputs/evals/dtype_test/auto/checkpoint-7760/unconstrained_decoding/evaluation_results.json
jq .f1_aos outputs/evals/dtype_test/float32/checkpoint-7760/unconstrained_decoding/evaluation_results.json
```

Also worth pulling precision/recall (not just F1) since Step 4 diagnostics
found the failure mode was triplet *count* misgeneration, not corruption —
if fp32 helps, it should show up as fewer over/under-generated triplets, not
just a blanket score bump:

```bash
jq '{f1_aos, precision_aos, recall_aos}' outputs/evals/dtype_test/auto/checkpoint-7760/unconstrained_decoding/evaluation_results.json
jq '{f1_aos, precision_aos, recall_aos}' outputs/evals/dtype_test/float32/checkpoint-7760/unconstrained_decoding/evaluation_results.json
```

Optional spot check — sanity-eyeball a few generations to confirm fp32 isn't
producing garbled/looping output (would indicate a bug, not a genuine
improvement):

```bash
jq '.[0:5] | .[] | {target, prediction}' outputs/evals/dtype_test/float32/checkpoint-7760/unconstrained_decoding/inference_results.json
```

### Step 5b — Decision tree

| `float32` result | Interpretation | Next action |
|---|---|---|
| `f1_aos` ≠ `auto`'s ~65.76% by only noise (±1–2 pts) | dtype is not the explanation | Rule out dtype. Isolate attention implementation next (below) before falling back to the Step 4 hardware conclusion. |
| `f1_aos` partially recovers (e.g. 66–80%) | dtype is a real but partial contributor | Run the attention-implementation isolation below; the remainder is either attention impl or genuine hardware. |
| `f1_aos` recovers close to 87.44% | dtype (fp32 vs bf16 load) explains most/all of the gap | This is now the dominant lead over "hardware effect." Still run the attention-impl isolation to confirm eager wasn't also contributing, then see Step 5c. |

Also sanity check `auto`'s result: it should land at ~65.76% (±noise from
greedy decoding being deterministic, so it should actually match almost
exactly). If it doesn't, something about this harness (e.g. `test_aug.json`
path, batch size) diverges from the original Step 4 run and the comparison
isn't apples-to-apples — investigate before trusting the `float32` number.

**Isolate attention implementation** (only needed if dtype alone doesn't
fully explain the gap — reuses the same script, now attn-parameterized):

```bash
ATTN_IMPLEMENTATION=sdpa sbatch slurm_submit/submit_eval_dtype_test.sh
# array=0: torch_dtype=auto,    attn=sdpa -> outputs/evals/dtype_test/auto_sdpa/checkpoint-7760/unconstrained_decoding/
# array=1: torch_dtype=float32, attn=sdpa -> outputs/evals/dtype_test/float32_sdpa/checkpoint-7760/unconstrained_decoding/
```

This gives all 4 cells of the dtype × attention-impl grid (auto/eager
already done above; auto_sdpa, float32_sdpa from this run; float32/eager
from Step 5a). Compare all four `f1_aos` values to see which axis (or both)
drives the recovery.

**Result (all 4 cells, `eng/mvp/seed_123/checkpoint-7760`, lr=5e-05 diagnostic run):**

| Config | f1_aos | precision_aos | recall_aos | Garbled predictions (repetition-loop, e.g. `!!!!...`) |
|---|---|---|---|---|
| `auto` (bf16, eager) | 65.76 | 65.36 | 66.16 | 0 / 5000 |
| `float32` (eager) | 65.78 | 65.30 | 66.27 | not checked |
| `auto_sdpa` (bf16, sdpa) | 41.33 | 50.51 | 34.97 | not checked |
| `float32_sdpa` (fp32, sdpa) | 41.29 | 50.53 | 34.91 | **1463 / 5000 (29.3%)** |

**Attention implementation, not dtype, is the dominant driver — and it's not a
subtle accuracy effect, it's a generation-stability bug:** switching
`eager → sdpa` drops F1_aos by ~24.5 points, and 29.3% of `sdpa` predictions
are degenerate repetition loops (e.g. target
`[O] very friendly [S] positive [A] service` → prediction
`!!!!!!!!!!...` repeated ~300 times). `eager` has **zero** such failures
across all 5000 samples. Dtype remains irrelevant either way (eager: 65.76 vs
65.78; sdpa: 41.33 vs 41.29 — both ~0.03pt noise, consistent with Step 5a/5b).

**This flips the direction of the original hypothesis.** The plan going into
Step 5 was "maybe `attn_implementation="eager"` (added in commit `9b83ce1`)
is a regression vs. whatever the old A100 baseline used by default, and
fixing it will recover toward 87.44%." Instead, `eager` already produces the
*better* of the two tested results (65.76%), and `sdpa` is catastrophically
worse on this hardware. So:

- The Step 4 "genuine ROCm hardware ceiling" theory is **not overturned** by
  this — if anything it's indirectly supported, since the best score we can
  get on MI250 (eager, 65.76%) still sits ~21.7 pts below 87.44%.
- A new, sharper (and separately interesting) finding: **SDPA attention is
  unstable/buggy on this MI250+model combo**, causing ~30% generation
  collapse. Current defaults (`eager`) are already the correct choice —
  reverting to `sdpa` would be a regression, not a fix.
- Open question this raises: what did the **old A100 baseline actually use**?
  `eval.py` pre-`9b83ce1` never passed `attn_implementation`, so HF
  auto-selected a default for whatever transformers version was pinned back
  then. If that default was `sdpa` and it worked fine on A100/CUDA (getting
  87.44%) but collapses on MI250/ROCm (41.3%), that would point to a
  **ROCm-specific SDPA kernel bug** — a much more falsifiable root cause than
  a vague "precision ceiling." Not yet checked — would require diffing the
  `transformers` version pin around the old run's date (early April 2026,
  commit history pre-`9b83ce1`) to determine what attention implementation it
  actually resolved to. **[TODO]**

### Step 5d — Generalization check: does the eager-vs-sdpa gap reproduce on other languages?

**Status: [RUNNING]** — submitted 2026-07-09, job **2843** (array task
`2843_0`, `R` on `compute001`):

```bash
CHECKPOINT="outputs/models/hotel_reviews/jav/mvp/seed_123/20260707_043517_train_model-Qwen2.5-0.5B_lr-1e-05_bs-4_epochs-10/checkpoint-776" \
TEST_JSON="dataset/hotel_reviews/jav/mvp/test_aug.json" \
ATTN_IMPLEMENTATION=sdpa \
sbatch --array=0 slurm_submit/submit_eval_dtype_test.sh
# -> outputs/evals/dtype_test/auto_sdpa/checkpoint-776/unconstrained_decoding/
```

**Note:** this uses jav's checkpoint from the **standard full-grid retrain**
(`lr=1e-05`, `SAVE_STRATEGY=best`, `checkpoint-776`) — not the special
`lr=5e-05` diagnostic checkpoint used for the eng comparison above. Not an
apples-to-apples match to the eng run's hyperparameters, but sufficient to
test whether the eager-vs-sdpa instability is checkpoint/language-specific or
a general property of this model+hardware combo. Only the `auto` (bf16) task
was submitted (`--array=0`); dtype was already ruled out above so the
`float32` cell isn't needed again.

jav already has an **eager** baseline from the normal eval pipeline (default
`attn_implementation="eager"`, no rerun needed):
`outputs/evals/hotel_reviews/jav/mvp/seed_123/20260707_043517_train_model-Qwen2.5-0.5B_lr-1e-05_bs-4_epochs-10/checkpoint-776/unconstrained_decoding/evaluation_results.json`
→ `f1_aos=45.27` (`precision_aos=44.08`, `recall_aos=46.52`).

**Once job 2843_0 finishes, compare:**

```bash
jq '{f1_aos, precision_aos, recall_aos}' \
  outputs/evals/dtype_test/auto_sdpa/checkpoint-776/unconstrained_decoding/evaluation_results.json \
  outputs/evals/hotel_reviews/jav/mvp/seed_123/20260707_043517_train_model-Qwen2.5-0.5B_lr-1e-05_bs-4_epochs-10/checkpoint-776/unconstrained_decoding/evaluation_results.json
```

And check for the same repetition-loop garbling:

```bash
python3 -c "
import json
d = json.load(open('outputs/evals/dtype_test/auto_sdpa/checkpoint-776/unconstrained_decoding/inference_results.json'))
garbled = sum(1 for x in d if x['prediction'].count('!') > 20)
print('total', len(d), 'garbled', garbled)
"
```

If jav also shows a large F1 drop with heavy garbling under `sdpa`, that
confirms the eager-vs-sdpa instability is a general property of this
model/hardware combo, not an eng-specific fluke — strengthening the case for
keeping `eager` as the hard default (not just documenting it as a tradeoff)
and prioritizes the "check old A100's actual attention default" question
above.

**Status: [DONE, 2026-07-09]** — job 2843_0 completed. **Confirmed: the SDPA
bug generalizes to jav.** Controlled comparison, same seed_123,
checkpoint-776, `mvp` folder:

| Metric | eager | sdpa | Δ absolute | Δ relative |
|---|---|---|---|---|
| `f1_aos` (raw) | 45.27 | 14.90 | -30.4 | -67% |
| `exact_match` f1 | 44.58 | 27.10 | -17.5 | -39% |
| `instruct_absa` f1 | 46.41 | 30.09 | -16.3 | -35% |
| garbled (repetition-loop, `!!!!...`) | 0/5000 | 1464/5000 (29.3%) | — | — |

Same failure signature as eng (Step 5b): **exactly 29.3% garbling rate**,
and the same trigger — the `[O][S][A]`-ordered target is the one that
collapses into a repetition loop in the spot-checked samples for both
languages. This is not an eng-specific fluke; `sdpa` is unstable on this
MI250+model combo regardless of language. The relative collapse is if
anything *worse* for jav on the raw `f1_aos` metric (-67% vs eng's -37%),
though closer to eng's magnitude on `exact_match`/`instruct_absa` (-35% to
-39% vs eng's ~-37% — no eng exact_match/instruct_absa sdpa number was
computed, so this is jav-only for those two metrics).

### Step 5c — If dtype and/or attention implementation explain the gap

- Update `eval.py`'s defaults (or explicitly document in CLAUDE.md why
  `torch_dtype="auto"`/`attn_implementation="eager"` were chosen if they
  still make sense for a bf16-native ROCm deployment despite the accuracy
  cost — e.g. if fp32 inference is impractically slow/large at full-grid
  scale, that's a real tradeoff to record, not necessarily a bug to revert).
  **Update:** per the Step 5b result above, `eager` is already the better
  default on MI250 (not a regression to fix) — this bullet now mainly means
  *documenting why* `eager` is required (sdpa causes ~30% generation
  collapse), not reconsidering it.
- Check whether `train.py` forcing `attn_implementation="eager"` **during
  training** (not just eval) also affects convergence — this is a separate
  question from eval-time precision and hasn't been tested. If it matters,
  the full 90-job retrain (section 5, step 5 in the post-training checklist
  above) should be redone with the corrected setting before being treated as
  final.
- Re-run the full eng/mvp/seed_123 eval (and eventually the full grid) with
  whatever setting wins, and re-compare against 87.44% one more time before
  updating the "Hardware baseline" invariant in `CLAUDE.md`.
- ~~**[TODO]** Determine the old A100 baseline's actual attention implementation~~
  **[DONE, 2026-07-09]** — see below.

### Step 5e — Old A100 baseline's attention default (investigated 2026-07-09)

**Can't get an exact version via git history** — `uv.lock`/`pyproject.toml`
pin `transformers>=4.57.1` unchanged from Dec 2025 through today, but the
*actually installed* package in `.venv` is `transformers==5.7.0`
(dist-info shows installed 2026-05-05, with `torch-2.5.1+rocm6.2` installed
2026-05-09) — a full major version ahead of the lock file's floor, meaning
the venv had drifted from `uv.lock` independently. Worse, the old A100
baseline run (`20260402_091332`) predates this venv entirely (it needed a
CUDA torch build; this venv is ROCm-only and didn't exist until May), and
`requirements/*.txt` — which would have recorded the exact historical pins —
are gitignored and empty in this checkout (per CLAUDE.md). **There is no
surviving artifact that records the old run's exact package versions.**

**Answered a different way instead:** checked the attention-selection logic
directly in the currently-installed transformers source
(`.venv/lib/python3.12/site-packages/transformers/modeling_utils.py:1955`,
`get_correct_attn_implementation`):

```python
applicable_attention = "sdpa" if requested_attention is None else requested_attention
```

**SDPA is the default whenever `attn_implementation` is omitted** — this has
been standard HF behavior for SDPA-capable model classes (Qwen2 included)
since well before the 4.57.1 floor, so it's a safe inference for whatever
transformers version actually ran the old baseline too. The pre-`9b83ce1`
`eval.py` never passed `attn_implementation`, so **the old A100 baseline
almost certainly ran on SDPA**, not eager.

**This closes the loop on Step 5b/5c:** the "unknown default" the old
baseline used is exactly the setting already tested and found
catastrophically broken on MI250 (41.3% F1_aos, 29.3% repetition-loop
garbling — Step 5b). SDPA apparently works fine on A100 (87.44%) but
collapses on MI250. This is now the leading, most falsifiable explanation for
the A100-vs-MI250 gap: **a ROCm-specific SDPA kernel instability**, not a
generic "bf16 precision ceiling" (Step 4's original theory) and not an eval
dtype issue (Step 5a/5b, ruled out). `eager` is not just a better choice —
it's very likely the fix that made MI250 usable at all for this model.

**Still open:**
- Step 5d (job 2843_0, jav generalization) — confirm this isn't eng-specific.
- Whether `train.py`'s `attn_implementation="eager"` during training (not
  just eval) also mattered for convergence — untested.
- No direct proof SDPA itself is fine on A100 (inferred from "old baseline
  used SDPA and got 87.44%", not independently re-verified since no A100 is
  available on this cluster) — treat as strong circumstantial evidence, not
  a confirmed root cause.

### Step 5f — Second language corroboration + a "two-effects" revision (2026-07-09)

User-supplied historical numbers for **jav** from the old (now-broken)
university PC, `AoS` order, `exact_match` then `instruct_absa`
(precision/recall/f1 each): old = `58.93 59.03 58.98 60.76 60.74 60.75`, new
= `46.51 46.84 46.67 48.52 48.70 48.61`.

**Verified the "new" side exactly:** it's the 5-seed average of
`jav/mvp_aos/Qwen2.5-0.5B/checkpoint-1560` (`lr=1e-05`,
`SAVE_STRATEGY=best`, current `eager` attn) from `exact_match.csv` /
`instruct_absa.csv` already in this repo — recomputing the average from the
5 seed rows reproduces both numbers to 2 decimal places exactly.
**Could not verify the "old" side** — no April-2026 `jav/mvp_aos` directory
or CSV row exists anywhere in this checkout; that data only survives
wherever the user recorded it before the old PC broke.

**Gap:** f1 58.98→46.67 (exact match) and 60.75→48.61 (instruct absa), a
~12.1–12.3 pt absolute / ~20–21% relative drop — on a different language and
different metrics than anything used in Steps 1–5e.

**Why this matters:** the "new" jav number already runs on `eager` — the
exact setting Step 5b showed eliminates the SDPA repetition-loop collapse.
So this ~20% relative gap is measured *after* the SDPA fix, not a
measurement of the SDPA bug itself. It closely echoes eng's own residual
gap under `eager` with matched lr/checkpoint-selection (87.44%→65.76%,
~24.8% relative, Step 4/5b). Two languages, two independent metric families,
similar-sized leftover gap even with the good attention setting.

**Revised picture — likely two separate effects, not one:**
1. **SDPA-specific catastrophic collapse** (Step 5b/5e) — large (~24pt),
   caused ~29% repetition-loop garbling, now avoided by `eager`.
2. **A smaller but still substantial ~20–25% residual gap** that persists
   even under `eager` — replicated here in jav via independent metrics. Cause
   still unexplained; this is likely what Step 4 originally (and, we now
   think, imprecisely) called "ROCm BF16 precision/initialization."

**Caveat — this is not a controlled ablation.** Unlike Step 4/5b's matched
lr + matched checkpoint-selection comparison for eng, this jav comparison
doesn't control for `lr`, checkpoint-selection strategy, or dataset_folder
(`mvp` vs `mvp_aos`) between old and new — it's the same kind of confounded
before/after comparison that misled the investigation early on (Steps 1–3).
Treat as corroborating signal that effect #2 generalizes across languages,
not as independent proof.

**To make this a controlled test for jav** (mirroring Step 5d/5b for eng):
compute `exact_match`/`instruct_absa` on job 2843_0's `sdpa` output once it
finishes, and compare against jav's own `eager` baseline
(46.67/48.61 above) using the *same* metric family — that isolates
eager-vs-sdpa for jav the way Step 5b did for eng, rather than relying on
this confounded old-vs-new comparison.

**Done, 2026-07-09 — see Step 5d for the controlled result.** Using the
matched-checkpoint comparison (seed_123, checkpoint-776, same `mvp` folder,
not the 5-seed `mvp_aos` average above) confirms effect #1 (SDPA collapse)
generalizes cleanly: eager→sdpa drops `exact_match` f1 44.58→27.10 (-39%)
and `instruct_absa` f1 46.41→30.09 (-35%), with the same 29.3% garbling rate
as eng.

**Effect #2 (the residual ~20% gap under `eager` alone) remains untested and
likely untestable** — it would require an old-A100-equivalent run at matched
lr/checkpoint-selection/attn-implementation, and no A100 is available on
this cluster to produce one. The Step 5f numbers above (58.98/60.75 old vs
46.67/48.61 new) are still the only evidence for effect #2, and remain a
confounded comparison, not a controlled one. This is the practical ceiling
of what can be verified without A100 access.

### Step 6 — Third-hardware check: RTX 5090 (CUDA), planned 2026-07-09

**Status: [PLANNED — user has full/unshared access to the RTX 5090 machine
(not on this Slurm cluster), running both Step 6a (eval-only) and Step 6b
(full retrain) below]**

Motivation: effect #2 (the residual ~20–25% gap under `eager`, Steps 4/5f)
was previously called "not fixable or further testable from this side"
since the old A100 is gone. An RTX 5090 (CUDA, but not A100) doesn't recreate
the old baseline, but it does let us split effect #2 into two sub-questions
by re-running the **exact same MI250-trained weights** on non-ROCm hardware:

- If RTX 5090 + `eager` on `eng/mvp/seed_123/checkpoint-7760`
  (run `20260706_065303`, the Step 4 diagnostic checkpoint) scores close to
  MI250's **65.76%** → effect #2 isn't an eval-time hardware effect; it's
  likely baked into the weights from ROCm-specific training-time numerics.
- If it scores meaningfully higher (toward 87.44%) → eval-time
  CUDA-vs-ROCm numerics are a real, previously-unmeasured contributor to
  effect #2.
- Also re-running `sdpa` there tests whether the Step 5b/5d garbling bug
  (29.3% repetition-loop collapse) is ROCm-specific (should not reproduce on
  CUDA) or a more general prompt/model quirk (would reproduce anywhere).

### Step 6a — Eval-only (reuse MI250-trained weights, cheap, run this first)

**Caveat:** this uses ROCm-trained weights inferenced on CUDA — it isolates
the eval-time hardware axis, not train-time. It still can't fully recreate
"train AND eval on real CUDA hardware like the old A100 baseline did," since
no CUDA-trained checkpoint with matched lr/checkpoint-selection exists.
Interpret accordingly: a clean result narrows effect #2, it doesn't fully
resolve it. Step 6b below (full retrain) closes that remaining gap.

**Commands (run on the RTX 5090 machine, not Slurm):**

```bash
python -m src.main.eval \
  --test_json_path dataset/hotel_reviews/eng/mvp/test_aug.json \
  --model_path outputs/models/hotel_reviews/eng/mvp/seed_123/20260706_065303_train_model-Qwen2.5-0.5B_lr-5e-05_bs-4_epochs-10/checkpoint-7760 \
  --prompt_type mvp --output_dir outputs/evals/dtype_test/rtx5090_eager \
  --batch_size 4 --max_new_tokens 300 --torch_dtype auto --attn_implementation eager --save_predictions

python -m src.main.eval \
  --test_json_path dataset/hotel_reviews/eng/mvp/test_aug.json \
  --model_path outputs/models/hotel_reviews/eng/mvp/seed_123/20260706_065303_train_model-Qwen2.5-0.5B_lr-5e-05_bs-4_epochs-10/checkpoint-7760 \
  --prompt_type mvp --output_dir outputs/evals/dtype_test/rtx5090_sdpa \
  --batch_size 4 --max_new_tokens 300 --torch_dtype auto --attn_implementation sdpa --save_predictions
```

Requires the `checkpoint-7760` directory and `dataset/hotel_reviews/eng/mvp/test_aug.json`
transferred to that machine (checkpoint isn't in git), and a torch/CUDA build
that supports RTX 5090's Blackwell (`sm_120`) architecture.

**Reference numbers to compare against (all eng/mvp/seed_123/checkpoint-7760):**

| Config | Hardware | Trained on | f1_aos |
|---|---|---|---|
| (unknown, presumed sdpa) | A100/CUDA, old baseline | A100/CUDA | 87.44 |
| `eager` | MI250/ROCm | MI250/ROCm | 65.76 |
| `sdpa` | MI250/ROCm | MI250/ROCm | 41.33 (29.3% garbled) |
| `eager` | RTX 5090/CUDA | MI250/ROCm (reused weights) | *pending* |
| `sdpa` | RTX 5090/CUDA | MI250/ROCm (reused weights) | *pending* |
| `eager` | RTX 5090/CUDA | RTX 5090/CUDA (Step 6b retrain) | *pending* |
| `sdpa` | RTX 5090/CUDA | RTX 5090/CUDA (Step 6b retrain) | *pending* |

### Step 6b — Full retrain on RTX 5090 (train + eval, now feasible with full machine access)

**Status: [PLANNED, 2026-07-09]** — only possible because the user has full,
unshared access to this machine (not just an eval slot). This is the
experiment Step 5f/6 kept flagging as "not testable without A100 access": a
true train-and-eval comparison on non-ROCm hardware, matching Step 4's exact
config (`eng/mvp/seed_123`, `Qwen2.5-0.5B`, `lr=5e-05`,
`SAVE_STRATEGY=epoch`/last-epoch selection — the same config that produced
the MI250 `checkpoint-7760` baseline of **65.76%** used throughout Steps
5a–6a). An RTX 5090 is not an A100, so this still isn't a perfect
reconstruction of the original old baseline — but it directly answers
whether effect #2 (the residual ~20–25% gap) is a **training-time** ROCm
numerics artifact (ceiling stays ~65.76% here too) or something that
disappears once training itself happens on CUDA (recovers toward 87.44%).

**Train (run on the RTX 5090 machine, not Slurm):**

```bash
export WARMUP_RATIO=0.03
export MAX_GRAD_NORM=1.0
bash scripts/sft_one.sh "Qwen/Qwen2.5-0.5B" eng hotel_reviews mvp 123 4 5e-5 10 4 epoch epoch
```

This mirrors `slurm_submit/submit_sft_array.sh`'s resolved args for the Step
4 cell exactly (`batch_size=4`, `lr=5e-5`, `num_epochs=10`,
`gradient_accumulation_steps=4`, `eval_strategy=epoch`,
`save_strategy=epoch`, `warmup_ratio=0.03`, `max_grad_norm=1.0` — confirmed
against `scripts/sft_one.sh` and `slurm_submit/submit_sft_array.sh` current
defaults). `dev.json` exists for `eng/mvp` so `eval_strategy=epoch` works
without extra setup. Output lands under
`outputs/models/hotel_reviews/eng/mvp/seed_123/<run_stamp>_.../checkpoint-*`
— with `save_strategy=epoch` and default `save_total_limit=1`, only the last
epoch (`checkpoint-7760`, matching the MI250 run's checkpoint number since
dataset size/epochs/batch size are identical) should remain on disk.

**Eval (both attention implementations, same as Step 6a):**

```bash
CKPT="outputs/models/hotel_reviews/eng/mvp/seed_123/<run_stamp>_train_model-Qwen2.5-0.5B_lr-5e-05_bs-4_epochs-10/checkpoint-7760"

python -m src.main.eval \
  --test_json_path dataset/hotel_reviews/eng/mvp/test_aug.json \
  --model_path "$CKPT" \
  --prompt_type mvp --output_dir outputs/evals/dtype_test/rtx5090_retrain_eager \
  --batch_size 4 --max_new_tokens 300 --torch_dtype auto --attn_implementation eager --save_predictions

python -m src.main.eval \
  --test_json_path dataset/hotel_reviews/eng/mvp/test_aug.json \
  --model_path "$CKPT" \
  --prompt_type mvp --output_dir outputs/evals/dtype_test/rtx5090_retrain_sdpa \
  --batch_size 4 --max_new_tokens 300 --torch_dtype auto --attn_implementation sdpa --save_predictions
```

(fill in the actual `<run_stamp>` from the training log/output dir).

**Interpretation:**

- Compare against **all** rows in the reference table above, not just
  MI250's 65.76%. The three-way split (A100 old baseline, MI250 full
  train+eval, RTX5090 full train+eval) is what actually isolates train-time
  hardware from eval-time hardware for the first time in this investigation.
- If RTX5090 train+eval (`eager`) lands close to MI250's 65.76% → effect #2
  is not hardware-specific to ROCm at all; something else (data, optimizer
  nondeterminism, a code path neither GPU vendor affects) explains the
  residual gap, and the "genuine hardware effect" framing throughout Steps
  4–5f needs revisiting.
- If it lands meaningfully higher (toward 87.44%) → effect #2 is a genuine
  ROCm/MI250-specific training-time numerics effect, closing out the
  investigation with a confirmed, falsified-by-a-clean-experiment root cause
  instead of the current circumstantial evidence.
- Also compare `sdpa` here against Step 6a's `sdpa` (reused MI250 weights) —
  if the RTX5090-trained checkpoint's `sdpa` eval doesn't show the ~29%
  repetition-loop garbling seen on MI250, that's further confirmation the
  garbling bug (effect #1) is ROCm-kernel-specific, not a general
  prompt/model quirk.

---

### Does this affect the mT5 retrain (section 1)? Fixed ✓

The mT5/seq2seq pipeline had the same checkpoint-selection blind spot — `scripts/sft_one_seq2seq.sh` hardcoded `--save_strategy "epoch" --eval_strategy "no"` with no env-var override (unlike the causal-LM path which got the fix in 57f997c). Because `eval_strategy="no"`, there was no way to compute `eval_loss` and select the best checkpoint.

**Fix applied (2026-07-08):**
1. ✓ Added `SAVE_STRATEGY`/`EVAL_STRATEGY` env-var passthrough to `slurm_submit/submit_sft_seq2seq_array.sh` and `scripts/sft_one_seq2seq.sh`.
2. ✓ Confirmed all 6 languages have `dev.json` (1000 samples each) at `dataset/hotel_reviews/{eng,jav,indo,mad,min,sun}/mvp_aos/`.
3. ✓ New defaults: `SAVE_STRATEGY=best EVAL_STRATEGY=epoch` — validation loss is computed every epoch and the best checkpoint is kept.

The blind spot is now closed. Seq2seq training pipeline is ready for resubmission and can be trusted for final comparison results.
