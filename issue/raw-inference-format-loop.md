# Raw Inference Format Loop

## Symptom

Inference outputs are mostly empty strings or repeated format tokens such as:

```text
]S]S]S]S]S...
```

Expected outputs should follow the ABSA target format:

```text
[A] service [O] very friendly [S] positive
[A] bed [O] twin bed [S] negative [SSEP] [A] room [O] different [S] negative
```

## Likely Cause

The prompt format is probably not the main issue. Training and inference both use the MVP prompt style:

```text
<input> =>
```

The stronger issue is that the model is not learning or using a reliable stop condition. In `src/main/train.py`, the completion is currently built without appending the tokenizer EOS token:

```python
"completion": " " + instance["target"]
```

During inference, generation can continue until `max_new_tokens`, so the model loops common target-format tokens like `[S]`, `[O]`, and `[SSEP]`.

There are also signs of unstable training:

- Existing trainer logs showed repeated `grad_norm: Infinity`.
- Some checkpoints had weak eval loss/token accuracy.
- Eval reported tokenizer/model embedding size mismatch and resized embeddings at inference.

## What To Do

1. Add EOS to the training completion in `src/main/train.py`.

   Target behavior:

   ```python
   "completion": " " + instance["target"] + tokenizer.eos_token
   ```

2. Pass EOS and PAD IDs during generation in `src/main/eval.py`.

   Target behavior:

   ```python
   model.generate(
       **inputs,
       max_new_tokens=args.max_new_tokens,
       do_sample=False,
       eos_token_id=tokenizer.eos_token_id,
       pad_token_id=tokenizer.pad_token_id,
       logits_processor=logits_processor,
   )
   ```

3. Keep `MAX_NEW_TOKENS` smaller while debugging.

   Suggested:

   ```bash
   MAX_NEW_TOKENS=80
   LIMIT_SAMPLES=20
   DEBUG_GENERATIONS=1
   FORCE_RERUN=1
   ```

4. Retrain only one small checkpoint first, not the full Slurm array.

   Use one language, one seed, and inspect debug generations before launching all jobs.

5. Check the loss curve before trusting the checkpoint.

   Look at W&B project `absa-sft` or local `trainer_state.json`.

   Red flags:

   - `grad_norm: Infinity`
   - eval loss not improving
   - token accuracy stuck low
   - predictions repeat `[S]`, `[O]`, or `[SSEP]`

6. Prefer BF16 on MI250/ROCm if available.

   The current automatic BF16 detection may not be reliable outside NVIDIA CUDA capability checks. If the run is using FP16 on MI250, that can contribute to unstable gradients.

## Recommended Validation

After applying the EOS/generation fixes and retraining one small run:

```bash
LIMIT_SAMPLES=20 DEBUG_GENERATIONS=1 MAX_NEW_TOKENS=80 FORCE_RERUN=1 \
bash scripts/eval_all_noconstraint.sh
```

Good signs:

- First generated token is usually `[A]`.
- Output stops naturally before `MAX_NEW_TOKENS`.
- No repeated `]S]S]S...` loops.
- Predictions contain complete `[A] ... [O] ... [S] ...` triplets.

## Bottom Line

Do not trust the current bad checkpoints. Add EOS during SFT, pass EOS/PAD during inference, verify BF16 stability, retrain one small run, then rerun the full Slurm array only after debug generations look correct.

---

## Resolution (2026-05-18)

### What was done

**1. Added EOS to training completions (`src/main/train.py`)**

`split_prompt_and_completion` now appends `tokenizer.eos_token` to every completion:

```python
"completion": " " + instance["target"] + tokenizer.eos_token
```

This teaches the model to predict EOS after a complete ABSA triplet sequence, giving it a reliable stop condition.

**2. EOS and PAD IDs already present in eval (`src/main/eval.py`)**

`eos_token_id` and `pad_token_id` were already passed to `model.generate()`. No change needed here.

**3. Fixed BF16/dtype detection (`src/main/train.py`)**

The original detection used `torch.cuda.get_device_capability()`, which is an NVIDIA CUDA concept and does not work on AMD ROCm. This caused two separate bugs:

- On a Slurm GPU node (ROCm, GPU allocated): `torch.cuda.is_available()` returned True but capability check was unreliable, risking FP16 instead of BF16.
- In a regular shell (no GPU allocated): `torch.version.hip is not None` was True (PyTorch compiled with ROCm) even with no GPU assigned, causing a crash when BF16 was forced without a GPU.

Fixed logic:

```python
is_rocm = torch.version.hip is not None
if torch.cuda.is_available():
    if is_rocm:
        bf16 = True   # ROCm GPU present → BF16
    else:
        bf16 = gpu_capability[0] >= 8  # NVIDIA → check compute capability
else:
    bf16 = False  # no GPU → CPU with FP32
```

**4. CPU fallback uses FP32 instead of FP16**

When no GPU is available (regular shell debug runs), the model now loads with `torch.float32` instead of `torch.float16`. FP16 on CPU is largely unsupported by PyTorch and produces NaN loss and NaN grad norm across all steps.

### Validation result

Debug run: 50 samples, 3 epochs, CPU FP32, `google/gemma-3-270m`, English hotel reviews.

- Loss went from `2.8` → `~0.3` across 3 epochs.
- Token accuracy went from `55%` → `~90%`.
- No NaN loss or grad norm at any step.

Eval on 20 test samples with `--max_new_tokens 80 --debug_generations`:

- First generated token was `[A]` on every sample.
- Most outputs stopped naturally before the token budget (EOS predicted).
- No `]S]S]S...` loops.
- F1 AOS: 37.3% — low but expected for a tiny debug checkpoint.

### What remains

- Old checkpoints (before this fix) are invalid. Do not evaluate them.
- Retrain all languages and seeds in Slurm using the corrected `train.py`.
- Full-dataset GPU runs with BF16 are expected to produce substantially better F1.
