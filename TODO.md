# TODO

---

## 1. Retrain mT5-base

### Status: Ready for submission with fixed checkpoint selection

Slurm array scripts updated — 30 jobs (6 langs × 5 seeds), 5 concurrent. Now includes validation-based checkpoint selection (`save_strategy=best`, `eval_strategy=epoch`):

```bash
sbatch slurm_submit/submit_sft_seq2seq_array.sh
```

The training pipeline no longer has the blind-spot bug documented below (see section "Does this affect the mT5 retrain..."). All 6 languages have `dev.json` for validation. The default now trains with best-checkpoint selection enabled.

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
