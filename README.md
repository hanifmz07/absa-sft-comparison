# ABSA SFT Comparison

Repository for training and evaluating Aspect-Based Sentiment Analysis (ABSA) models across multiple prompt formats and languages.

This project contains:
- Causal LM SFT (Qwen)
- Seq2Seq SFT (mT5)
- Constrained and unconstrained decoding evaluation
- Multi-language loop runners for batch experiments

## Project Layout

- src/main: training and evaluation entrypoints
- src/utils: constrained decoding, evaluation utilities, I/O helpers
- scripts: experiment runner scripts
- dataset: datasets grouped by dataset type, language, and dataset folder
- outputs, outputs_seq2seq: model checkpoints and evaluation outputs
- logs: training and evaluation logs

## Requirements

- Python 3.12+
- CUDA-capable GPU (recommended)

## Environment Setup

1. Create a virtual environment with uv.
2. Install dependencies from pyproject.toml with uv.
3. Activate the environment.

Example:

```bash
uv sync
source .venv/bin/activate
```

### PyTorch Variant

The base project accepts both the default PyTorch line and the ROCm 6.2 build used on AMD GPU nodes. Install the PyTorch variant that matches the machine.

Default machine:

```bash
pip install -r requirements/torch-default.txt
pip install -e . --no-deps
```

AMD ROCm 6.2 machine:

```bash
pip install -r requirements/torch-rocm6.2.txt
pip install -e . --no-deps
```

Use `--no-deps` when refreshing the editable project install after selecting a PyTorch variant so pip does not replace the selected torch wheel.

Optional environment variables (recommended):

```bash
export WANDB_API_KEY="your_wandb_api_key"
export HF_CACHE_DIR="$HOME/.cache/huggingface"
```

You can also place them in a .env file at the repository root.

## Data Convention

Most scripts expect this structure:

```text
dataset/{dataset_type}/{language}/{dataset_folder}/
	train.json
	dev.json        # optional, if eval_strategy is not "no"
	test.json       # used by eval_all_noconstraint.sh and eval_all_seq2seq.sh
	test_aug.json   # used by eval_all.sh (causal LM constrained decoding)
```

## Run Experiments

### 1. Causal LM SFT (Qwen)

Single language:

```bash
./scripts/sft_all.sh <language> <dataset_type> <dataset_folder> <batch_size>
```

Example:

```bash
./scripts/sft_all.sh eng hotel_reviews mvp 4
```

Loop across languages:

```bash
./scripts/sft_all_loop_langs.sh <dataset_type> <dataset_folder> <batch_size>
```

Example:

```bash
./scripts/sft_all_loop_langs.sh hotel_reviews mvp 4
```

### 2. Seq2Seq SFT (mT5)

Single language:

```bash
./scripts/sft_all_seq2seq.sh <language> <dataset_type> <dataset_folder> <batch_size>
```

Loop across languages:

```bash
./scripts/sft_all_seq2seq_loop_langs.sh <dataset_type> <dataset_folder> <batch_size>
```

Example:

```bash
./scripts/sft_all_seq2seq_loop_langs.sh hotel_reviews mvp 4
```

### 3. Evaluation (Causal LM)

Constrained decoding:

```bash
./scripts/eval_all.sh <language> <dataset_type> <dataset_folder> <batch_size>
```

Unconstrained decoding:

```bash
./scripts/eval_all_noconstraint.sh <language> <dataset_type> <dataset_folder> <batch_size>
```

Loop unconstrained eval across languages:

```bash
./scripts/eval_all_noconstraint_loop_langs.sh <dataset_type> <dataset_folder> <batch_size> [languages...]
```

Examples:

```bash
./scripts/eval_all.sh eng hotel_reviews mvp 4
./scripts/eval_all_noconstraint.sh eng hotel_reviews mvp 4
./scripts/eval_all_noconstraint_loop_langs.sh hotel_reviews mvp 4 eng jav mad
```

### 4. Evaluation (Seq2Seq mT5)

Single language:

```bash
./scripts/eval_all_seq2seq.sh <language> <dataset_type> <dataset_folder> <batch_size> [use_constrained_decoding]
```

Loop across languages:

```bash
./scripts/eval_all_seq2seq_loop_langs.sh <dataset_type> <dataset_folder> <batch_size> [use_constrained_decoding] [languages...]
```

Examples:

```bash
./scripts/eval_all_seq2seq.sh eng hotel_reviews mvp 4 false
./scripts/eval_all_seq2seq.sh eng hotel_reviews mvp 4 true
./scripts/eval_all_seq2seq_loop_langs.sh hotel_reviews mvp 4 false eng jav mad
```

Notes:

- Seq2Seq eval reads checkpoints from outputs_seq2seq/models/{dataset_type}/{language}/{dataset_folder}/seed_{seed}/checkpoint-*.
- The script evaluates the latest checkpoint per seed.

### 5. Post-Evaluation Metrics (from inference results)

These scripts evaluate already-generated result files under:

`{output_dir}/evals/{dataset_type}/{language}/{dataset_folder}/**/{inference_results.json|voting_results.json}`

For `mvp`, both `inference_results.json` and `voting_results.json` are processed.
For other dataset folders, only `inference_results.json` is processed.

#### Exact Match

How it works:

- Compares predicted tuples/triplets against target tuples/triplets with strict string matching.
- A prediction counts as true positive only when the tuple text exactly matches a target tuple.
- Unmatched predictions are false positives; unmatched targets are false negatives.

Single dataset folder:

```bash
./scripts/eval_exact_match.sh <output_dir> <dataset_type> <language> <dataset_folder>
```

Loop default setup (hotel_reviews, langs: eng/indo/jav/mad/min/sun, folders: mvp_aos/mvp):

```bash
./scripts/eval_all_exact_match.sh <output_dir>
```

Saved outputs (same directory as each processed input file):

- `exact_match.json`
- `exact_match_detail.json`
- `voting_exact_match.json` (for `voting_results.json`)
- `voting_exact_match_detail.json` (for `voting_results.json`)

#### Semantic Similarity

How it works:

- Starts from exact-match mismatches, then compares unmatched prediction-target pairs using embedding cosine similarity.
- If cosine similarity is above threshold (`0.9`), the pair is treated as a semantic match.
- Remaining unmatched predictions/targets become false positives/false negatives.

Single dataset folder:

```bash
./scripts/eval_semantic_similarity.sh <output_dir> <dataset_type> <language> <dataset_folder> <embedding_model_name>
```

Loop default setup:

```bash
./scripts/eval_all_semantic_similarity.sh <output_dir> <embedding_model_name>
```

Saved outputs (same directory as each processed input file):

- `semantic_metrics.json`
- `semantic_metrics_detail.json`
- `voting_semantic_matrics.json` (for `voting_results.json`)
- `voting_semantic_metrics_detail.json` (for `voting_results.json`)

#### InstructABSA-style Metric

How it works:

- Converts ABSA text format (`[A] ... [O] ... [S] ...`) into AOSTE-like triplet text (`aspect:opinion:sentiment`).
- Uses InstructABSA-style overlap matching (`pred in gt` or `gt in pred`) instead of strict equality.
- This allows partial/expanded wording to count as a match when semantically aligned in string form.

Single dataset folder:

```bash
./scripts/eval_instruct_absa.sh <output_dir> <dataset_type> <language> <dataset_folder>
```

Loop default setup:

```bash
./scripts/eval_all_instruct_absa.sh <output_dir>
```

Saved outputs (same directory as each processed input file):

- `instruct_absa.json`
- `instruct_absa_detail.json`
- `voting_instruct_absa.json` (for `voting_results.json`)
- `voting_instruct_absa_detail.json` (for `voting_results.json`)

Notes:

- In exact-match, semantic, and instruct-absa outputs, `precision`, `recall`, and `f1` are stored in 0-100 scale.
- Detail files include per-instance false positives and false negatives.

## Outputs and Logs

### Model outputs

- Causal LM checkpoints:
	- outputs/models/{dataset_type}/{language}/{dataset_folder}/seed_{seed}/...
- Seq2Seq checkpoints:
	- outputs_seq2seq/models/{dataset_type}/{language}/{dataset_folder}/seed_{seed}/...

### Evaluation outputs

- outputs/evals/{dataset_type}/{language}/{dataset_folder}/seed_{seed}/...
	- .../constrained_decoding/evaluation_results.json
	- .../unconstrained_decoding/evaluation_results.json
	- inference_results.json and raw_inference_results.json
- outputs_seq2seq/evals/{dataset_type}/{language}/{dataset_folder}/{seed_dir}/{checkpoint_dir}/...
	- .../constrained_decoding/evaluation_results.json
	- .../unconstrained_decoding/evaluation_results.json
	- .../inference_results.json and .../raw_inference_results.json

### Logs

- Training logs:
	- logs/sft_full/...
	- logs/sft_seq2seq_full/...
- Evaluation logs:
	- logs/eval/...
	- logs/eval_seq2seq/...

## Notes

- Scripts run across seeds: 9584, 123, 2024, 31415, 777.
- Prompt type is inferred from dataset folder name in script logic (mvp, gas, legoabsa).
- JSON loading uses encoding fallback in Python entrypoints to handle mixed-encoding datasets.

## Direct Module Commands (Optional)

If you prefer running modules directly instead of shell wrappers:

```bash
python -m src.main.train --help
python -m src.main.train_seq2seq --help
python -m src.main.eval --help
python -m src.main.eval_seq2seq --help
```
