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
	test.json       # used by eval_all_noconstraint.sh
	test_aug.json   # used by eval_all.sh
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

### Logs

- Training logs:
	- logs/sft_full/...
	- logs/sft_seq2seq_full/...
- Evaluation logs:
	- logs/eval/...

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
