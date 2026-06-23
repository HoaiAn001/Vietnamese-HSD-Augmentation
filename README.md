# Vietnamese Hate Speech Detection Augmentation

This repository is a reproducible research pipeline for studying whether data augmentation improves Vietnamese Hate Speech Detection (HSD). The current implementation focuses on a solid research deliverable by the end of August 2026: fixed data splits, BT/EDA/LLM augmentation outputs, PhoBERT training, ablation metrics, and error analysis.

Frontend, FastAPI deployment, Docker, and full MLflow tracking are intentionally postponed until the research pipeline is complete.

## Current Status

Implemented:

- Reproducible config in `configs/config.yaml`
- Data split preparation script
- OOP augmentation interface with Back Translation, EDA, and controlled LLM prompt logging
- PhoBERT/XLM-R compatible training and evaluation scripts
- Metrics export for Accuracy, Macro F1, Weighted F1, HATE F1, confusion matrix, and predictions
- Dataset/model card drafts in `docs/`

In progress:

- Final processed train/dev/test datasets
- Actual large-scale augmentation runs
- Baseline and ablation experiment results

Planned after the research result is stable:

- Statistical significance testing
- MLflow tracking
- Web demo and deployment

## Project Structure

```text
Vietnamese-HSD-Augmentation/
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── augmented/
├── docs/
│   ├── dataset_card.md
│   └── model_card.md
├── notebooks/
│   ├── 01_data_collection.ipynb
│   └── 02_back_translation.ipynb
├── results/
│   ├── error_analysis/
│   ├── figures/
│   └── metrics/
├── scripts/
│   ├── prepare_splits.py
│   ├── run_augmentation.py
│   ├── train_classifier.py
│   ├── evaluate_model.py
│   └── summarize_ablation.py
├── src/
│   ├── augmentation/
│   ├── models/
│   └── utils/
├── saved_models/
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

Use Python 3.10+ if possible. GPU is strongly recommended for transformer training.

## Data Contract

The training pipeline expects CSV files with:

- `text`: Vietnamese text
- `label`: one of `CLEAN`, `OFFENSIVE`, `HATE`

Default paths are configured in `configs/config.yaml`:

- `data/processed/train.csv`
- `data/processed/dev.csv`
- `data/processed/test.csv`

If VLSP or another private dataset is used, keep the raw file outside Git and document the local path in `configs/config.yaml`.

## Run Pipeline

Create fixed splits from one normalized CSV:

```bash
python scripts/prepare_splits.py --input data/raw/combined.csv
```

Run augmentation:

```bash
python scripts/run_augmentation.py --method eda
python scripts/run_augmentation.py --method bt
python scripts/run_augmentation.py --method llm
```

LLM augmentation writes prompts to `data/augmented/llm_prompts.jsonl`. Fill generations externally, then ingest them:

```bash
python scripts/run_augmentation.py --method llm --llm-generated data/augmented/llm_generated.csv
```

Train baseline:

```bash
python scripts/train_classifier.py --experiment baseline
```

Train with one augmentation set:

```bash
python scripts/train_classifier.py --experiment eda_only --augmented-file data/augmented/eda_only.csv
```

Evaluate:

```bash
python scripts/evaluate_model.py --model-dir saved_models/baseline --experiment baseline
```

Summarize ablation:

```bash
python scripts/summarize_ablation.py
```

## Required Experiments

Minimum experiments for the August 2026 deliverable:

- `baseline`
- `bt_only`
- `eda_only`
- `llm_only`
- `combined`

Primary comparison metrics:

- Macro F1
- HATE F1
- Per-class F1
- Confusion matrix

## Notes

- Notebook code is useful for exploration, but reusable logic should live in `src/` and `scripts/`.
- Generated data, model checkpoints, and private datasets are intentionally excluded from Git.
- The project should prioritize reproducible research results before deployment work.
