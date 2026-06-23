# Model Card: Vietnamese HSD Classifier

## Model

Default model: `vinai/phobert-base`

Alternative model: XLM-R via the same training script by changing `training.model_name` in `configs/config.yaml`.

## Intended Use

Classify Vietnamese text into `CLEAN`, `OFFENSIVE`, or `HATE` for research on augmentation effectiveness.

## Training Data

The model is trained on fixed processed splits in `data/processed/`, optionally combined with augmentation outputs in `data/augmented/`.

## Metrics

Primary metrics:

- Macro F1
- HATE F1
- Per-class F1
- Confusion matrix

## Limitations

The classifier may misclassify reclaimed language, slang, sarcasm, coded hate, dialectal text, and domain-shifted social media content. Outputs should not be used as the only basis for punitive moderation decisions.

## Evaluation Plan

Run the required ablation set: baseline, BT only, EDA only, LLM only, and combined augmentation. Compare final test metrics and inspect HATE false negatives in `results/error_analysis/`.
