# Dataset Card: Vietnamese HSD Combined

## Dataset Summary

This dataset is intended for Vietnamese Hate Speech Detection experiments with three labels: `CLEAN`, `OFFENSIVE`, and `HATE`.

## Sources

- ViHSD from Hugging Face
- VLSP 2019 if the private file is available locally
- Optional text-only corpora for augmentation source material

## Labels

- `CLEAN`: non-abusive text
- `OFFENSIVE`: offensive or toxic text without clear hate-speech targeting
- `HATE`: hate speech targeting a person or group

## Processing

Text is normalized to NFC Unicode, whitespace is collapsed, empty rows are removed, labels are mapped to the canonical three-class schema, and duplicate `(text, label)` pairs are removed.

## Splits

Fixed train/dev/test splits are produced by `scripts/prepare_splits.py` using the seed in `configs/config.yaml`.

## Hugging Face Versioning

The Hub dataset repo should use one config per experiment:

- `baseline`: original processed train/dev/test
- `bt`: Back Translation augmented train with unchanged validation/test
- `eda`: EDA augmented train with unchanged validation/test
- `llm`: LLM augmented train with unchanged validation/test
- `combined`: combined augmented train with unchanged validation/test

Validation and test splits must remain untouched across configs.

## Limitations

Private datasets cannot be redistributed through this repository. Label definitions may differ across source datasets, so merged data should be manually inspected before final experiments.

## Ethical Considerations

The dataset contains toxic and hateful language. Access should be limited to research, moderation, and safety use cases. Do not use generated samples to amplify harassment.
