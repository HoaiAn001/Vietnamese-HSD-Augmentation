from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset


@dataclass(frozen=True)
class HFDatasetConfig:
    repo_id: str
    config_name: str
    train_split: str = "train"
    dev_split: str = "validation"
    test_split: str = "test"


def hf_dataset_config(config: dict, config_name: str | None = None) -> HFDatasetConfig:
    hf_cfg = config["huggingface"]
    splits = hf_cfg.get("split_names", {})
    return HFDatasetConfig(
        repo_id=hf_cfg["dataset_repo_id"],
        config_name=config_name or hf_cfg["baseline_config"],
        train_split=splits.get("train", "train"),
        dev_split=splits.get("dev", "validation"),
        test_split=splits.get("test", "test"),
    )


def load_hf_splits(config: dict, config_name: str | None = None) -> dict[str, pd.DataFrame]:
    hf = hf_dataset_config(config, config_name)
    dataset = load_dataset(hf.repo_id, hf.config_name)
    return {
        "train": dataset[hf.train_split].to_pandas(),
        "dev": dataset[hf.dev_split].to_pandas(),
        "test": dataset[hf.test_split].to_pandas(),
    }


def push_hf_splits(
    config: dict,
    train: pd.DataFrame,
    dev: pd.DataFrame,
    test: pd.DataFrame,
    config_name: str,
    private: bool | None = None,
    token: str | None = None,
) -> None:
    hf_cfg = config["huggingface"]
    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(train.reset_index(drop=True), preserve_index=False),
            "validation": Dataset.from_pandas(dev.reset_index(drop=True), preserve_index=False),
            "test": Dataset.from_pandas(test.reset_index(drop=True), preserve_index=False),
        }
    )
    dataset.push_to_hub(
        hf_cfg["dataset_repo_id"],
        config_name=config_name,
        private=hf_cfg["private"] if private is None else private,
        token=token,
    )
