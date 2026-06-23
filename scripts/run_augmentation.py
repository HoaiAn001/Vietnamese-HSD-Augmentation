from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.augmentation import BackTranslationAugmenter, EDAAugmenter, LLMAugmenter
from src.utils.io import load_config, read_table, write_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run configured augmentation strategy.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--method", choices=["bt", "eda", "llm", "combined"], required=True)
    parser.add_argument("--input", default=None, help="Defaults to config data.train_file.")
    parser.add_argument("--llm-generated", default=None, help="CSV with original_text,generated_text,label.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    aug_cfg = config["augmentation"]
    input_file = args.input or config["data"]["train_file"]
    train = read_table(input_file)
    outputs: list[pd.DataFrame] = []

    if args.method in {"bt", "combined"}:
        bt_cfg = aug_cfg["back_translation"]
        augmenter = BackTranslationAugmenter(
            source_language=bt_cfg["source_language"],
            intermediate_language=bt_cfg["intermediate_language"],
            random_state=config["project"]["seed"],
        )
        bt = augmenter.augment(train, max_samples=bt_cfg["max_samples"], labels=["OFFENSIVE", "HATE"])
        write_table(bt, bt_cfg["output_file"])
        outputs.append(bt)

    if args.method in {"eda", "combined"}:
        eda_cfg = aug_cfg["eda"]
        augmenter = EDAAugmenter(
            operations=eda_cfg["operations"],
            alpha=eda_cfg["alpha"],
            num_aug=eda_cfg["num_aug"],
            random_state=config["project"]["seed"],
        )
        eda = augmenter.augment(train, labels=["OFFENSIVE", "HATE"])
        write_table(eda, eda_cfg["output_file"])
        outputs.append(eda)

    if args.method in {"llm", "combined"}:
        llm_cfg = aug_cfg["llm"]
        if args.llm_generated:
            generated = read_table(args.llm_generated)
            llm = pd.DataFrame(
                [
                    {
                        "text": row["generated_text"],
                        "label": row["label"],
                        "original_text": row["original_text"],
                        "augmentation_method": "llm",
                        "metadata": row.get("metadata", "external_generation"),
                    }
                    for row in generated.to_dict(orient="records")
                ]
            )
            write_table(llm, llm_cfg["output_file"])
            outputs.append(llm)
        else:
            prompt_log = Path(llm_cfg["prompt_log_file"])
            if prompt_log.exists():
                prompt_log.unlink()
            augmenter = LLMAugmenter(
                prompt_template=llm_cfg["prompt_template"],
                prompt_log_file=prompt_log,
                random_state=config["project"]["seed"],
            )
            augmenter.augment(train, max_samples=llm_cfg["max_samples"], labels=["OFFENSIVE", "HATE"])
            print(f"Wrote LLM prompt log to {prompt_log}. Fill generations and rerun with --llm-generated.")

    if args.method == "combined" and outputs:
        combined = pd.concat(outputs, ignore_index=True)
        write_table(combined, Path(config["paths"]["augmented_dir"]) / "combined.csv")
        print(f"Wrote combined augmentation with {len(combined)} rows.")
    elif outputs:
        print(f"Wrote {args.method} augmentation with {sum(len(df) for df in outputs)} rows.")


if __name__ == "__main__":
    main()
