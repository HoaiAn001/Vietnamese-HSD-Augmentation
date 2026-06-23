from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize experiment metrics into one CSV.")
    parser.add_argument("--metrics-dir", default="results/metrics")
    parser.add_argument("--output", default="results/metrics/ablation_summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for path in sorted(Path(args.metrics_dir).glob("*_test_metrics.json")):
        with path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        rows.append(
            {
                "experiment": path.name.replace("_test_metrics.json", ""),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "hate_f1": metrics["hate_f1"],
                "hate_support": metrics["hate_support"],
            }
        )
    summary = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False, encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
