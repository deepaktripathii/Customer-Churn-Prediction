"""Train, evaluate and export the churn models.

Usage::

    python scripts/train.py                        # all available models
    python scripts/train.py --models logistic_regression
    python scripts/train.py --output-dir artifacts/experiments/run-01
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

import _bootstrap  # noqa: F401  (adds src/ to sys.path)
from churn_prediction.config import get_settings
from churn_prediction.data.loader import DatasetNotFoundError, load_dataset, profile_dataset
from churn_prediction.data.quality import run_quality_checks
from churn_prediction.features.cleaning import clean_churn_data
from churn_prediction.models.registry import available_models
from churn_prediction.models.train import train_and_save


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Customer Churn Prediction churn models.")
    parser.add_argument("--data", default=None, help="Raw CSV path (default: data/raw/telco_churn.csv).")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=f"Subset of models to train. Available: {', '.join(available_models())}",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Artifact directory (default: ./artifacts).",
    )
    parser.add_argument(
        "--metric",
        default="f1",
        choices=["f1", "recall", "roc_auc", "precision"],
        help="Metric used to pick the best model.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=None,
        help="Fraction of rows reserved for the final test set (default 0.2).",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Train and print metrics without writing artifacts.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_settings()
    if args.output_dir:
        settings = settings.with_artifact_dir(args.output_dir)
    if args.test_size:
        from dataclasses import replace

        settings = replace(settings, test_size=args.test_size)

    try:
        raw = load_dataset(args.data)
    except DatasetNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    profile = profile_dataset(raw, source=str(args.data or settings.raw_data_path))
    print(f"Dataset : {profile.n_rows:,} rows x {profile.n_columns} columns")
    print(f"Churn   : {profile.target_distribution}")

    quality = run_quality_checks(raw)
    for issue in quality.warnings:
        print(f"  [warning] {issue.message}")

    cleaned, cleaning_report = clean_churn_data(raw)
    print(
        f"Cleaned : {cleaning_report.rows_in:,} -> {cleaning_report.rows_out:,} rows "
        f"({cleaning_report.duplicates_removed} duplicates removed)"
    )

    models = args.models or available_models()
    print(f"Models  : {', '.join(models)}")

    runs, best = train_and_save(
        cleaned,
        model_names=models,
        settings=settings,
        cleaning_report=cleaning_report.to_dict(),
        dataset_info={
            "source": str(args.data or settings.raw_data_path),
            "n_columns_raw": profile.n_columns,
        },
        progress=lambda message: print(f"  - {message}"),
        metric=args.metric,
    )

    table = pd.DataFrame([run.summary_row() for run in runs.values()])
    table = table.sort_values(args.metric, ascending=False).reset_index(drop=True)
    print("\nModel comparison (test set):")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    best_run = runs[best]
    print(
        f"\nBest model: {best_run.display_name} "
        f"(threshold {best_run.threshold:.3f}, recall {best_run.test_metrics['recall']:.3f})"
    )
    if best_run.test_metrics["recall"] < 0.7:
        print(
            "  [warning] Recall below 0.70 - many churners would be missed. "
            "Consider a lower threshold or more training data."
        )
    print(f"Artifacts : {settings.artifact_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
