"""Model training and artifact persistence (Feature 4).

Workflow
--------
1. Stratified three-way split: train / validation / test.
2. Fit every registered model on the training split.
3. Choose the decision threshold on the validation split by maximising F1.
4. Refit on train + validation, then report metrics on the untouched test set.
5. Persist pipelines, metrics, a model card and drift reference statistics.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

from churn_prediction.config import TARGET_COLUMN, Settings, get_settings
from churn_prediction.features.engineering import add_derived_features
from churn_prediction.features.pipeline import (
    build_pipeline,
    feature_names,
    split_feature_types,
    stratified_split,
)
from churn_prediction.logging_config import get_logger
from churn_prediction.models.evaluate import comparison_table, evaluate_pipeline, evaluate_scores
from churn_prediction.models.registry import available_models, get_model_spec

logger = get_logger(__name__)


class TrainingError(RuntimeError):
    """Raised when training cannot produce a usable model."""


@dataclass
class ModelRun:
    """Everything produced by training a single algorithm."""

    name: str
    display_name: str
    pipeline: Any
    threshold: float
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    cv_roc_auc: float
    cv_roc_auc_std: float
    train_seconds: float
    model_columns: list[str] = field(default_factory=list)

    def summary_row(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "threshold": round(self.threshold, 4),
            "accuracy": self.test_metrics["accuracy"],
            "precision": self.test_metrics["precision"],
            "recall": self.test_metrics["recall"],
            "f1": self.test_metrics["f1"],
            "roc_auc": self.test_metrics["roc_auc"],
            "pr_auc": self.test_metrics["pr_auc"],
            "cv_roc_auc": round(self.cv_roc_auc, 4),
            "train_seconds": round(self.train_seconds, 2),
        }


def optimal_threshold(y_true: Sequence[int] | np.ndarray, y_score: Sequence[float]) -> float:
    """Pick the probability cut-off that maximises F1 on validation data."""
    from sklearn.metrics import precision_recall_curve

    y_true_arr = np.asarray(y_true).astype(int)
    y_score_arr = np.asarray(y_score, dtype=float)
    if len(np.unique(y_true_arr)) < 2:
        return 0.5

    precision, recall, thresholds = precision_recall_curve(y_true_arr, y_score_arr)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    # ``thresholds`` has one fewer element than precision/recall.
    best_index = int(np.argmax(f1[:-1])) if len(thresholds) else 0
    threshold = float(thresholds[best_index]) if len(thresholds) else 0.5
    return float(np.clip(threshold, 0.05, 0.95))


def infer_defaults(df: pd.DataFrame, *, target_column: str = TARGET_COLUMN) -> dict[str, Any]:
    """Typical customer profile used to fill unspecified payload fields."""
    defaults: dict[str, Any] = {}
    for column in df.columns:
        if column == target_column:
            continue
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            value = float(series.median())
            defaults[column] = int(value) if value.is_integer() and series.dtype.kind in "iu" else value
        else:
            mode = series.mode(dropna=True)
            defaults[column] = str(mode.iloc[0]) if not mode.empty else "Unknown"
    return defaults


def _to_python(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def build_reference_stats(X: pd.DataFrame, *, n_bins: int = 10) -> dict[str, Any]:
    """Reference distributions used by the PSI drift detector."""
    stats: dict[str, Any] = {}
    for column in X.columns:
        series = X[column]
        if pd.api.types.is_numeric_dtype(series):
            values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
            if values.size == 0:
                continue
            quantiles = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
            edges = [float(q) for q in quantiles]
            if len(edges) < 2:
                edges = [float(values.min()), float(values.max()) + 1e-9]
            counts, _ = np.histogram(values, bins=edges)
            proportions = (counts / counts.sum()).tolist()
            stats[str(column)] = {
                "type": "numeric",
                "edges": edges,
                "proportions": [float(p) for p in proportions],
            }
        else:
            value_counts = series.astype(str).value_counts(normalize=True)
            if value_counts.empty:
                continue
            top = value_counts.head(25)
            other = float(value_counts.iloc[25:].sum()) if len(value_counts) > 25 else 0.0
            proportions = {str(k): float(v) for k, v in top.items()}
            if other > 0:
                proportions["__other__"] = other
            stats[str(column)] = {"type": "categorical", "proportions": proportions}
    return stats


def _cv_score(
    pipeline: Any,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    folds: int,
) -> tuple[float, float]:
    n_splits = max(2, min(folds, int(y.value_counts().min())))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=get_settings().random_state)
    try:
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Cross-validation failed: %s", exc)
        return float("nan"), float("nan")
    return float(np.mean(scores)), float(np.std(scores))


def train_models(
    df: pd.DataFrame,
    *,
    model_names: Sequence[str] | None = None,
    settings: Settings | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, ModelRun]:
    """Train the requested models on a cleaned dataframe.

    ``df`` must already be cleaned (numeric target encoded as 0/1). Feature
    engineering is applied here so training and inference share one code path.
    """
    cfg = settings or get_settings()
    if TARGET_COLUMN not in df.columns:
        raise TrainingError(f"Target column '{TARGET_COLUMN}' not found; run cleaning first.")

    names = list(model_names) if model_names else available_models()
    unknown = [name for name in names if name not in available_models()]
    if unknown:
        raise TrainingError(
            f"Unavailable model(s): {', '.join(unknown)}. Installed: {', '.join(available_models())}"
        )

    engineered = add_derived_features(df)
    y = engineered[TARGET_COLUMN].astype(int)
    if y.nunique() < 2:
        raise TrainingError("Target contains a single class; cannot train a classifier.")
    X = engineered.drop(columns=[TARGET_COLUMN])
    if "customerID" in X.columns:
        X = X.drop(columns=["customerID"])

    numeric, categorical = split_feature_types(X, target_column=TARGET_COLUMN)
    if not numeric and not categorical:
        raise TrainingError("No usable features found in the dataset.")

    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(
        X,
        y,
        test_size=cfg.test_size,
        validation_size=cfg.validation_size,
        random_state=cfg.random_state,
    )

    runs: dict[str, ModelRun] = {}
    for name in names:
        spec = get_model_spec(name)
        if progress:
            progress(f"Training {spec.display_name}...")
        started = time.perf_counter()

        pipeline = build_pipeline(
            spec.builder(cfg.random_state),
            numeric,
            categorical,
            scale_numeric=spec.requires_scaling,
        )
        pipeline.fit(X_train, y_train)
        val_scores = pipeline.predict_proba(X_val)[:, 1]
        threshold = optimal_threshold(y_val, val_scores)
        validation_metrics = evaluate_scores(y_val, val_scores, threshold, include_curves=False)

        # Refit on train + validation once the threshold is fixed.
        X_fit = pd.concat([X_train, X_val], axis=0)
        y_fit = pd.concat([y_train, y_val], axis=0)
        pipeline.fit(X_fit, y_fit)

        test_metrics = evaluate_pipeline(pipeline, X_test, y_test, threshold)
        cv_mean, cv_std = _cv_score(pipeline, X_fit, y_fit, folds=cfg.cv_folds)
        elapsed = time.perf_counter() - started

        runs[name] = ModelRun(
            name=name,
            display_name=spec.display_name,
            pipeline=pipeline,
            threshold=threshold,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
            cv_roc_auc=cv_mean,
            cv_roc_auc_std=cv_std,
            train_seconds=elapsed,
            model_columns=[str(column) for column in X.columns],
        )
        logger.info(
            "%s | threshold=%.3f | f1=%.3f | recall=%.3f | roc_auc=%.3f",
            spec.display_name,
            threshold,
            test_metrics["f1"],
            test_metrics["recall"],
            test_metrics["roc_auc"],
        )

    return runs


def select_best(runs: dict[str, ModelRun], *, metric: str = "f1") -> str:
    """Best model by test ``metric``, falling back to ROC AUC."""
    if not runs:
        raise TrainingError("No models were trained.")

    def score(run: ModelRun) -> tuple[float, float]:
        primary = run.test_metrics.get(metric) or 0.0
        secondary = run.test_metrics.get("roc_auc") or 0.0
        return float(primary), float(secondary)

    return max(runs.values(), key=score).name


def save_artifacts(
    runs: dict[str, ModelRun],
    *,
    best_name: str,
    cleaned_df: pd.DataFrame,
    reference_stats: dict[str, Any],
    cleaning_report: dict[str, Any] | None = None,
    dataset_info: dict[str, Any] | None = None,
    settings: Settings | None = None,
) -> dict[str, Path]:
    """Write model binaries, metrics, model card and reference stats to disk."""
    cfg = settings or get_settings()
    cfg.ensure_directories()

    trained_at = datetime.now(timezone.utc)
    run_id = trained_at.strftime("%Y.%m.%d-%H%M%S")
    defaults = infer_defaults(cleaned_df)
    paths: dict[str, Path] = {}

    for run in runs.values():
        version = f"{run.name}-{run_id}"
        payload = {
            "pipeline": run.pipeline,
            "model_name": run.name,
            "display_name": run.display_name,
            "version": version,
            "threshold": run.threshold,
            "trained_at": trained_at.isoformat(),
            "defaults": defaults,
            "model_columns": run.model_columns,
            "metrics": run.test_metrics,
            "validation_metrics": run.validation_metrics,
        }
        model_path = cfg.model_dir / f"{run.name}.joblib"
        joblib.dump(payload, model_path, compress=3)
        paths[run.name] = model_path

    best = runs[best_name]
    dataset_info = dataset_info or {}
    churn_rate = float(cleaned_df[TARGET_COLUMN].mean()) if TARGET_COLUMN in cleaned_df else None

    metrics_payload = {
        "run_id": run_id,
        "generated_at": trained_at.isoformat(),
        "best_model": best_name,
        "dataset": {
            "n_rows": int(cleaned_df.shape[0]),
            "n_features": int(best.model_columns.__len__()),
            "churn_rate": churn_rate,
            **dataset_info,
        },
        "cleaning": cleaning_report or {},
        "models": {
            run.name: {
                "display_name": run.display_name,
                "version": f"{run.name}-{run_id}",
                "threshold": round(run.threshold, 4),
                "validation": run.validation_metrics,
                "test": run.test_metrics,
                "cv_roc_auc": None if np.isnan(run.cv_roc_auc) else round(run.cv_roc_auc, 4),
                "cv_roc_auc_std": None
                if np.isnan(run.cv_roc_auc_std)
                else round(run.cv_roc_auc_std, 4),
                "train_seconds": round(run.train_seconds, 2),
            }
            for run in runs.values()
        },
        "comparison": comparison_table(
            {run.name: run.test_metrics for run in runs.values()}
        ),
        "features": {
            "model_columns": best.model_columns,
            "engineered_features": feature_names(best.pipeline),
        },
    }
    cfg.metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    paths["metrics"] = cfg.metrics_path

    model_card = {
        "model_name": best.name,
        "display_name": best.display_name,
        "version": f"{best.name}-{run_id}",
        "trained_at": trained_at.isoformat(),
        "algorithm": get_model_spec(best.name).description,
        "dataset": metrics_payload["dataset"],
        "decision_threshold": round(best.threshold, 4),
        "threshold_rationale": (
            "Threshold maximises F1 on a stratified validation split (20% of the "
            "training data); the test set is never used for tuning."
        ),
        "performance": {
            "validation": best.validation_metrics,
            "test": best.test_metrics,
            "cv_roc_auc": None if np.isnan(best.cv_roc_auc) else round(best.cv_roc_auc, 4),
        },
        "intended_use": (
            "Prioritise retention outreach for telecom customers. Output is a "
            "probability, not a causal claim: it must support, not replace, "
            "human retention decisions."
        ),
        "limitations": [
            "Trained on a public IBM Telco sample; production performance must be "
            "re-validated on the operator's own customer base.",
            "Labels are historical churn outcomes; the model cannot know about "
            "competitor offers or service outages that happened after the snapshot.",
            "Probability calibration is not guaranteed for tree ensembles; use the "
            "risk bands rather than the raw score for individual decisions.",
            "Monitoring drift measures input distribution change, not realised "
            "model accuracy; outcome labels arrive late and must be joined back.",
        ],
        "features_used": best.model_columns,
    }
    cfg.model_card_path.write_text(json.dumps(model_card, indent=2), encoding="utf-8")
    paths["model_card"] = cfg.model_card_path

    cfg.reference_stats_path.write_text(json.dumps(reference_stats, indent=2), encoding="utf-8")
    paths["reference_stats"] = cfg.reference_stats_path

    logger.info("Artifacts written to %s", cfg.artifact_dir)
    return paths


def train_and_save(
    df: pd.DataFrame,
    *,
    model_names: Sequence[str] | None = None,
    settings: Settings | None = None,
    cleaning_report: dict[str, Any] | None = None,
    dataset_info: dict[str, Any] | None = None,
    progress: Callable[[str], None] | None = None,
    metric: str = "f1",
) -> tuple[dict[str, ModelRun], str]:
    """Train, select the best model and persist all artifacts."""
    cfg = settings or get_settings()
    runs = train_models(df, model_names=model_names, settings=cfg, progress=progress)
    best_name = select_best(runs, metric=metric)

    engineered = add_derived_features(df).drop(columns=[TARGET_COLUMN])
    if "customerID" in engineered.columns:
        engineered = engineered.drop(columns=["customerID"])
    reference_stats = build_reference_stats(engineered)

    save_artifacts(
        runs,
        best_name=best_name,
        cleaned_df=df,
        reference_stats=reference_stats,
        cleaning_report=cleaning_report,
        dataset_info=dataset_info,
        settings=cfg,
    )
    return runs, best_name


__all__ = [
    "ModelRun",
    "TrainingError",
    "build_reference_stats",
    "infer_defaults",
    "optimal_threshold",
    "save_artifacts",
    "select_best",
    "train_and_save",
    "train_models",
]
