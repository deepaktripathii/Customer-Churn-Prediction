"""Model evaluation and comparison (Feature 5).

Accuracy alone is misleading for churn: only ~27% of customers churn, so a
model that always predicts "retained" is 73% accurate and completely useless.
The metrics computed here make the recall/precision trade-off explicit, and
the decision threshold is chosen on a validation split instead of being left
at the arbitrary 0.5 default.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _safe_auc(metric: Any, y_true: np.ndarray, y_score: np.ndarray, default: float) -> float:
    if len(np.unique(y_true)) < 2:
        return default
    return float(metric(y_true, y_score))


def classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute the full metric bundle at a given decision threshold."""
    y_true_arr = np.asarray(y_true).astype(int)
    y_score_arr = np.asarray(y_score, dtype=float)
    y_pred = (y_score_arr >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()

    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(accuracy_score(y_true_arr, y_pred)), 4),
        "precision": round(
            float(precision_score(y_true_arr, y_pred, zero_division=0)), 4
        ),
        "recall": round(float(recall_score(y_true_arr, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true_arr, y_pred, zero_division=0)), 4),
        "roc_auc": round(
            _safe_auc(roc_auc_score, y_true_arr, y_score_arr, 0.5), 4
        ),
        "pr_auc": round(
            _safe_auc(average_precision_score, y_true_arr, y_score_arr, 0.0), 4
        ),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "support": {
            "retained": int((y_true_arr == 0).sum()),
            "churned": int((y_true_arr == 1).sum()),
            "total": int(len(y_true_arr)),
        },
    }


def _subsample(points: np.ndarray, max_points: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    idx = np.linspace(0, len(points) - 1, max_points).astype(int)
    return points[idx]


def roc_curve_points(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    *,
    max_points: int = 120,
) -> list[dict[str, float]]:
    """ROC curve as JSON friendly points (for the dashboard)."""
    y_true_arr = np.asarray(y_true).astype(int)
    if len(np.unique(y_true_arr)) < 2:
        return []
    fpr, tpr, _ = roc_curve(y_true_arr, np.asarray(y_score, dtype=float))
    points = np.column_stack([fpr, tpr])
    points = _subsample(points, max_points)
    return [{"fpr": round(float(x), 4), "tpr": round(float(y), 4)} for x, y in points]


def pr_curve_points(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    *,
    max_points: int = 120,
) -> list[dict[str, float]]:
    """Precision-recall curve as JSON friendly points."""
    y_true_arr = np.asarray(y_true).astype(int)
    if len(np.unique(y_true_arr)) < 2:
        return []
    precision, recall, _ = precision_recall_curve(y_true_arr, np.asarray(y_score, dtype=float))
    points = np.column_stack([recall[::-1], precision[::-1]])
    points = _subsample(points, max_points)
    return [
        {"recall": round(float(x), 4), "precision": round(float(y), 4)} for x, y in points
    ]


def evaluate_scores(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    threshold: float,
    *,
    include_curves: bool = True,
) -> dict[str, Any]:
    """Metrics plus optional curve data for a set of predicted probabilities."""
    metrics = classification_metrics(y_true, y_score, threshold)
    if include_curves:
        metrics["roc_curve"] = roc_curve_points(y_true, y_score)
        metrics["pr_curve"] = pr_curve_points(y_true, y_score)
    return metrics


def evaluate_pipeline(
    pipeline: Any,
    X: pd.DataFrame,
    y: Sequence[int] | pd.Series,
    threshold: float,
    *,
    include_curves: bool = True,
) -> dict[str, Any]:
    """Evaluate a fitted pipeline on a feature frame."""
    scores = pipeline.predict_proba(X)[:, 1]
    return evaluate_scores(y, scores, threshold, include_curves=include_curves)


COMPARISON_COLUMNS = (
    "model",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "threshold",
)


def comparison_table(
    results: Mapping[str, Mapping[str, Any]],
    *,
    sort_by: str = "f1",
) -> list[dict[str, Any]]:
    """Flatten ``{model: metrics}`` into a sorted table for the dashboard."""
    rows: list[dict[str, Any]] = []
    for model_name, metrics in results.items():
        row = {"model": model_name}
        for column in COMPARISON_COLUMNS[1:]:
            row[column] = metrics.get(column)
        rows.append(row)

    def sort_key(row: dict[str, Any]) -> float:
        value = row.get(sort_by)
        return float(value) if isinstance(value, (int, float)) else -1.0

    return sorted(rows, key=sort_key, reverse=True)


__all__ = [
    "COMPARISON_COLUMNS",
    "classification_metrics",
    "comparison_table",
    "evaluate_pipeline",
    "evaluate_scores",
    "pr_curve_points",
    "roc_curve_points",
]
