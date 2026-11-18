"""Population Stability Index (PSI) based drift detection.

PSI compares the distribution of a feature at training time with the
distribution observed in live traffic. It answers "has the world moved?", not
"is the model still accurate?" - realised accuracy needs labelled outcomes
which arrive weeks later.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from churn_prediction.features.engineering import add_derived_features

#: Conventional PSI interpretation thresholds.
PSI_STABLE = 0.10
PSI_MODERATE = 0.25

_EPSILON = 1e-6


def population_stability_index(
    expected: Sequence[float],
    actual: Sequence[float],
    *,
    epsilon: float = _EPSILON,
) -> float:
    """Compute PSI between two proportion vectors of the same length."""
    expected_arr = np.clip(np.asarray(expected, dtype=float), epsilon, None)
    actual_arr = np.clip(np.asarray(actual, dtype=float), epsilon, None)
    if expected_arr.shape != actual_arr.shape:
        raise ValueError("Expected and actual distributions must have equal length.")
    return float(np.sum((actual_arr - expected_arr) * np.log(actual_arr / expected_arr)))


def drift_status(score: float) -> str:
    """Map a PSI score to a human readable status."""
    if score < PSI_STABLE:
        return "stable"
    if score < PSI_MODERATE:
        return "moderate"
    return "significant"


def _numeric_actual_proportions(values: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    edges_arr = np.asarray(edges, dtype=float)
    clipped = np.clip(values, edges_arr[0], edges_arr[-1])
    counts, _ = np.histogram(clipped, bins=edges_arr)
    total = counts.sum()
    if total == 0:
        return np.zeros(len(counts), dtype=float)
    return counts / total


def _categorical_actual_proportions(
    series: pd.Series, expected: Mapping[str, float]
) -> np.ndarray:
    observed = series.astype(str).value_counts(normalize=True)
    actual: list[float] = []
    for category in expected:
        if category == "__other__":
            known = [name for name in expected if name != "__other__"]
            actual.append(float(observed.drop(labels=known, errors="ignore").sum()))
        else:
            actual.append(float(observed.get(category, 0.0)))
    return np.asarray(actual, dtype=float)


def compute_drift(
    reference_stats: Mapping[str, Any],
    payloads: Sequence[Mapping[str, Any]],
    *,
    min_samples: int = 20,
) -> dict[str, Any]:
    """Compare live prediction payloads against the training reference.

    Returns an overall verdict plus a per-feature breakdown sorted by PSI.
    """
    n_samples = len(payloads)
    if n_samples < min_samples:
        return {
            "status": "insufficient_data",
            "n_samples": n_samples,
            "min_samples": min_samples,
            "overall_psi": None,
            "max_psi": None,
            "features": [],
        }

    frame = add_derived_features(pd.DataFrame(list(payloads)))
    features: list[dict[str, Any]] = []

    for name, reference in reference_stats.items():
        if name not in frame.columns:
            continue
        kind = reference.get("type")
        if kind == "numeric":
            edges = reference.get("edges") or []
            expected = reference.get("proportions") or []
            if len(edges) < 2 or not expected:
                continue
            values = pd.to_numeric(frame[name], errors="coerce").dropna().to_numpy(dtype=float)
            if values.size == 0:
                continue
            actual = _numeric_actual_proportions(values, edges)
        elif kind == "categorical":
            expected_map = reference.get("proportions") or {}
            if not expected_map:
                continue
            expected = list(expected_map.values())
            actual = _categorical_actual_proportions(frame[name], expected_map)
        else:  # pragma: no cover - defensive
            continue

        score = population_stability_index(expected, actual)
        features.append(
            {
                "feature": name,
                "type": kind,
                "psi": round(score, 4),
                "status": drift_status(score),
            }
        )

    features.sort(key=lambda item: item["psi"], reverse=True)
    scores = [item["psi"] for item in features]
    overall = float(np.mean(scores)) if scores else 0.0
    worst = max(scores) if scores else 0.0

    return {
        "status": drift_status(overall),
        "n_samples": n_samples,
        "min_samples": min_samples,
        "overall_psi": round(overall, 4),
        "max_psi": round(worst, 4),
        "features": features,
    }


__all__ = [
    "PSI_MODERATE",
    "PSI_STABLE",
    "compute_drift",
    "drift_status",
    "population_stability_index",
]
