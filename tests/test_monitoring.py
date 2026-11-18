"""Feature 7: monitoring store and drift detection."""

from __future__ import annotations

import numpy as np
import pytest

from churn_prediction.api.schemas import CustomerFeatures
from churn_prediction.features.engineering import add_derived_features
from churn_prediction.models.train import build_reference_stats
from churn_prediction.monitoring.drift import (
    compute_drift,
    drift_status,
    population_stability_index,
)
from churn_prediction.monitoring.store import MonitoringStore


def _store(tmp_path) -> MonitoringStore:
    return MonitoringStore(f"sqlite:///{(tmp_path / 'monitoring.db').as_posix()}")


def _log(store: MonitoringStore, probability: float, risk: str = "Low") -> None:
    store.log_prediction(
        payload={"tenure": 5, "MonthlyCharges": 90.0, "Contract": "One year"},
        churn_probability=probability,
        prediction="Churn" if probability >= 0.5 else "No Churn",
        risk_level=risk,
        model_name="logistic_regression",
        model_version="logistic_regression-1",
        latency_ms=12.5,
    )


def test_store_roundtrip(tmp_path):
    store = _store(tmp_path)
    _log(store, 0.2)
    _log(store, 0.8, risk="High")

    assert store.count() == 2
    recent = store.recent(limit=5)
    assert recent[0]["churn_probability"] == 0.8
    assert store.payloads(limit=5)


def test_summary_aggregates(tmp_path):
    store = _store(tmp_path)
    _log(store, 0.2, risk="Low")
    _log(store, 0.4, risk="Medium")
    _log(store, 0.9, risk="High")

    summary = store.summary()

    assert summary["total_predictions"] == 3
    assert summary["risk_distribution"] == {"Low": 1, "Medium": 1, "High": 1}
    assert summary["avg_latency_ms"] == pytest.approx(12.5)
    assert summary["daily"][0]["count"] == 3


def test_empty_summary(tmp_path):
    summary = _store(tmp_path).summary()

    assert summary["total_predictions"] == 0
    assert summary["daily"] == []


def test_reset(tmp_path):
    store = _store(tmp_path)
    _log(store, 0.3)
    store.reset()

    assert store.count() == 0


def test_psi_is_zero_for_identical_distributions():
    values = np.array([0.2, 0.3, 0.5])

    assert population_stability_index(values, values) == pytest.approx(0.0, abs=1e-9)


def test_psi_detects_shift():
    expected = np.array([0.5, 0.3, 0.2])
    actual = np.array([0.1, 0.2, 0.7])

    assert population_stability_index(expected, actual) > 0.2


def test_psi_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        population_stability_index([0.5, 0.5], [1.0])


@pytest.mark.parametrize(
    ("score", "expected"),
    [(0.05, "stable"), (0.15, "moderate"), (0.4, "significant")],
)
def test_drift_status_bands(score, expected):
    assert drift_status(score) == expected


def test_compute_drift_needs_samples():
    result = compute_drift({}, [], min_samples=10)

    assert result["status"] == "insufficient_data"
    assert result["features"] == []


def test_compute_drift_against_training_reference(cleaned_frame):
    features = add_derived_features(cleaned_frame).drop(columns=["Churn"])
    reference = build_reference_stats(features)
    payloads = cleaned_frame.drop(columns=["Churn"]).to_dict(orient="records")

    stable = compute_drift(reference, payloads, min_samples=20)

    assert stable["status"] == "stable"
    assert stable["overall_psi"] < 0.05

    shifted = []
    for row in payloads:
        copy = dict(row)
        copy["MonthlyCharges"] = float(copy["MonthlyCharges"]) * 3
        shifted.append(copy)
    drifted = compute_drift(reference, shifted, min_samples=20)

    monthly = next(item for item in drifted["features"] if item["feature"] == "MonthlyCharges")
    assert monthly["status"] == "significant"
    assert drifted["max_psi"] > 0.25


def test_schema_maps_api_fields_to_dataset_columns():
    payload = CustomerFeatures(
        tenure=3, monthly_charges=1200, contract="Month-to-month"
    ).to_model_payload()

    assert payload["MonthlyCharges"] == 1200
    assert payload["Contract"] == "Month-to-month"
    assert payload["TotalCharges"] == pytest.approx(3600.0)
