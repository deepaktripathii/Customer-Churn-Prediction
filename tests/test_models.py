"""Features 4, 5 and 6: training, evaluation and prediction."""

from __future__ import annotations

import json

import numpy as np
import pytest

from retainiq.models.evaluate import (
    classification_metrics,
    comparison_table,
    evaluate_scores,
)
from retainiq.models.predictor import ChurnPredictor, ModelNotTrainedError
from retainiq.models.registry import available_models
from retainiq.models.train import (
    infer_defaults,
    optimal_threshold,
    train_models,
)

SAMPLE_CUSTOMER = {
    "tenure": 3,
    "MonthlyCharges": 1200.0,
    "TotalCharges": 3600.0,
    "Contract": "Month-to-month",
    "InternetService": "Fiber optic",
    "PaymentMethod": "Electronic check",
}


def test_registry_contains_required_models():
    models = available_models()

    assert "logistic_regression" in models
    assert "random_forest" in models


def test_optimal_threshold_is_bounded():
    rng = np.random.default_rng(0)
    scores = rng.random(200)
    labels = (scores + rng.normal(0, 0.2, 200) > 0.6).astype(int)

    threshold = optimal_threshold(labels, scores)

    assert 0.05 <= threshold <= 0.95


def test_metrics_bundle_contains_expected_keys():
    y_true = [0, 0, 1, 1, 1, 0]
    y_score = [0.1, 0.2, 0.8, 0.7, 0.4, 0.3]

    metrics = classification_metrics(y_true, y_score, threshold=0.5)

    for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
        assert key in metrics
    assert metrics["confusion_matrix"]["true_positive"] == 2
    assert metrics["support"]["total"] == 6


def test_metrics_handle_single_class():
    metrics = classification_metrics([0, 0, 0], [0.1, 0.2, 0.3], threshold=0.5)

    assert metrics["roc_auc"] == 0.5
    assert metrics["pr_auc"] == 0.0


def test_curves_are_json_friendly():
    rng = np.random.default_rng(1)
    scores = rng.random(50)
    labels = (scores > 0.5).astype(int)

    payload = evaluate_scores(labels, scores, threshold=0.5)

    assert payload["roc_curve"]
    assert set(payload["roc_curve"][0]) == {"fpr", "tpr"}
    assert set(payload["pr_curve"][0]) == {"recall", "precision"}


def test_comparison_table_sorts_descending():
    table = comparison_table(
        {
            "a": {"f1": 0.5, "recall": 0.5, "precision": 0.5, "accuracy": 0.7, "roc_auc": 0.8, "pr_auc": 0.6, "threshold": 0.5},
            "b": {"f1": 0.8, "recall": 0.8, "precision": 0.7, "accuracy": 0.8, "roc_auc": 0.9, "pr_auc": 0.7, "threshold": 0.4},
        }
    )

    assert table[0]["model"] == "b"


def test_train_models_returns_metrics(cleaned_frame):
    runs = train_models(cleaned_frame, model_names=["logistic_regression"])

    run = runs["logistic_regression"]
    assert 0.0 <= run.threshold <= 1.0
    assert set(run.test_metrics) >= {"f1", "recall", "roc_auc", "confusion_matrix"}
    assert len(run.model_columns) > 5


def test_infer_defaults_roundtrips_types(cleaned_frame):
    defaults = infer_defaults(cleaned_frame)

    assert "Contract" in defaults
    assert isinstance(defaults["tenure"], (int, float))
    assert isinstance(defaults["Contract"], str)
    assert "Churn" not in defaults


def test_artifacts_and_predictor(trained_settings):
    assert trained_settings.metrics_path.exists()
    assert trained_settings.model_card_path.exists()
    assert trained_settings.reference_stats_path.exists()

    metrics = json.loads(trained_settings.metrics_path.read_text(encoding="utf-8"))
    assert metrics["best_model"] == "logistic_regression"
    assert metrics["models"]["logistic_regression"]["test"]["roc_auc"] >= 0.5

    predictor = ChurnPredictor(settings=trained_settings)
    prediction = predictor.predict(SAMPLE_CUSTOMER)

    assert prediction.prediction in {"Churn", "No Churn"}
    assert 0.0 <= prediction.churn_probability <= 1.0
    assert prediction.risk_level in {"Low", "Medium", "High"}
    assert prediction.model_version.startswith("logistic_regression-")


def test_predictor_accepts_partial_payload(trained_settings):
    predictor = ChurnPredictor(settings=trained_settings)
    prediction = predictor.predict({"tenure": 1, "monthly_charges": 50, "contract": "Two year"})

    assert 0.0 <= prediction.churn_probability <= 1.0


def test_explanations_are_ranked(trained_settings):
    predictor = ChurnPredictor(settings=trained_settings)
    impacts = predictor.explain(SAMPLE_CUSTOMER, top_k=3)

    assert impacts
    absolute = [abs(item.impact) for item in impacts]
    assert absolute == sorted(absolute, reverse=True)


def test_explanations_accept_api_field_names(trained_settings):
    predictor = ChurnPredictor(settings=trained_settings)
    impacts = predictor.explain(
        {
            "tenure": 2,
            "monthly_charges": 1200,
            "contract": "Month-to-month",
        }
    )

    features = {item.feature for item in impacts}
    assert "tenure" in features
    assert "MonthlyCharges" in features


def test_missing_model_raises(tmp_path):
    with pytest.raises(ModelNotTrainedError):
        ChurnPredictor(artifact_dir=tmp_path)


def test_predictor_info(trained_settings):
    predictor = ChurnPredictor(settings=trained_settings)
    info = predictor.info()

    assert info["model_name"] == "logistic_regression"
    assert info["threshold"] == pytest.approx(round(predictor.threshold, 4))
    assert info["available_models"]


def test_predictions_respect_threshold(trained_settings):
    predictor = ChurnPredictor(settings=trained_settings)
    prediction = predictor.predict(SAMPLE_CUSTOMER)

    expected = "Churn" if prediction.churn_probability >= predictor.threshold else "No Churn"
    assert prediction.prediction == expected
