"""Batch and single-customer churn prediction (Feature 6).

The predictor loads the exported scikit-learn pipeline, fills any fields the
caller did not supply with the training-time defaults, and returns a churn
probability, a risk band and per-feature marginal contributions.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from churn_prediction.config import Settings, get_settings, risk_level
from churn_prediction.features.engineering import add_derived_features
from churn_prediction.features.fields import to_model_columns
from churn_prediction.logging_config import get_logger

logger = get_logger(__name__)

MODEL_DIR_NAME = "models"


class ModelNotTrainedError(FileNotFoundError):
    """Raised when no exported model artifact can be found."""


#: Friendlier names used by the explanation list in the API and dashboard.
DISPLAY_NAMES = {
    "tenure": "Tenure (months)",
    "MonthlyCharges": "Monthly charges",
    "TotalCharges": "Total charges",
    "Contract": "Contract type",
    "InternetService": "Internet service",
    "PaymentMethod": "Payment method",
    "OnlineSecurity": "Online security add-on",
    "TechSupport": "Tech support add-on",
    "DeviceProtection": "Device protection add-on",
    "OnlineBackup": "Online backup add-on",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming movies",
    "PaperlessBilling": "Paperless billing",
    "Partner": "Has partner",
    "Dependents": "Has dependents",
    "SeniorCitizen": "Senior citizen",
    "PhoneService": "Phone service",
    "MultipleLines": "Multiple lines",
    "gender": "Gender",
    "tenure_group": "Tenure group",
    "contract_months": "Contract length (months)",
    "is_monthly_contract": "Month-to-month contract",
    "avg_charge_per_month": "Average charge per month",
    "spend_gap_ratio": "Spend vs expected",
    "service_count": "Number of services",
    "addon_count": "Protection add-ons",
    "streaming_count": "Streaming services",
    "family_size": "Family members on plan",
    "has_support": "Has support bundle",
    "autopay": "Automatic payment",
    "fiber_customer": "Fibre customer",
}


@dataclass
class Prediction:
    prediction: str
    churn_probability: float
    risk_level: str
    threshold: float
    model_name: str
    model_version: str
    latency_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FeatureImpact:
    feature: str
    label: str
    value: Any
    impact: float

    @property
    def direction(self) -> str:
        return "increases risk" if self.impact > 0 else "reduces risk"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["direction"] = self.direction
        return data


class ChurnPredictor:
    """Loads a trained pipeline and serves predictions and explanations."""

    def __init__(
        self,
        model_name: str | None = None,
        *,
        settings: Settings | None = None,
        artifact_dir: str | Path | None = None,
    ) -> None:
        cfg = settings or get_settings()
        if artifact_dir is not None:
            cfg = cfg.with_artifact_dir(artifact_dir)
        self.settings = cfg

        model_dir = cfg.model_dir
        if not model_dir.exists():
            raise ModelNotTrainedError(
                f"No model directory at '{model_dir}'. Run `python scripts/train.py` first."
            )

        resolved_name = model_name or self._resolve_default_model()
        model_path = model_dir / f"{resolved_name}.joblib"
        if not model_path.exists():
            raise ModelNotTrainedError(
                f"Model '{resolved_name}' not found at '{model_path}'. Train it with "
                f"`python scripts/train.py --models {resolved_name}`."
            )

        payload = joblib.load(model_path)
        self.pipeline = payload["pipeline"]
        self.model_name: str = payload.get("model_name", resolved_name)
        self.display_name: str = payload.get("display_name", self.model_name)
        self.version: str = payload.get("version", "unknown")
        self.threshold: float = float(payload.get("threshold", 0.5))
        self.trained_at: str | None = payload.get("trained_at")
        self.metrics: dict[str, Any] = payload.get("metrics", {})
        self.defaults: dict[str, Any] = dict(payload.get("defaults", {}))
        self.model_columns: list[str] = list(payload.get("model_columns", []))
        logger.info("Loaded model %s (%s)", self.model_name, self.version)

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def trained_models(settings: Settings | None = None) -> list[str]:
        cfg = settings or get_settings()
        if not cfg.model_dir.exists():
            return []
        return sorted(path.stem for path in cfg.model_dir.glob("*.joblib"))

    def _resolve_default_model(self) -> str:
        trained = self.trained_models(self.settings)
        if not trained:
            raise ModelNotTrainedError(
                f"No .joblib models in '{self.settings.model_dir}'. "
                "Run `python scripts/train.py` first."
            )
        preferred = self.settings.default_model
        if preferred in trained:
            return preferred
        # Fall back to the best model recorded in metrics.json.
        import json

        if self.settings.metrics_path.exists():
            try:
                payload = json.loads(self.settings.metrics_path.read_text(encoding="utf-8"))
                best = payload.get("best_model")
                if best in trained:
                    return best
            except (ValueError, OSError):  # pragma: no cover - corrupted metrics
                logger.warning("Could not parse %s", self.settings.metrics_path)
        return trained[0]

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    def _prepare(self, payload: Mapping[str, Any] | None) -> dict[str, Any]:
        row = dict(self.defaults)
        normalised = to_model_columns(payload)
        for key, value in normalised.items():
            if key not in row:
                continue
            default = row[key]
            if value is None:
                continue
            if isinstance(default, (int, float)) and not isinstance(default, bool):
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    logger.debug("Ignoring non numeric value for %s: %r", key, value)
                    continue
                value = int(numeric) if isinstance(default, int) else numeric
            else:
                value = str(value)
            row[key] = value
        return row

    def _score(self, row: Mapping[str, Any]) -> float:
        frame = pd.DataFrame([dict(row)])
        frame = add_derived_features(frame)
        if self.model_columns:
            frame = frame.reindex(columns=self.model_columns)
        probabilities = self.pipeline.predict_proba(frame)[:, 1]
        return float(probabilities[0])

    def predict(self, payload: Mapping[str, Any] | None = None) -> Prediction:
        """Predict churn for a single customer payload."""
        started = time.perf_counter()
        probability = self._score(self._prepare(payload))
        latency_ms = (time.perf_counter() - started) * 1000
        return Prediction(
            prediction="Churn" if probability >= self.threshold else "No Churn",
            churn_probability=round(probability, 4),
            risk_level=risk_level(probability, self.settings),
            threshold=round(self.threshold, 4),
            model_name=self.model_name,
            model_version=self.version,
            latency_ms=round(latency_ms, 2),
        )

    def explain(
        self,
        payload: Mapping[str, Any] | None = None,
        *,
        top_k: int = 5,
    ) -> list[FeatureImpact]:
        """Marginal effect of each supplied field versus the typical customer.

        For every field whose value differs from the training-time default we
        flip only that field back to the default and measure how much the churn
        probability moves. This is model agnostic (works for linear models,
        forests and boosted trees) and cheap enough to run per request.
        """
        row = self._prepare(payload)
        baseline_probability = self._score(row)

        impacts: list[FeatureImpact] = []
        for column, value in row.items():
            default = self.defaults.get(column)
            if default is None or value == default:
                continue
            baseline_row = dict(row)
            baseline_row[column] = default
            delta = baseline_probability - self._score(baseline_row)
            if abs(delta) < 1e-6:
                continue
            impacts.append(
                FeatureImpact(
                    feature=column,
                    label=DISPLAY_NAMES.get(column, column),
                    value=value,
                    impact=round(float(delta), 4),
                )
            )

        impacts.sort(key=lambda item: abs(item.impact), reverse=True)
        return impacts[:top_k]

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def info(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "display_name": self.display_name,
            "version": self.version,
            "trained_at": self.trained_at,
            "threshold": round(self.threshold, 4),
            "risk_bands": {
                "low_below": self.settings.low_risk_max,
                "high_from": self.settings.high_risk_min,
            },
            "features": self.model_columns,
            "metrics": self.metrics,
            "available_models": self.trained_models(self.settings),
        }


__all__ = [
    "ChurnPredictor",
    "DISPLAY_NAMES",
    "FeatureImpact",
    "ModelNotTrainedError",
    "Prediction",
]
