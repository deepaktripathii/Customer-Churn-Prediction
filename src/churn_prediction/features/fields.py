"""Mapping between the public API field names and dataset column names.

Keeping the mapping in a dependency-free module avoids a circular import
between the API schemas and the predictor: both import from here.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: public (snake_case) name -> dataset column name
MODEL_FIELD_MAP: dict[str, str] = {
    "gender": "gender",
    "senior_citizen": "SeniorCitizen",
    "partner": "Partner",
    "dependents": "Dependents",
    "tenure": "tenure",
    "phone_service": "PhoneService",
    "multiple_lines": "MultipleLines",
    "internet_service": "InternetService",
    "online_security": "OnlineSecurity",
    "online_backup": "OnlineBackup",
    "device_protection": "DeviceProtection",
    "tech_support": "TechSupport",
    "streaming_tv": "StreamingTV",
    "streaming_movies": "StreamingMovies",
    "contract": "Contract",
    "paperless_billing": "PaperlessBilling",
    "payment_method": "PaymentMethod",
    "monthly_charges": "MonthlyCharges",
    "total_charges": "TotalCharges",
}

MODEL_COLUMNS: frozenset[str] = frozenset(MODEL_FIELD_MAP.values())


def to_model_columns(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalise a payload that may use API names, dataset names, or both."""
    normalised: dict[str, Any] = {}
    for key, value in (payload or {}).items():
        if key in MODEL_COLUMNS:
            normalised[key] = value
            continue
        mapped = MODEL_FIELD_MAP.get(str(key))
        if mapped is not None:
            normalised[mapped] = value
    return normalised


__all__ = ["MODEL_COLUMNS", "MODEL_FIELD_MAP", "to_model_columns"]
