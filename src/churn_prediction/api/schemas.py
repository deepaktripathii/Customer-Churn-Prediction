"""Pydantic schemas for the Customer Churn Prediction API.

The public API uses ``snake_case`` names that are friendly to front-end code,
while the trained pipeline expects the original dataset column names. The
mapping lives in :data:`MODEL_FIELD_MAP` and is applied by
:meth:`CustomerFeatures.to_model_payload`.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from churn_prediction.features.fields import MODEL_FIELD_MAP

YesNo = Literal["Yes", "No"]

SAMPLE_CUSTOMER = {
    "tenure": 3,
    "monthly_charges": 1200,
    "total_charges": 3600,
    "contract": "Month-to-month",
    "internet_service": "Fiber optic",
    "payment_method": "Electronic check",
    "online_security": "No",
    "tech_support": "No",
}


class CustomerFeatures(BaseModel):
    """A single customer record. Only three fields are mandatory.

    Any omitted field is filled with the training-time median (numeric) or mode
    (categorical), so the endpoint can be called with a partial profile.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": SAMPLE_CUSTOMER},
    )

    tenure: int = Field(
        ..., ge=0, le=120, description="Months the customer has stayed with the company."
    )
    monthly_charges: float = Field(
        ..., gt=0, le=2000, description="Current monthly bill in the operator's currency."
    )
    contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        ..., description="Contract commitment length."
    )
    total_charges: float | None = Field(
        default=None, ge=0, description="Lifetime billed amount. Defaults to tenure x monthly."
    )
    internet_service: Literal["DSL", "Fiber optic", "No"] | None = None
    payment_method: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] | None = None

    gender: Literal["Male", "Female"] | None = None
    senior_citizen: YesNo | None = None
    partner: YesNo | None = None
    dependents: YesNo | None = None
    phone_service: YesNo | None = None
    multiple_lines: Literal["Yes", "No", "No phone service"] | None = None
    online_security: Literal["Yes", "No", "No internet service"] | None = None
    online_backup: Literal["Yes", "No", "No internet service"] | None = None
    device_protection: Literal["Yes", "No", "No internet service"] | None = None
    tech_support: Literal["Yes", "No", "No internet service"] | None = None
    streaming_tv: Literal["Yes", "No", "No internet service"] | None = None
    streaming_movies: Literal["Yes", "No", "No internet service"] | None = None
    paperless_billing: YesNo | None = None

    @field_validator("payment_method", "internet_service", mode="before")
    @classmethod
    def _blank_to_none(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    def to_model_payload(self) -> dict[str, Any]:
        """Convert to the dataset column names used by the pipeline."""
        data = self.model_dump(exclude_none=True)
        payload = {MODEL_FIELD_MAP[key]: value for key, value in data.items() if key in MODEL_FIELD_MAP}
        if "TotalCharges" not in payload:
            payload["TotalCharges"] = float(self.tenure) * float(self.monthly_charges)
        return payload


class FeatureImpact(BaseModel):
    feature: str
    label: str
    value: Any = None
    direction: Literal["increases risk", "reduces risk"]
    impact: float = Field(description="Change in churn probability versus a typical customer.")


class PredictionResponse(BaseModel):
    prediction: Literal["Churn", "No Churn"]
    churn_probability: float = Field(ge=0, le=1)
    risk_level: Literal["Low", "Medium", "High"]
    threshold: float = Field(ge=0, le=1)
    model_name: str
    model_version: str
    latency_ms: float
    explanations: list[FeatureImpact] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_name: str | None = None
    model_version: str | None = None
    database: Literal["ok", "unavailable"]
    predictions_logged: int = 0
    version: str
    timestamp: str


class ModelSummary(BaseModel):
    name: str
    display_name: str
    family: str
    description: str


class ModelInfoResponse(BaseModel):
    model_name: str
    display_name: str
    version: str
    trained_at: str | None
    threshold: float
    risk_bands: dict[str, float]
    features: list[str]
    metrics: dict[str, Any]
    available_models: list[str]
    registry: list[ModelSummary]


class PredictionLogEntry(BaseModel):
    id: int
    created_at: str | None
    model_name: str
    model_version: str
    prediction: str
    risk_level: str
    churn_probability: float
    latency_ms: float
    payload: dict[str, Any]


class MonitoringSummary(BaseModel):
    total_predictions: int
    window_size: int
    churn_predictions: int
    churn_rate: float | None
    avg_churn_probability: float | None
    risk_distribution: dict[str, int]
    model_versions: dict[str, int]
    avg_latency_ms: float | None
    p95_latency_ms: float | None
    first_prediction_at: str | None
    last_prediction_at: str | None
    predictions_last_24h: int
    daily: list[dict[str, Any]]


class DriftFeature(BaseModel):
    feature: str
    type: str
    psi: float
    status: Literal["stable", "moderate", "significant"]


class DriftResponse(BaseModel):
    status: Literal["stable", "moderate", "significant", "insufficient_data"]
    n_samples: int
    min_samples: int
    overall_psi: float | None
    max_psi: float | None
    features: list[DriftFeature]


__all__ = [
    "CustomerFeatures",
    "DriftFeature",
    "DriftResponse",
    "FeatureImpact",
    "HealthResponse",
    "MODEL_FIELD_MAP",
    "ModelInfoResponse",
    "ModelSummary",
    "MonitoringSummary",
    "PredictionLogEntry",
    "PredictionResponse",
    "SAMPLE_CUSTOMER",
]
