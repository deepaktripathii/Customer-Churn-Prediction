"""Feature engineering (Feature 3).

Cleaning fixes broken values; feature engineering creates *new* columns that
encode domain knowledge about customer retention. A month-to-month fibre
customer who has been with the company for two months is very different from a
customer on a two year contract in year five, and these derived ratios make
that difference explicit for the models.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from churn_prediction.config import ID_COLUMN, TARGET_COLUMN
from churn_prediction.logging_config import get_logger

logger = get_logger(__name__)

#: Tenure buckets used by the marketing team for lifecycle campaigns.
TENURE_BINS = [-1, 12, 24, 48, 72]
TENURE_LABELS = ["0-12m", "13-24m", "25-48m", "49-72m"]

#: Contract length in months, used to derive commitment features.
CONTRACT_MONTHS = {"Month-to-month": 1, "One year": 12, "Two year": 24}

SERVICE_COLUMNS = (
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
)
ADDON_COLUMNS = (
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
)
STREAMING_COLUMNS = ("StreamingTV", "StreamingMovies")

DERIVED_COLUMNS = (
    "tenure_group",
    "contract_months",
    "is_monthly_contract",
    "avg_charge_per_month",
    "expected_total_charges",
    "spend_gap_ratio",
    "service_count",
    "addon_count",
    "streaming_count",
    "family_size",
    "has_support",
    "autopay",
    "fiber_customer",
)


def _yes_count(df: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    present = [column for column in columns if column in df.columns]
    if not present:
        return pd.Series(0, index=df.index, dtype="int64")
    return (df[present] == "Yes").sum(axis=1).astype("int64")


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with engineered churn features appended.

    Every block is guarded by a column-existence check so the same function can
    be reused by the API for a partially specified customer payload.
    """
    out = df.copy()

    tenure = pd.to_numeric(out.get("tenure", pd.Series(np.nan, index=out.index)), errors="coerce")
    monthly = pd.to_numeric(
        out.get("MonthlyCharges", pd.Series(np.nan, index=out.index)), errors="coerce"
    )
    total = pd.to_numeric(
        out.get("TotalCharges", pd.Series(np.nan, index=out.index)), errors="coerce"
    )

    if "tenure" in out.columns:
        out["tenure_group"] = pd.cut(
            tenure,
            bins=TENURE_BINS,
            labels=TENURE_LABELS,
            include_lowest=True,
            right=True,
        ).astype(object)

    if "Contract" in out.columns:
        out["contract_months"] = (
            out["Contract"].map(CONTRACT_MONTHS).astype("float64").fillna(1.0)
        )
        out["is_monthly_contract"] = (out["Contract"] == "Month-to-month").astype("int8")

    if "MonthlyCharges" in out.columns and "TotalCharges" in out.columns:
        safe_tenure = tenure.clip(lower=1)
        avg = total / safe_tenure
        # New customers only have one bill, so fall back to the monthly charge.
        out["avg_charge_per_month"] = np.where(tenure > 0, avg, monthly)
        out["expected_total_charges"] = safe_tenure * monthly
        denominator = out["expected_total_charges"].abs().clip(lower=1.0)
        out["spend_gap_ratio"] = (total - out["expected_total_charges"]) / denominator

    if any(column in out.columns for column in SERVICE_COLUMNS):
        out["service_count"] = _yes_count(out, SERVICE_COLUMNS)
    if any(column in out.columns for column in ADDON_COLUMNS):
        out["addon_count"] = _yes_count(out, ADDON_COLUMNS)
    if any(column in out.columns for column in STREAMING_COLUMNS):
        out["streaming_count"] = _yes_count(out, STREAMING_COLUMNS)

    if "Partner" in out.columns or "Dependents" in out.columns:
        out["family_size"] = _yes_count(out, ("Partner", "Dependents"))

    if "TechSupport" in out.columns or "OnlineSecurity" in out.columns:
        out["has_support"] = (_yes_count(out, ("TechSupport", "OnlineSecurity")) > 0).astype(
            "int8"
        )

    if "PaymentMethod" in out.columns:
        out["autopay"] = (
            out["PaymentMethod"].astype(str).str.contains("automatic", case=False, na=False)
        ).astype("int8")

    if "InternetService" in out.columns:
        out["fiber_customer"] = (out["InternetService"] == "Fiber optic").astype("int8")

    return out


def feature_frame(
    df: pd.DataFrame,
    *,
    target_column: str = TARGET_COLUMN,
    id_column: str = ID_COLUMN,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Engineer features and split the dataframe into ``X`` and ``y``.

    ``y`` is ``None`` when the target column is not present (inference time).
    """
    engineered = add_derived_features(df)
    y = engineered[target_column].astype(int) if target_column in engineered else None
    drop = [column for column in (target_column, id_column) if column in engineered.columns]
    X = engineered.drop(columns=drop)
    return X, y


__all__ = [
    "ADDON_COLUMNS",
    "CONTRACT_MONTHS",
    "DERIVED_COLUMNS",
    "SERVICE_COLUMNS",
    "TENURE_BINS",
    "TENURE_LABELS",
    "add_derived_features",
    "feature_frame",
]
