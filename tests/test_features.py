"""Feature 3: feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from retainiq.features.engineering import (
    DERIVED_COLUMNS,
    add_derived_features,
    feature_frame,
)


def test_all_derived_columns_are_created(cleaned_frame):
    engineered = add_derived_features(cleaned_frame)

    for column in DERIVED_COLUMNS:
        assert column in engineered.columns, f"missing engineered column {column}"


def test_contract_features(cleaned_frame):
    engineered = add_derived_features(cleaned_frame)

    monthly = engineered[engineered["Contract"] == "Month-to-month"]
    two_year = engineered[engineered["Contract"] == "Two year"]

    assert (monthly["is_monthly_contract"] == 1).all()
    assert (two_year["contract_months"] == 24).all()


def test_tenure_group_buckets(cleaned_frame):
    engineered = add_derived_features(cleaned_frame)
    groups = engineered.dropna(subset=["tenure_group"])["tenure_group"].astype(str).unique()

    assert set(groups) <= {"0-12m", "13-24m", "25-48m", "49-72m"}


def test_service_counts_are_bounded(cleaned_frame):
    engineered = add_derived_features(cleaned_frame)

    assert engineered["service_count"].between(0, 8).all()
    assert engineered["addon_count"].between(0, 4).all()
    assert engineered["family_size"].between(0, 2).all()


def test_spend_ratio_is_finite(cleaned_frame):
    engineered = add_derived_features(cleaned_frame)
    ratio = engineered["spend_gap_ratio"]

    assert np.isfinite(ratio).all()


def test_engineered_features_add_no_missing_values(cleaned_frame):
    engineered = add_derived_features(cleaned_frame)
    numeric_derived = [
        "avg_charge_per_month",
        "expected_total_charges",
        "spend_gap_ratio",
        "service_count",
        "contract_months",
    ]

    assert not engineered[numeric_derived].isna().any().any()


def test_partial_frame_is_supported():
    partial = pd.DataFrame({"tenure": [3], "MonthlyCharges": [1200.0], "TotalCharges": [3600.0]})
    engineered = add_derived_features(partial)

    assert engineered.loc[0, "avg_charge_per_month"] == pytest.approx(1200.0)
    assert engineered.loc[0, "spend_gap_ratio"] == pytest.approx(0.0)


def test_feature_frame_splits_target(cleaned_frame):
    X, y = feature_frame(cleaned_frame)

    assert "Churn" not in X.columns
    assert y is not None
    assert set(y.unique()) == {0, 1}


def test_feature_frame_without_target(cleaned_frame):
    X, y = feature_frame(cleaned_frame.drop(columns=["Churn"]))

    assert y is None
    assert "tenure_group" in X.columns
