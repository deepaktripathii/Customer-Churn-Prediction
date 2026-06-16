"""Feature 2: cleaning and preprocessing."""

from __future__ import annotations

import pandas as pd
import pytest

from retainiq.features.cleaning import clean_churn_data
from retainiq.features.pipeline import build_pipeline, split_feature_types


def test_cleaning_removes_duplicates_and_identifier(raw_frame):
    cleaned, report = clean_churn_data(raw_frame)

    assert report.duplicates_removed == 2
    assert len(cleaned) == len(raw_frame) - 2
    assert "customerID" not in cleaned.columns
    assert "customerID" in report.dropped_columns


def test_cleaning_fills_missing_values(raw_frame):
    cleaned, report = clean_churn_data(raw_frame)

    assert not cleaned.isna().any().any()
    assert "TotalCharges" in report.missing_filled
    assert cleaned["TotalCharges"].dtype != object


def test_new_customers_get_zero_total_charges(raw_frame):
    cleaned, _ = clean_churn_data(raw_frame)
    new_customers = cleaned[cleaned["tenure"] == 0]

    assert (new_customers["TotalCharges"] == 0).all()


def test_senior_citizen_is_normalised(raw_frame):
    cleaned, _ = clean_churn_data(raw_frame)

    assert set(cleaned["SeniorCitizen"].unique()) <= {"Yes", "No"}


def test_target_is_binary(raw_frame):
    cleaned, report = clean_churn_data(raw_frame)

    assert report.encoded_target is True
    assert set(cleaned["Churn"].unique()) == {0, 1}


def test_cleaning_does_not_mutate_input(raw_frame):
    snapshot = raw_frame.copy(deep=True)
    clean_churn_data(raw_frame)

    pd.testing.assert_frame_equal(raw_frame, snapshot)


def test_whitespace_is_stripped(raw_frame):
    dirty = raw_frame.copy()
    dirty.loc[0, "Contract"] = "  Month-to-month  "
    cleaned, _ = clean_churn_data(dirty)

    assert "Month-to-month" in cleaned["Contract"].unique()


def test_unknown_target_value_raises(raw_frame):
    dirty = raw_frame.copy()
    dirty.loc[0, "Churn"] = "maybe"

    with pytest.raises(ValueError, match="Unexpected values"):
        clean_churn_data(dirty)


def test_empty_dataset_raises():
    with pytest.raises(ValueError):
        clean_churn_data(pd.DataFrame())


def test_feature_types_are_disjoint(cleaned_frame):
    numeric, categorical = split_feature_types(cleaned_frame)

    assert numeric and categorical
    assert not set(numeric) & set(categorical)
    assert "Churn" not in numeric + categorical


def test_pipeline_encodes_categories(cleaned_frame):
    from sklearn.linear_model import LogisticRegression

    numeric, categorical = split_feature_types(cleaned_frame)
    pipeline = build_pipeline(LogisticRegression(max_iter=200), numeric, categorical)
    pipeline.fit(cleaned_frame.drop(columns=["Churn"]), cleaned_frame["Churn"])

    probabilities = pipeline.predict_proba(cleaned_frame.drop(columns=["Churn"]))[:, 1]

    assert probabilities.shape[0] == len(cleaned_frame)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
