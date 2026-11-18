"""Feature 1: dataset loading and profiling."""

from __future__ import annotations

import pandas as pd
import pytest

from churn_prediction.data.loader import (
    DatasetNotFoundError,
    format_profile,
    load_dataset,
    profile_dataset,
)


def test_load_dataset_reads_csv(tmp_path, raw_frame):
    csv_path = tmp_path / "churn.csv"
    raw_frame.to_csv(csv_path, index=False)

    loaded = load_dataset(csv_path)

    assert loaded.shape == raw_frame.shape
    assert loaded["customerID"].dtype == object


def test_load_dataset_missing_file_raises(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        load_dataset(tmp_path / "does-not-exist.csv")


def test_profile_reports_shape_missing_and_duplicates(raw_frame):
    profile = profile_dataset(raw_frame, source="unit-test", preview_rows=3)

    assert profile.n_rows == len(raw_frame)
    assert profile.n_columns == raw_frame.shape[1]
    assert profile.duplicate_rows == 2
    assert profile.total_missing > 0  # blank TotalCharges cells
    assert len(profile.preview) == 3
    assert profile.target_distribution is not None
    assert sum(profile.target_distribution.values()) == len(raw_frame)


def test_profile_handles_dataset_without_target(raw_frame):
    profile = profile_dataset(raw_frame.drop(columns=["Churn"]))

    assert profile.target_distribution is None


def test_format_profile_contains_key_sections(raw_frame):
    profile = profile_dataset(raw_frame, source="churn.csv", preview_rows=2)
    rendered = format_profile(profile)

    assert "Shape" in rendered
    assert "Duplicate" in rendered
    assert "Target distribution" in rendered


def test_profile_is_json_friendly(raw_frame):
    import json

    profile = profile_dataset(raw_frame)
    serialised = json.dumps(profile.to_dict(), default=str)

    assert '"n_rows"' in serialised


def test_empty_dataframe_profile():
    profile = profile_dataset(pd.DataFrame({"Churn": []}))
    assert profile.n_rows == 0
