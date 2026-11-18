"""Shared pytest fixtures.

The synthetic dataset mirrors the schema of the IBM Telco CSV, including the
two quirks that make cleaning necessary: blank ``TotalCharges`` for brand new
customers and duplicate rows. Tests never download the real dataset, so the
suite runs offline and in under a few seconds.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from churn_prediction.config import get_settings  # noqa: E402
from churn_prediction.features.cleaning import clean_churn_data  # noqa: E402
from churn_prediction.models.train import train_and_save  # noqa: E402

YES_NO = ["Yes", "No"]
ADDON = ["Yes", "No", "No internet service"]


def build_raw_frame(n: int = 400, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 72, n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.2, 0.25])
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.35, 0.45, 0.2])
    monthly = np.round(rng.uniform(20, 120, n), 2)

    churn_probability = np.where(contract == "Month-to-month", 0.45, 0.12)
    churn_probability = np.where(internet == "Fiber optic", churn_probability + 0.1, churn_probability)
    churn_probability = np.clip(churn_probability - tenure / 400, 0.02, 0.95)
    churn = rng.random(n) < churn_probability

    total = np.round(tenure * monthly, 2).astype(object)
    total[tenure == 0] = np.nan  # blank TotalCharges, exactly like the real dataset

    frame = pd.DataFrame(
        {
            "customerID": [f"CUST-{1000 + i}" for i in range(n)],
            "gender": rng.choice(["Male", "Female"], n),
            "SeniorCitizen": rng.integers(0, 2, n),
            "Partner": rng.choice(YES_NO, n),
            "Dependents": rng.choice(YES_NO, n),
            "tenure": tenure,
            "PhoneService": rng.choice(YES_NO, n),
            "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
            "InternetService": internet,
            "OnlineSecurity": rng.choice(ADDON, n),
            "OnlineBackup": rng.choice(ADDON, n),
            "DeviceProtection": rng.choice(ADDON, n),
            "TechSupport": rng.choice(ADDON, n),
            "StreamingTV": rng.choice(ADDON, n),
            "StreamingMovies": rng.choice(ADDON, n),
            "Contract": contract,
            "PaperlessBilling": rng.choice(YES_NO, n),
            "PaymentMethod": rng.choice(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                n,
            ),
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "Churn": np.where(churn, "Yes", "No"),
        }
    )
    # Two exact duplicate customers to exercise de-duplication.
    return pd.concat([frame, frame.head(2)], ignore_index=True)


@pytest.fixture()
def raw_frame() -> pd.DataFrame:
    return build_raw_frame()


@pytest.fixture()
def cleaned_frame(raw_frame: pd.DataFrame) -> pd.DataFrame:
    cleaned, _ = clean_churn_data(raw_frame)
    return cleaned


@pytest.fixture(scope="session")
def trained_settings(tmp_path_factory: pytest.TempPathFactory):
    """Small trained artifact directory shared by the model and API tests."""
    artifacts = tmp_path_factory.mktemp("churn_prediction-artifacts")
    base = get_settings().with_artifact_dir(artifacts)
    settings = replace(base, test_size=0.25, cv_folds=2, default_model="logistic_regression")

    cleaned, report = clean_churn_data(build_raw_frame())
    train_and_save(
        cleaned,
        model_names=["logistic_regression"],
        settings=settings,
        cleaning_report=report.to_dict(),
    )
    return settings
