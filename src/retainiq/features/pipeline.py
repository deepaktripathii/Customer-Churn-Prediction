"""Reusable scikit-learn preprocessing and modelling pipelines.

The preprocessing is stored *inside* the exported pipeline, which means the
API can accept raw customer records (strings and all) and needs no separate
encoding logic at inference time. This is what prevents training/serving skew.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from retainiq.config import ID_COLUMN, TARGET_COLUMN


def split_feature_types(
    df: pd.DataFrame,
    *,
    target_column: str = TARGET_COLUMN,
    id_column: str = ID_COLUMN,
) -> tuple[list[str], list[str]]:
    """Return ``(numeric_features, categorical_features)`` for a cleaned frame."""
    features = [column for column in df.columns if column not in (target_column, id_column)]
    numeric = [column for column in features if pd.api.types.is_numeric_dtype(df[column])]
    categorical = [column for column in features if column not in numeric]
    return numeric, categorical


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    *,
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """Build the imputation + encoding + scaling transformer."""
    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median"))
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]

    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), numeric_features),
            ("categorical", Pipeline(categorical_steps), categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(
    estimator: Any,
    numeric_features: list[str],
    categorical_features: list[str],
    *,
    scale_numeric: bool = True,
) -> Pipeline:
    """Wrap an estimator together with its preprocessing steps."""
    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(
                    numeric_features, categorical_features, scale_numeric=scale_numeric
                ),
            ),
            ("model", estimator),
        ]
    )


def feature_names(pipeline: Pipeline) -> list[str]:
    """Names of the columns seen by the estimator (after one-hot encoding)."""
    preprocessor = pipeline.named_steps["preprocessor"]
    try:
        return [str(name) for name in preprocessor.get_feature_names_out()]
    except Exception:  # pragma: no cover - only for unfitted pipelines
        return []


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.2,
    validation_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Three-way stratified split: train / validation / test.

    The validation set is used to pick the decision threshold and the test set
    is touched exactly once, for the final reported metrics.
    """
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=validation_size,
        stratify=y_train_full,
        random_state=random_state,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


__all__ = [
    "build_pipeline",
    "build_preprocessor",
    "feature_names",
    "split_feature_types",
    "stratified_split",
]
