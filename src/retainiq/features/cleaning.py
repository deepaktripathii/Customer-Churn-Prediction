"""Data cleaning and preprocessing (Feature 2).

The raw Telco CSV is *almost* model ready, but not quite:

* ``TotalCharges`` is stored as text and contains blank strings for brand new
  customers (tenure = 0).
* ``SeniorCitizen`` is 0/1 while every other yes/no column is "Yes"/"No".
* Duplicate records inflate the training set.
* Categorical columns must be encoded and numeric columns scaled, which is
  handled by the scikit-learn pipeline in :mod:`retainiq.features.pipeline`.

Cleaning is deliberately separate from feature engineering: here we only fix
what is broken, we do not invent new information.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd

from retainiq.config import ID_COLUMN, TARGET_COLUMN
from retainiq.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class CleaningReport:
    """Audit trail of everything the cleaning step changed."""

    rows_in: int = 0
    rows_out: int = 0
    duplicates_removed: int = 0
    missing_before: dict[str, int] = field(default_factory=dict)
    missing_filled: dict[str, int] = field(default_factory=dict)
    coerced_columns: list[str] = field(default_factory=list)
    dropped_columns: list[str] = field(default_factory=list)
    encoded_target: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _coerce_total_charges(df: pd.DataFrame, report: CleaningReport) -> None:
    if "TotalCharges" not in df.columns:
        return
    if pd.api.types.is_numeric_dtype(df["TotalCharges"]):
        return

    before = int(df["TotalCharges"].isna().sum())
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    converted = int(df["TotalCharges"].isna().sum()) - before
    report.coerced_columns.append("TotalCharges")
    report.notes.append(
        f"Coerced 'TotalCharges' to float; {max(converted, 0)} blank values became NaN."
    )


def _normalise_senior_citizen(df: pd.DataFrame, report: CleaningReport) -> None:
    if "SeniorCitizen" not in df.columns:
        return
    if not pd.api.types.is_numeric_dtype(df["SeniorCitizen"]):
        return
    mapping = {0: "No", 1: "Yes"}
    df["SeniorCitizen"] = df["SeniorCitizen"].map(mapping).fillna(df["SeniorCitizen"].astype(str))
    report.notes.append("Mapped numeric 'SeniorCitizen' (0/1) to 'No'/'Yes' for consistency.")


def _fill_missing(df: pd.DataFrame, report: CleaningReport) -> None:
    """Impute numeric columns with the median and categorical ones with the mode."""
    for column in df.columns:
        if not df[column].isna().any():
            continue

        missing = int(df[column].isna().sum())
        if pd.api.types.is_numeric_dtype(df[column]):
            if column == "TotalCharges" and "tenure" in df.columns:
                # A customer with no tenure has not been billed yet.
                df.loc[df["tenure"].fillna(0) == 0, column] = 0.0
            fill_value = float(df[column].median()) if df[column].notna().any() else 0.0
            df[column] = df[column].fillna(fill_value)
            report.notes.append(f"Filled {missing} missing '{column}' values with median {fill_value:.2f}.")
        else:
            mode = df[column].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "Unknown"
            df[column] = df[column].fillna(fill_value)
            report.notes.append(f"Filled {missing} missing '{column}' values with mode '{fill_value}'.")

        report.missing_filled[column] = missing


def _encode_target(df: pd.DataFrame, target_column: str, report: CleaningReport) -> None:
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is missing from the dataset.")

    if df[target_column].dtype == object or str(df[target_column].dtype).startswith("string"):
        normalised = df[target_column].astype(str).str.strip().str.title()
        unexpected = sorted(set(normalised.unique()) - {"Yes", "No"})
        if unexpected:
            raise ValueError(
                f"Unexpected values in target '{target_column}': {unexpected}. "
                "Expected 'Yes'/'No'."
            )
        df[target_column] = normalised.map({"Yes": 1, "No": 0}).astype(int)
        report.encoded_target = True
        report.notes.append("Encoded target 'Churn' as 1 (churned) / 0 (retained).")


def clean_churn_data(
    df: pd.DataFrame,
    *,
    target_column: str = TARGET_COLUMN,
    id_column: str = ID_COLUMN,
    drop_id: bool = True,
    remove_duplicates: bool = True,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean the raw churn dataframe.

    Returns the cleaned dataframe **and** a report that can be rendered in the
    dashboard or appended to the model card.

    The function is pure: the input dataframe is never mutated.
    """
    if df.empty:
        raise ValueError("Cannot clean an empty dataset.")

    work = df.copy()
    report = CleaningReport(rows_in=int(work.shape[0]))

    # 1. Normalise whitespace in text columns and treat blanks as missing.
    for column in work.select_dtypes(include="object").columns:
        work[column] = work[column].astype("string").str.strip()
    work = work.replace({"": pd.NA, " ": pd.NA})

    # 2. Fix incorrect dtypes.
    _coerce_total_charges(work, report)
    _normalise_senior_citizen(work, report)

    # 3. Remove duplicate records, ignoring the customer identifier so that two
    #    rows describing the same customer are also caught.
    if remove_duplicates:
        subset = [column for column in work.columns if column != id_column]
        duplicate_mask = work.duplicated(subset=subset, keep="first")
        removed = int(duplicate_mask.sum())
        if removed:
            work = work.loc[~duplicate_mask].copy()
        report.duplicates_removed = removed

    report.missing_before = {
        str(column): int(count) for column, count in work.isna().sum().items() if count
    }

    # 4. Handle missing values.
    _fill_missing(work, report)

    # 5. Encode the target and drop the identifier.
    _encode_target(work, target_column, report)
    if drop_id and id_column in work.columns:
        work = work.drop(columns=[id_column])
        report.dropped_columns.append(id_column)

    # 6. Final type guard: everything except the target must be numeric or text.
    for column in work.columns:
        if column == target_column:
            continue
        if work[column].dtype == "string":
            work[column] = work[column].astype(object)

    work = work.dropna().reset_index(drop=True)
    report.rows_out = int(work.shape[0])

    if report.rows_out == 0:
        raise ValueError("Cleaning removed every row; check the input dataset.")

    logger.info(
        "Cleaned dataset: %d -> %d rows (%d duplicates removed, %d cells imputed)",
        report.rows_in,
        report.rows_out,
        report.duplicates_removed,
        sum(report.missing_filled.values()),
    )
    return work, report


__all__ = ["CleaningReport", "clean_churn_data"]
