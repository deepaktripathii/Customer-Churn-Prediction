"""Data quality checks that run before a model is trained.

The checks are intentionally cheap and dependency-light. Each issue carries a
severity so the training pipeline can decide whether to fail fast or merely
warn the operator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from churn_prediction.config import ID_COLUMN, TARGET_COLUMN


@dataclass
class QualityIssue:
    level: str  # "info" | "warning" | "error"
    code: str
    message: str
    affected_rows: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "level": self.level,
            "code": self.code,
            "message": self.message,
            "affected_rows": self.affected_rows,
        }


@dataclass
class QualityReport:
    issues: list[QualityIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[QualityIssue]:
        return [issue for issue in self.issues if issue.level == "error"]

    @property
    def warnings(self) -> list[QualityIssue]:
        return [issue for issue in self.issues if issue.level == "warning"]

    @property
    def passed(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "issues": [issue.to_dict() for issue in self.issues],
            "n_errors": len(self.errors),
            "n_warnings": len(self.warnings),
        }


def run_quality_checks(df: pd.DataFrame, *, target_column: str = TARGET_COLUMN) -> QualityReport:
    """Run a fixed battery of checks against the raw churn dataframe."""
    report = QualityReport()

    if df.empty:
        report.issues.append(QualityIssue("error", "empty_dataset", "Dataset contains no rows."))
        return report

    duplicates = int(df.duplicated().sum())
    if duplicates:
        report.issues.append(
            QualityIssue(
                "warning",
                "duplicate_rows",
                f"{duplicates} duplicate rows found.",
                affected_rows=duplicates,
            )
        )

    for column in df.columns:
        missing = int(df[column].isna().sum())
        if missing:
            share = missing / len(df) * 100
            report.issues.append(
                QualityIssue(
                    "warning",
                    "missing_values",
                    f"Column '{column}' has {missing} missing values ({share:.1f}%).",
                    affected_rows=missing,
                )
            )

    if "TotalCharges" in df.columns and df["TotalCharges"].dtype == object:
        report.issues.append(
            QualityIssue(
                "warning",
                "numeric_stored_as_text",
                "Column 'TotalCharges' is stored as text and must be coerced to numeric.",
                affected_rows=int(len(df)),
            )
        )

    for column in ("tenure", "MonthlyCharges"):
        if column in df.columns:
            numeric = pd.to_numeric(df[column], errors="coerce")
            negatives = int((numeric < 0).sum())
            if negatives:
                report.issues.append(
                    QualityIssue(
                        "error",
                        "negative_values",
                        f"Column '{column}' has {negatives} negative values.",
                        affected_rows=negatives,
                    )
                )

    constant_columns = [
        column for column in df.columns if df[column].nunique(dropna=False) <= 1
    ]
    if constant_columns:
        report.issues.append(
            QualityIssue(
                "info",
                "constant_columns",
                f"Constant columns carry no signal: {', '.join(constant_columns)}.",
                affected_rows=int(len(df)),
            )
        )

    if target_column in df.columns:
        counts = df[target_column].value_counts(normalize=True, dropna=False)
        if len(counts) > 1:
            minority = float(counts.min())
            if minority < 0.2:
                report.issues.append(
                    QualityIssue(
                        "warning",
                        "class_imbalance",
                        f"Target '{target_column}' is imbalanced "
                        f"(minority class {minority:.1%}); use class weights and recall-aware metrics.",
                        affected_rows=int(len(df)),
                    )
                )
        else:
            report.issues.append(
                QualityIssue(
                    "error",
                    "single_class_target",
                    f"Target '{target_column}' contains only one class.",
                    affected_rows=int(len(df)),
                )
            )
    else:
        report.issues.append(
            QualityIssue(
                "error",
                "missing_target",
                f"Target column '{target_column}' is absent.",
            )
        )

    if ID_COLUMN in df.columns:
        report.issues.append(
            QualityIssue(
                "info",
                "identifier_column",
                f"Column '{ID_COLUMN}' is an identifier and must be dropped before training.",
            )
        )

    return report


__all__ = ["QualityIssue", "QualityReport", "run_quality_checks"]
