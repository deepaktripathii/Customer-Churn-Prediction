"""Dataset loading and profiling (Feature 1: dataset management).

The module deliberately keeps the raw CSV untouched. Profiling answers the
questions a data scientist asks on the first day of a project: how many rows
and columns, what are the column types, where are the missing values, how many
duplicate records exist, and what does the data actually look like.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from retainiq.config import ID_COLUMN, TARGET_COLUMN, get_settings
from retainiq.logging_config import get_logger

logger = get_logger(__name__)


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the raw churn dataset is missing."""


@dataclass
class DatasetProfile:
    """A serialisable summary of a tabular dataset."""

    source: str
    n_rows: int
    n_columns: int
    columns: list[str]
    dtypes: dict[str, str]
    missing_values: dict[str, int]
    total_missing: int
    duplicate_rows: int
    memory_usage_mb: float
    preview: list[dict[str, Any]] = field(default_factory=list)
    target_distribution: dict[str, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_dataset(
    path: str | Path | None = None,
    *,
    nrows: int | None = None,
    keep_default_na: bool = True,
) -> pd.DataFrame:
    """Load the raw churn dataset from disk.

    Args:
        path: CSV location. Defaults to ``data/raw/telco_churn.csv``.
        nrows: Optionally read only the first ``nrows`` rows (useful for tests).
        keep_default_na: Passed to :func:`pandas.read_csv`.
    """
    csv_path = Path(path) if path is not None else get_settings().raw_data_path
    if not csv_path.exists():
        raise DatasetNotFoundError(
            f"Dataset not found at '{csv_path}'. "
            "Run `python scripts/download_data.py` to fetch the Telco churn CSV."
        )

    df = pd.read_csv(csv_path, nrows=nrows, keep_default_na=keep_default_na)
    if ID_COLUMN in df.columns:
        df[ID_COLUMN] = df[ID_COLUMN].astype(str)
    logger.info("Loaded dataset %s with shape %s", csv_path.name, df.shape)
    return df


def profile_dataset(
    df: pd.DataFrame,
    *,
    source: str = "",
    preview_rows: int = 5,
    target_column: str = TARGET_COLUMN,
) -> DatasetProfile:
    """Build a :class:`DatasetProfile` describing ``df``."""
    missing = df.isna().sum()
    missing = {str(col): int(count) for col, count in missing.items() if count > 0}

    target_distribution: dict[str, int] | None = None
    if target_column in df.columns:
        target_distribution = {
            str(label): int(count) for label, count in df[target_column].value_counts().items()
        }

    return DatasetProfile(
        source=source,
        n_rows=int(df.shape[0]),
        n_columns=int(df.shape[1]),
        columns=[str(col) for col in df.columns],
        dtypes={str(col): str(dtype) for col, dtype in df.dtypes.items()},
        missing_values=missing,
        total_missing=int(sum(missing.values())),
        duplicate_rows=int(df.duplicated().sum()),
        memory_usage_mb=round(float(df.memory_usage(deep=True).sum()) / 1024**2, 3),
        preview=df.head(preview_rows).to_dict(orient="records"),
        target_distribution=target_distribution,
    )


def format_profile(profile: DatasetProfile) -> str:
    """Human readable rendering of a profile for the CLI."""
    lines = [
        f"Dataset   : {profile.source or '<in-memory>'}",
        f"Shape     : {profile.n_rows:,} rows x {profile.n_columns} columns",
        f"Memory    : {profile.memory_usage_mb:.2f} MB",
        f"Duplicates: {profile.duplicate_rows}",
        f"Missing   : {profile.total_missing} cells across {len(profile.missing_values)} columns",
        "",
        "Columns:",
    ]
    for column in profile.columns:
        dtype = profile.dtypes.get(column, "?")
        missing = profile.missing_values.get(column, 0)
        suffix = f"  ({missing} missing)" if missing else ""
        lines.append(f"  - {column:<22} {dtype}{suffix}")

    if profile.target_distribution:
        lines.append("")
        lines.append("Target distribution:")
        for label, count in profile.target_distribution.items():
            share = count / profile.n_rows * 100 if profile.n_rows else 0.0
            lines.append(f"  - {label:<8} {count:>6,} ({share:5.1f}%)")

    if profile.preview:
        preview = pd.DataFrame(profile.preview)
        lines.append("")
        lines.append("Preview:")
        lines.append(preview.to_string(index=False))

    return "\n".join(lines)


def print_profile(path: str | Path | None = None, *, preview_rows: int = 5) -> DatasetProfile:
    df = load_dataset(path)
    profile = profile_dataset(
        df,
        source=str(path or get_settings().raw_data_path),
        preview_rows=preview_rows,
    )
    print(format_profile(profile))
    return profile


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect the raw churn dataset.")
    parser.add_argument("--path", default=None, help="CSV file (default: data/raw/telco_churn.csv)")
    parser.add_argument("--preview-rows", type=int, default=5)
    parser.add_argument("--json", action="store_true", help="Emit the raw profile as JSON")
    args = parser.parse_args(argv)

    df = load_dataset(args.path)
    profile = profile_dataset(
        df,
        source=str(args.path or get_settings().raw_data_path),
        preview_rows=args.preview_rows,
    )
    if args.json:
        print(json.dumps(profile.to_dict(), indent=2, default=str))
    else:
        print(format_profile(profile))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
