"""Download the public IBM Telco Customer Churn dataset.

Usage::

    python scripts/download_data.py            # saves data/raw/telco_churn.csv
    python scripts/download_data.py --force    # overwrite an existing file
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

import _bootstrap  # noqa: F401  (adds src/ to sys.path)
from churn_prediction.config import TELCO_DATASET_MIRRORS, TELCO_DATASET_URL, get_settings
from churn_prediction.data.loader import format_profile, load_dataset, profile_dataset
from churn_prediction.logging_config import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS = {"customerID", "tenure", "MonthlyCharges", "Contract", "Churn"}


def download_dataset(url: str, destination: Path, *, timeout: int = 60) -> Path:
    """Download ``url`` to ``destination`` and validate that it looks like the dataset."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s", url)
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    text = response.content.decode("utf-8", errors="replace")
    header = text.splitlines()[0] if text else ""
    missing = REQUIRED_COLUMNS - set(header.split(","))
    if missing:
        raise ValueError(
            f"Downloaded file is not the Telco churn dataset (missing columns: {sorted(missing)})."
        )

    destination.write_bytes(response.content)
    logger.info("Saved %.1f KB to %s", len(response.content) / 1024, destination)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download the Telco customer churn dataset.")
    parser.add_argument("--url", default=TELCO_DATASET_URL)
    parser.add_argument("--output", default=None, help="Destination CSV path.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing file.")
    args = parser.parse_args(argv)

    settings = get_settings()
    destination = Path(args.output) if args.output else settings.raw_data_path

    if destination.exists() and not args.force:
        print(f"Dataset already present at {destination} (use --force to re-download).")
    else:
        urls = [args.url, *[url for url in TELCO_DATASET_MIRRORS if url != args.url]]
        last_error: Exception | None = None
        for url in urls:
            try:
                download_dataset(url, destination)
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning("Download failed from %s: %s", url, exc)
        else:
            print(f"All download mirrors failed. Last error: {last_error}", file=sys.stderr)
            return 1

    df = load_dataset(destination)
    print(format_profile(profile_dataset(df, source=str(destination))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
