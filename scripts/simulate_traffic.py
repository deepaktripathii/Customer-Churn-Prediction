"""Generate demo traffic so the monitoring dashboard has data to show.

Examples::

    python scripts/simulate_traffic.py --n 300
    python scripts/simulate_traffic.py --n 100 --drift          # shift charges to demo PSI drift
    python scripts/simulate_traffic.py --api-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401  (adds src/ to sys.path)
from retainiq.config import TARGET_COLUMN, get_settings
from retainiq.data.loader import load_dataset
from retainiq.features.cleaning import clean_churn_data
from retainiq.models.predictor import ChurnPredictor
from retainiq.monitoring.store import MonitoringStore


def sample_customers(
    df: pd.DataFrame,
    n: int,
    *,
    seed: int = 7,
    drift: bool = False,
) -> list[dict]:
    """Draw realistic customer payloads from the cleaned dataset."""
    rng = np.random.default_rng(seed)
    features = [column for column in df.columns if column != TARGET_COLUMN]
    rows = df[features].sample(n=n, replace=True, random_state=seed).reset_index(drop=True)

    payloads: list[dict] = []
    for _, row in rows.iterrows():
        record = {column: row[column] for column in features}
        record["tenure"] = int(max(0, record["tenure"] + int(rng.integers(-3, 4))))
        multiplier = 1.0 + float(rng.uniform(-0.1, 0.1))
        if drift:
            multiplier *= 1.35  # simulate a price increase in live traffic
        record["MonthlyCharges"] = round(float(record["MonthlyCharges"]) * multiplier, 2)
        record["TotalCharges"] = round(
            float(record["TotalCharges"]) * (multiplier if drift else 1.0), 2
        )
        payloads.append({key: _plain(value) for key, value in record.items()})
    return payloads


def _plain(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send demo prediction traffic to RetainIQ.")
    parser.add_argument("--n", type=int, default=200, help="Number of predictions to generate.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--drift",
        action="store_true",
        help="Inflate monthly charges by 35%% to demonstrate drift detection.",
    )
    parser.add_argument(
        "--api-url",
        default=None,
        help="POST to a running API instead of scoring locally.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Local model to use when --api-url is not set.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_settings()
    raw = load_dataset()
    cleaned, _ = clean_churn_data(raw)
    payloads = sample_customers(cleaned, args.n, seed=args.seed, drift=args.drift)

    if args.api_url:
        import requests

        url = args.api_url.rstrip("/") + "/predict"
        for payload in payloads:
            response = requests.post(url, json=_api_payload(payload), timeout=10)
            if response.status_code != 200:
                print(f"API error {response.status_code}: {response.text}", file=sys.stderr)
                return 1
        print(f"Sent {len(payloads)} predictions to {url}")
        return 0

    predictor = ChurnPredictor(model_name=args.model, settings=settings)
    store = MonitoringStore(settings=settings)
    risk_counts: dict[str, int] = {}
    for payload in payloads:
        prediction = predictor.predict(payload)
        store.log_prediction(
            payload=payload,
            churn_probability=prediction.churn_probability,
            prediction=prediction.prediction,
            risk_level=prediction.risk_level,
            model_name=prediction.model_name,
            model_version=prediction.model_version,
            latency_ms=prediction.latency_ms or 0.0,
        )
        risk_counts[prediction.risk_level] = risk_counts.get(prediction.risk_level, 0) + 1

    print(f"Logged {len(payloads)} predictions with {predictor.version}")
    print(f"Risk distribution: {risk_counts}")
    print(f"Database: {settings.database_url}")
    return 0


def _api_payload(payload: dict) -> dict:
    """Translate dataset column names into the public API schema."""
    from retainiq.api.schemas import MODEL_FIELD_MAP

    reverse = {target: source for source, target in MODEL_FIELD_MAP.items()}
    return {
        reverse[key]: value
        for key, value in payload.items()
        if key in reverse
    }


if __name__ == "__main__":
    raise SystemExit(main())
