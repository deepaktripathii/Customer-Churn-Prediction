# RetainIQ — AI-Powered Customer Retention Intelligence

An end-to-end churn prediction and monitoring platform for telecom operators.

It takes a raw customer dataset, cleans and engineers it, trains and compares
three classifiers, serves per-customer churn probabilities with a risk band and
explanations behind FastAPI, and monitors the live service for latency and input
drift from a Streamlit dashboard.

```
Raw Telco CSV → profile → clean → engineer → train (LR / RF / XGBoost) → evaluate
      → saved pipeline + metrics + drift reference
      → FastAPI /predict + /monitoring/*  → SQLite/Postgres prediction log
      → Streamlit dashboard
```

## Why this project

Churn is a rare, expensive event. A model that predicts "no churn" for everyone
is 73% accurate on this dataset and worthless. RetainIQ is built around that
reality: recall-aware metrics, a tuned decision threshold, per-customer
explanations, and monitoring that tells you when the world has moved away from
the training data.

## Feature checklist

| # | Feature | Where |
| --- | --- | --- |
| 1 | Dataset loading, preview, missing/duplicate profiling | `src/retainiq/data/loader.py`, `quality.py` |
| 2 | Cleaning: dtype fixes, de-duplication, imputation, target encoding | `src/retainiq/features/cleaning.py` |
| 3 | Feature engineering: tenure buckets, spend ratios, service depth | `src/retainiq/features/engineering.py` |
| 4 | Logistic Regression + Random Forest (+ optional XGBoost) training | `src/retainiq/models/registry.py`, `train.py` |
| 5 | Evaluation and model comparison (accuracy, precision, recall, F1, ROC/PR AUC, confusion matrix) | `src/retainiq/models/evaluate.py` |
| 6 | Customer-level prediction with probability, risk band and explanations | `src/retainiq/models/predictor.py` |
| 7 | FastAPI service, prediction logging, latency and PSI drift monitoring | `src/retainiq/api/`, `src/retainiq/monitoring/` |

## Results on the public IBM Telco dataset

7,043 raw rows → 7,021 after removing 22 duplicate customers. Stratified
80/20 split with a validation slice for threshold tuning; metrics below are on
the untouched test set at each model's F1-optimal threshold.

| Model | Threshold | Accuracy | Precision | Recall | F1 | ROC AUC | PR AUC | Train |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Random Forest** | 0.453 | 0.769 | 0.548 | **0.729** | **0.625** | **0.841** | **0.652** | 6.5 s |
| Logistic Regression | 0.659 | **0.785** | **0.592** | 0.605 | 0.598 | 0.839 | 0.641 | 3.4 s |
| XGBoost | 0.346 | 0.769 | 0.555 | 0.648 | 0.598 | 0.839 | 0.635 | 4.6 s |

Random Forest wins on F1 and recall, which is what retention teams care about:
it catches 73% of customers who actually churn. The linear model stays in the
registry because its coefficients are auditable.

## Quickstart

```bash
git clone https://github.com/deepaktripathii/RetainIQ-AI-Powered-Customer-Retention-Intelligence-Platform.git
cd RetainIQ-AI-Powered-Customer-Retention-Intelligence-Platform

python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt && pip install -e .

python scripts/download_data.py     # fetch the public Telco churn CSV
python scripts/train.py             # train, evaluate, export artifacts
```

Run the API:

```bash
uvicorn retainiq.api.main:app --reload
```

Run the dashboard in a second terminal:

```bash
streamlit run src/retainiq/dashboard/app.py
```

Open <http://localhost:8000/docs> for the API and
<http://localhost:8501> for the dashboard. Generate demo traffic so the
monitoring page has data:

```bash
python scripts/simulate_traffic.py --n 300
python scripts/simulate_traffic.py --n 100 --drift   # demonstrates PSI drift
```

## Prediction example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 3,
    "monthly_charges": 1200,
    "contract": "Month-to-month",
    "internet_service": "Fiber optic",
    "payment_method": "Electronic check"
  }'
```

```json
{
  "prediction": "Churn",
  "churn_probability": 0.706,
  "risk_level": "High",
  "threshold": 0.4528,
  "model_name": "random_forest",
  "model_version": "random_forest-2026.07.12-101500",
  "latency_ms": 106.7,
  "explanations": [
    { "feature": "tenure", "label": "Tenure (months)", "value": 3,
      "direction": "increases risk", "impact": 0.2107 }
  ]
}
```

Only `tenure`, `monthly_charges` and `contract` are mandatory; omitted fields
fall back to the training-time typical customer, so the endpoint is usable from
a partial CRM record.

## API surface

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Service, model and database status |
| POST | `/predict` | Churn probability, risk band, explanations |
| POST | `/predict/sample` | Score the documented sample customer |
| GET | `/model-info` | Model version, threshold, metrics, registry |
| GET | `/monitoring/summary` | Volume, churn rate, risk mix, latency, daily trend |
| GET | `/monitoring/predictions` | Recent prediction log entries |
| GET | `/monitoring/drift` | PSI per feature versus the training distribution |

Full reference: [`docs/api.md`](docs/api.md).

## How the modelling works

1. **Profile** - rows, columns, dtypes, missing values, duplicates, target balance.
2. **Clean** - strip whitespace, coerce `TotalCharges` to numeric, map
   `SeniorCitizen` to Yes/No, de-duplicate ignoring the customer id, impute
   numerics with the median (zero total charges for `tenure = 0`) and
   categoricals with the mode.
3. **Engineer** - `tenure_group`, `contract_months`, `is_monthly_contract`,
   `avg_charge_per_month`, `expected_total_charges`, `spend_gap_ratio`,
   `service_count`, `addon_count`, `streaming_count`, `family_size`,
   `has_support`, `autopay`, `fiber_customer`.
4. **Split** - stratified train / validation / test.
5. **Train** - each estimator is wrapped with the same `ColumnTransformer`
   (median imputer + scaler, mode imputer + one-hot), so the exported `.joblib`
   accepts raw customer records.
6. **Tune** - the decision threshold maximises F1 on the validation split; the
   chosen model is then refit on train + validation.
7. **Evaluate** - metrics and curves on the test set only.
8. **Explain** - each supplied field is reset to the typical customer and the
   probability delta is measured (model agnostic, works for linear models and
   tree ensembles alike).
9. **Monitor** - every request is logged with its payload and latency; PSI
   compares live input distributions against the training reference.

## Project structure

```
├── src/retainiq/
│   ├── config.py              # env-driven settings, risk bands
│   ├── data/                  # Feature 1: loader + quality checks
│   ├── features/              # Features 2 & 3: cleaning, engineering, pipelines
│   ├── models/                # Features 4, 5, 6: registry, train, evaluate, predict
│   ├── monitoring/            # Feature 7: prediction log + PSI drift
│   ├── api/                   # Feature 7: FastAPI app and schemas
│   └── dashboard/             # Streamlit UI
├── scripts/                   # download_data, train, simulate_traffic
├── tests/                     # 60+ tests on a synthetic dataset
├── docs/                      # architecture, API, model card, interview notes
├── notebooks/                 # exploratory analysis
├── Dockerfile · docker-compose.yml
└── artifacts/                 # generated (gitignored)
```

## Testing and CI

```bash
pip install -r requirements-dev.txt
pytest -q          # 63 tests, offline, ~20 seconds
ruff check src tests scripts
```

GitHub Actions runs lint, the test-suite, and a smoke training run on every push
(see `.github/workflows/ci.yml`).

## Docker

```bash
docker compose up --build
# API       → http://localhost:8000
# Dashboard → http://localhost:8501
```

The entrypoint downloads the dataset and trains an initial model if the mounted
`data/` and `artifacts/` volumes are empty, so a fresh clone works with one
command.

## Configuration

Copy `.env.example` to `.env` and adjust as needed. Useful variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `RETAINIQ_DATA_DIR` | `./data` | Raw dataset location |
| `RETAINIQ_ARTIFACT_DIR` | `./artifacts` | Models, metrics, reference stats |
| `RETAINIQ_DATABASE_URL` | `sqlite:///artifacts/monitoring.db` | Prediction log (PostgreSQL DSN supported) |
| `RETAINIQ_DEFAULT_MODEL` | `random_forest` | Model served by the API |
| `RETAINIQ_API_URL` | `http://localhost:8000` | Dashboard → API base URL |

## Limitations and next steps

- Trained on a public sample; re-validate on the operator's own customers.
- Probabilities are not calibrated for tree ensembles; use the risk bands for
  individual decisions.
- Drift monitoring covers inputs. Performance drift needs churn outcomes joined
  back to `prediction_logs` after the observation window.
- Next: probability calibration, SHAP explanations, MLflow tracking, a fairness
  audit across gender and senior-citizen segments.

## Documentation

- [Architecture and design decisions](docs/architecture.md)
- [API reference](docs/api.md)
- [Model card](docs/model_card.md)
- [Interview notes](docs/interview-notes.md)

## Licence

MIT. See [LICENSE](LICENSE). The dataset is published by IBM for educational
use; check the upstream terms before commercial use.
