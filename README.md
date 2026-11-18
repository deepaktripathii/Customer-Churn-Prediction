# Customer Churn Prediction

Churn prediction and monitoring service for telecom customer data.

Raw CSV -> profiling -> cleaning -> feature engineering -> model comparison
(Logistic Regression, Random Forest, XGBoost) -> FastAPI scoring with risk bands
-> drift monitoring -> Streamlit dashboard.

## Results

IBM Telco dataset, 7,021 rows after cleaning. Metrics are measured on a held-out
test set at each model's tuned decision threshold.

| Model | Precision | Recall | F1 | ROC AUC |
| --- | --- | --- | --- | --- |
| Random Forest | 0.548 | 0.729 | 0.625 | 0.841 |
| Logistic Regression | 0.592 | 0.605 | 0.598 | 0.839 |
| XGBoost | 0.555 | 0.648 | 0.598 | 0.839 |

Random Forest is served by default because it catches the most churners.

## Features

1. Dataset loading and profiling: rows, columns, dtypes, missing values,
   duplicates, preview and target balance.
2. Cleaning: whitespace, `TotalCharges` type fix, `SeniorCitizen` mapping,
   de-duplication, median/mode imputation and target encoding.
3. Feature engineering: tenure buckets, contract length, spend ratios, service
   counts and payment flags.
4. Training: stratified train/validation/test split, class weights, cross
   validation, threshold tuning on validation data, best-model selection.
5. Evaluation: accuracy, precision, recall, F1, ROC/PR curves, confusion matrix
   and a model comparison table.
6. Prediction: churn probability, low/medium/high risk band and per-field
   explanations for a single customer.
7. FastAPI: `/health`, `/predict`, `/model-info`, monitoring endpoints and a
   SQLite/PostgreSQL prediction log.
8. Monitoring: prediction volume, risk mix, latency (avg and p95), model
   versions and PSI input drift against the training distribution.

## Quickstart

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .

python scripts/download_data.py
python scripts/train.py
uvicorn churn_prediction.api.main:app --reload
```

Dashboard:

```bash
streamlit run src/churn_prediction/dashboard/app.py
```

Demo traffic for the monitoring page:

```bash
python scripts/simulate_traffic.py --n 300
```

## API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Model and database status |
| POST | `/predict` | Churn probability, risk band, explanations |
| GET | `/model-info` | Model version, threshold and metrics |
| GET | `/monitoring/summary` | Volume, latency, risk mix |
| GET | `/monitoring/predictions` | Recent predictions |
| GET | `/monitoring/drift` | PSI against the training reference |

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure": 3, "monthly_charges": 1200, "contract": "Month-to-month"}'
```

```json
{
  "prediction": "Churn",
  "churn_probability": 0.706,
  "risk_level": "High",
  "threshold": 0.4528,
  "model_name": "random_forest",
  "explanations": [
    {"feature": "tenure", "label": "Tenure (months)", "value": 3,
     "direction": "increases risk", "impact": 0.2107}
  ]
}
```

## Project structure

```
src/churn_prediction/
  config.py, logging_config.py
  data/          loader, quality checks
  features/      cleaning, engineering, pipelines, field mapping
  models/        registry, training, evaluation, prediction
  monitoring/    prediction log, PSI drift
  api/           FastAPI app and schemas
  dashboard/     Streamlit UI
scripts/         download, train, simulate traffic
tests/           64 offline tests
docs/            architecture, API, model card, interview notes
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
ruff check src tests scripts
```

CI runs lint, tests and a smoke training run on every push.

## Docker

```bash
docker compose up --build
# API       -> http://localhost:8000
# Dashboard -> http://localhost:8501
```
