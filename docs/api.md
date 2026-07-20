# API reference

Base URL: `http://localhost:8000` · Interactive docs: `/docs`

Every prediction is written to the monitoring store, so the endpoints below can
be exercised in any order.

## `GET /health`

Service and dependency status. Never raises: a missing model or database is
reported as `degraded`.

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "random_forest",
  "model_version": "random_forest-2026.07.12-101500",
  "database": "ok",
  "predictions_logged": 128,
  "version": "1.0.0",
  "timestamp": "2026-07-12T10:15:00+00:00"
}
```

## `GET /model-info`

Served model, decision threshold, risk bands, metrics and the registry of
trainable algorithms.

```json
{
  "model_name": "random_forest",
  "display_name": "Random Forest",
  "version": "random_forest-2026.07.12-101500",
  "trained_at": "2026-07-12T10:15:00+00:00",
  "threshold": 0.4528,
  "risk_bands": { "low_below": 0.35, "high_from": 0.65 },
  "features": ["gender", "SeniorCitizen", "tenure", "..."],
  "metrics": { "accuracy": 0.7687, "recall": 0.7285, "f1": 0.6251 },
  "available_models": ["logistic_regression", "random_forest", "xgboost"],
  "registry": [{ "name": "random_forest", "family": "tree-ensemble", "..." : "..." }]
}
```

## `POST /predict`

Only `tenure`, `monthly_charges` and `contract` are required. Omitted fields are
filled with the training-time typical customer.

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
  "latency_ms": 106.66,
  "explanations": [
    {
      "feature": "tenure",
      "label": "Tenure (months)",
      "value": 3,
      "direction": "increases risk",
      "impact": 0.2107
    },
    {
      "feature": "MonthlyCharges",
      "label": "Monthly charges",
      "value": 1200.0,
      "direction": "increases risk",
      "impact": 0.0258
    }
  ]
}
```

`impact` is the probability delta against the typical customer, so positive
values push the customer towards churn.

Errors:

| Status | Meaning |
| --- | --- |
| 422 | Payload validation failed (unknown field, negative tenure, bad enum) |
| 503 | No trained model is available; run `python scripts/train.py` |

## `GET /monitoring/summary`

Aggregates over the most recent 1,000 predictions: totals, churn rate, risk
distribution, per-version counts, latency (`avg`, `p95`) and a daily series.

## `GET /monitoring/predictions?limit=50`

Recent prediction log entries, newest first. Each entry includes the raw
payload, probability, risk band, model version and latency.

## `GET /monitoring/drift?limit=500&min_samples=20`

Population Stability Index per feature against the training distribution.

```json
{
  "status": "stable",
  "n_samples": 500,
  "min_samples": 20,
  "overall_psi": 0.0312,
  "max_psi": 0.0871,
  "features": [
    { "feature": "MonthlyCharges", "type": "numeric", "psi": 0.0871, "status": "stable" },
    { "feature": "Contract", "type": "categorical", "psi": 0.0104, "status": "stable" }
  ]
}
```

PSI bands: `< 0.10` stable, `0.10 - 0.25` moderate, `> 0.25` significant.
A minimum of 20 samples is enforced because PSI is meaningless on tiny batches.

## `POST /predict/sample`

Scores the documented sample customer. Useful for container health checks and
smoke tests.
