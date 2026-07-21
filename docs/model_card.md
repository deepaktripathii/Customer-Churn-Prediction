# Model card

This project generates `artifacts/model_card.json` on every training run, so the
numbers below are reproducible rather than hand-written. This page explains how
to read that file.

## Current benchmark

Public IBM Telco dataset, 7,021 rows after cleaning (22 duplicates removed),
80/20 stratified split with a 20% validation slice carved from the training
data. Metrics are computed on the untouched test set at each model's
F1-optimal threshold.

| Model | Threshold | Accuracy | Precision | Recall | F1 | ROC AUC | PR AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Random Forest | 0.453 | 0.769 | 0.548 | 0.729 | 0.625 | 0.841 | 0.652 |
| Logistic Regression | 0.659 | 0.785 | 0.592 | 0.605 | 0.598 | 0.839 | 0.641 |
| XGBoost | 0.346 | 0.769 | 0.555 | 0.648 | 0.598 | 0.839 | 0.635 |

Random Forest is selected by F1. Logistic Regression remains valuable because
its coefficients are directly interpretable and it trains in 3 seconds - if the
business needs a simple auditable rule set, it is a defensible fallback.

## Intended use

Prioritise retention outreach for telecom customers. The output is a
probability plus a risk band; it is a prioritisation tool, not a causal claim.
Sending a discount to a customer the model flags does not guarantee retention.

## Decision threshold

The threshold maximises F1 on the validation split. In churn, missing a churner
usually costs more than a wasted retention offer, so operators may prefer a
lower threshold (higher recall, lower precision). The threshold is stored with
the model, exposed by `/model-info`, and returned in every prediction response.

## Features

Raw columns (drop `customerID`) plus engineered features:

- `tenure_group` - lifecycle bucket (0-12, 13-24, 25-48, 49-72 months)
- `contract_months`, `is_monthly_contract` - commitment strength
- `avg_charge_per_month`, `expected_total_charges`, `spend_gap_ratio` - billing consistency
- `service_count`, `addon_count`, `streaming_count`, `has_support` - product depth
- `family_size`, `autopay`, `fiber_customer` - household and payment behaviour

## Limitations

- Trained on a public sample. Production performance must be re-validated on the
  operator's own customer base and billing system.
- Labels are historical churn outcomes; competitor promotions or outages after
  the snapshot are invisible to the model.
- Tree ensembles are not guaranteed to be calibrated. Use the risk bands, not
  the raw probability, for individual decisions.
- Monitoring measures input drift. Real accuracy drift requires churn outcomes
  joined back to prediction logs after the observation window.
- Fairness was not audited. `gender`, `SeniorCitizen` and `family_size` are
  inputs; before deployment, check error rates across these groups and consider
  removing protected attributes if they do not improve recall.
