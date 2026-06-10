# Dataset

## Source

The project uses the public **IBM Telco Customer Churn** dataset:

- Canonical copy: <https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv>
- Also published on Kaggle as *Telco Customer Churn*.

`python scripts/download_data.py` fetches the CSV into `data/raw/telco_churn.csv`.
The file is committed so the repository works offline, but you can delete it and
re-download at any time with `--force`.

## Why this dataset

- 7,043 rows and 21 columns: big enough to train three algorithms, small enough
  to train in seconds on a laptop.
- Realistic quirks: `TotalCharges` is stored as text with blank values,
  `SeniorCitizen` is 0/1 while other flags are "Yes"/"No", and there are
  duplicate rows. Feature 2 of the project exists because of these quirks.
- A healthy 26.5% churn rate: imbalanced, but not so extreme that recall
  becomes impossible to optimise.

## Schema

| Column | Type | Notes |
| --- | --- | --- |
| customerID | string | Identifier, dropped before training |
| gender | categorical | Male / Female |
| SeniorCitizen | 0/1 | Normalised to Yes / No during cleaning |
| Partner, Dependents | categorical | Yes / No |
| tenure | integer | Months with the company (0-72) |
| PhoneService, MultipleLines | categorical | Service flags |
| InternetService | categorical | DSL / Fiber optic / No |
| OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport | categorical | Add-on subscriptions |
| StreamingTV, StreamingMovies | categorical | Streaming subscriptions |
| Contract | categorical | Month-to-month / One year / Two year |
| PaperlessBilling | categorical | Yes / No |
| PaymentMethod | categorical | Four values, two are automatic |
| MonthlyCharges | float | Current bill |
| TotalCharges | text -> float | Blank for customers with `tenure = 0` |
| Churn | Yes / No | **Target**, encoded to 1 / 0 |

## Licence

The dataset is distributed by IBM for educational use. Check the upstream
repository for the exact terms before using it commercially.
