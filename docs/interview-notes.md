# Interview notes

Short, defensible answers to the questions this project invites.

## 1. Why is data preprocessing important?

Models are numeric functions: they cannot multiply the string "Month-to-month",
and a blank `TotalCharges` propagates NaN through every calculation. In this
dataset pre-processing fixes three concrete problems: `TotalCharges` is text
with blanks for new customers, `SeniorCitizen` is 0/1 while other flags are
"Yes"/"No", and 22 duplicate rows inflate the training set. Skipping these steps
either crashes training or, worse, silently drops rows via `dropna` and biases
the churn rate.

## 2. What is feature engineering and how does it improve performance?

Cleaning fixes existing values; feature engineering creates new signal. A
month-to-month customer with 3 months of tenure is not comparable to a two-year
contract customer in year five, so we add `contract_months`,
`is_monthly_contract`, `tenure_group`, `avg_charge_per_month`, `spend_gap_ratio`
and service counts. These make interactions explicit instead of hoping a linear
model discovers them. Tree models can learn interactions themselves, which is
why engineering matters most for Logistic Regression.

## 3. Why Random Forest over Logistic Regression?

Random Forest captured non-linear interactions (contract x tenure x fibre) and
reached 0.729 recall versus 0.605 for the linear baseline at a comparable ROC
AUC (~0.84). But the honest answer is "it depends": Logistic Regression is
easier to explain, faster to retrain, and its coefficients are auditable. If
regulators demand a rule set the company can read, the linear model wins despite
the lower F1.

## 4. Why not accuracy as the main metric?

Only 26.5% of customers churn. A model that predicts "retained" for everyone is
73.5% accurate and catches zero churners. Recall answers "of the customers who
actually left, how many did we flag?", precision answers "of the customers we
flagged, how many really left?". The F1 score balances them; the business case
decides the operating point.

## 5. Why tune the threshold instead of using 0.5?

0.5 is only optimal when the classes are balanced and the costs symmetric.
Here a false negative (a churner we missed) is usually more expensive than a
false positive (a retention offer sent to someone who would have stayed).
Thresholds are selected on a validation split by maximising F1, then frozen;
the test set is never used for tuning, otherwise the reported metric leaks.

## 6. Why keep preprocessing inside the saved pipeline?

If the API re-implements encoding separately, the day someone adds a category or
changes an imputation strategy, training and serving diverge. Exporting a single
scikit-learn `Pipeline` means the deployed artifact contains the imputer,
scaler, one-hot encoder and estimator. The API sends raw values and gets a
probability - one code path, no skew.

## 7. What does the monitoring layer actually measure?

Three things: traffic (volume, latency), behaviour (risk distribution, model
version mix) and input drift (PSI per feature against the training reference).
It does *not* measure accuracy: that needs the churn outcome, which arrives
weeks later. A complete loop joins outcomes back to `prediction_logs` and
recomputes precision/recall; that is the documented next step.

## 8. What would you improve next?

- Probability calibration (isotonic / Platt) so the score can be multiplied by
  customer lifetime value to prioritise revenue at risk.
- SHAP explanations for local, additive feature attribution.
- Outcome ingestion plus scheduled performance reports.
- Experiment tracking (MLflow) and a model registry instead of a versioned
  timestamp string.
- Fairness audit across gender and senior-citizen segments.
