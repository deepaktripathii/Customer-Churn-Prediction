"""Customer Churn Prediction Streamlit dashboard.

Run with::

    streamlit run src/churn_prediction/dashboard/app.py

The dashboard reads training artifacts locally and talks to the FastAPI service
when it is reachable, so it still works in a notebook-only environment.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd  # noqa: E402
import requests  # noqa: E402
import streamlit as st  # noqa: E402

from churn_prediction.config import get_settings  # noqa: E402
from churn_prediction.data.loader import DatasetNotFoundError, load_dataset, profile_dataset  # noqa: E402
from churn_prediction.data.quality import run_quality_checks  # noqa: E402
from churn_prediction.models.predictor import ChurnPredictor, ModelNotTrainedError  # noqa: E402
from churn_prediction.monitoring.drift import compute_drift  # noqa: E402
from churn_prediction.monitoring.store import MonitoringStore  # noqa: E402

st.set_page_config(
    page_title="Customer Churn Prediction | Churn Intelligence",
    page_icon="📉",
    layout="wide",
)

SETTINGS = get_settings()


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    path = SETTINGS.metrics_path
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


@st.cache_data(show_spinner=False)
def load_reference_stats() -> dict:
    path = SETTINGS.reference_stats_path
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


@st.cache_data(show_spinner=True)
def load_dataset_frame() -> pd.DataFrame | None:
    try:
        return load_dataset()
    except DatasetNotFoundError:
        return None


@st.cache_resource(show_spinner=False)
def get_predictor(model_name: str | None = None) -> ChurnPredictor | None:
    try:
        return ChurnPredictor(model_name=model_name, settings=SETTINGS)
    except ModelNotTrainedError:
        return None


def api_get(api_url: str, path: str, params: dict | None = None) -> dict | list | None:
    try:
        response = requests.get(f"{api_url.rstrip('/')}{path}", params=params, timeout=4)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def api_post(api_url: str, path: str, payload: dict) -> dict | None:
    try:
        response = requests.post(f"{api_url.rstrip('/')}{path}", json=payload, timeout=6)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("📉 Customer Churn Prediction")
    st.caption("Customer retention intelligence platform")
    page = st.radio(
        "Navigate",
        ["Overview", "Model performance", "Predict customer", "Monitoring"],
        label_visibility="collapsed",
    )
    st.divider()
    api_url = st.text_input(
        "API base URL",
        value=os.getenv("CHURN_API_URL", "http://localhost:8000"),
        help="Used for live prediction and monitoring; local artifacts are used as fallback.",
    )
    api_health = api_get(api_url, "/health")
    if api_health:
        st.success(f"API online · {api_health.get('model_name') or 'no model'}")
    else:
        st.warning("API offline · using local artifacts")
    st.divider()
    st.caption(f"Artifacts: `{SETTINGS.artifact_dir}`")


metrics = load_metrics()


# --------------------------------------------------------------------------- #
# Overview
# --------------------------------------------------------------------------- #
def render_overview() -> None:
    st.header("Project overview")
    st.write(
        "Customer Churn Prediction predicts which telecom customers are about to churn, explains the "
        "drivers behind each score, and monitors the model in production."
    )

    df = load_dataset_frame()
    if df is None:
        st.error(
            f"Dataset not found at `{SETTINGS.raw_data_path}`. "
            "Run `python scripts/download_data.py` first."
        )
        return

    profile = profile_dataset(df, source=str(SETTINGS.raw_data_path), preview_rows=5)
    churn_rate = None
    if profile.target_distribution:
        total = sum(profile.target_distribution.values())
        churned = profile.target_distribution.get("Yes", 0)
        churn_rate = churned / total if total else None

    columns = st.columns(4)
    columns[0].metric("Customers", f"{profile.n_rows:,}")
    columns[1].metric("Columns", f"{profile.n_columns}")
    columns[2].metric(
        "Churn rate", f"{churn_rate:.1%}" if churn_rate is not None else "n/a"
    )
    columns[3].metric("Missing cells", f"{profile.total_missing:,}")

    left, right = st.columns([1.4, 1])
    with left:
        st.subheader("Raw data preview")
        st.dataframe(df.head(20), width="stretch", height=380)
    with right:
        st.subheader("Target balance")
        if profile.target_distribution:
            st.bar_chart(
                pd.Series(profile.target_distribution, name="customers"),
                horizontal=True,
            )
        st.subheader("Data quality")
        quality = run_quality_checks(df)
        if quality.warnings:
            for issue in quality.warnings:
                st.warning(issue.message)
        else:
            st.success("No warnings - dataset looks clean.")
        st.caption(f"Duplicate rows: {profile.duplicate_rows}")

    with st.expander("Column dictionary"):
        st.dataframe(
            pd.DataFrame(
                {
                    "column": profile.columns,
                    "dtype": [profile.dtypes[c] for c in profile.columns],
                    "missing": [profile.missing_values.get(c, 0) for c in profile.columns],
                }
            ),
            width="stretch",
            height=420,
        )

    if metrics.get("cleaning"):
        cleaning = metrics["cleaning"]
        with st.expander("Last cleaning run"):
            st.json(
                {
                    "rows_in": cleaning.get("rows_in"),
                    "rows_out": cleaning.get("rows_out"),
                    "duplicates_removed": cleaning.get("duplicates_removed"),
                    "missing_filled": cleaning.get("missing_filled"),
                    "notes": cleaning.get("notes"),
                }
            )


# --------------------------------------------------------------------------- #
# Model performance
# --------------------------------------------------------------------------- #
def render_performance() -> None:
    st.header("Model performance")
    if not metrics.get("models"):
        st.info("No trained models yet. Run `python scripts/train.py`.")
        return

    models = metrics["models"]
    comparison = pd.DataFrame(metrics.get("comparison", []))
    if comparison.empty:
        comparison = pd.DataFrame(
            [
                {
                    "model": name,
                    "accuracy": payload["test"]["accuracy"],
                    "precision": payload["test"]["precision"],
                    "recall": payload["test"]["recall"],
                    "f1": payload["test"]["f1"],
                    "roc_auc": payload["test"]["roc_auc"],
                    "pr_auc": payload["test"]["pr_auc"],
                    "threshold": payload["threshold"],
                }
                for name, payload in models.items()
            ]
        )

    best = metrics.get("best_model")
    st.caption(
        f"Run `{metrics.get('run_id', 'n/a')}` · best model **{best}** · "
        "metrics measured on a held-out test split"
    )
    st.dataframe(
        comparison.style.format(
            {
                "accuracy": "{:.3f}",
                "precision": "{:.3f}",
                "recall": "{:.3f}",
                "f1": "{:.3f}",
                "roc_auc": "{:.3f}",
                "pr_auc": "{:.3f}",
                "threshold": "{:.3f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    selected = st.selectbox("Inspect model", list(models), index=list(models).index(best) if best in models else 0)
    payload = models[selected]
    test = payload["test"]
    validation = payload.get("validation", {})

    cols = st.columns(6)
    cols[0].metric("Accuracy", f"{test['accuracy']:.3f}")
    cols[1].metric("Precision", f"{test['precision']:.3f}")
    cols[2].metric("Recall", f"{test['recall']:.3f}")
    cols[3].metric("F1", f"{test['f1']:.3f}")
    cols[4].metric("ROC AUC", f"{test['roc_auc']:.3f}")
    cols[5].metric("Threshold", f"{payload['threshold']:.3f}")

    st.caption(
        f"Validation F1 {validation.get('f1', float('nan')):.3f} · "
        f"CV ROC AUC {payload.get('cv_roc_auc') or float('nan')} · "
        f"trained in {payload.get('train_seconds', 0)}s"
    )

    curve_left, curve_right = st.columns(2)
    roc = pd.DataFrame(test.get("roc_curve", []))
    pr = pd.DataFrame(test.get("pr_curve", []))
    with curve_left:
        st.subheader("ROC curve")
        if not roc.empty:
            st.line_chart(roc, x="fpr", y="tpr", height=300)
        else:
            st.info("ROC curve unavailable for this run.")
    with curve_right:
        st.subheader("Precision / recall")
        if not pr.empty:
            st.line_chart(pr, x="recall", y="precision", height=300)
        else:
            st.info("PR curve unavailable for this run.")

    st.subheader("Confusion matrix (test set)")
    cm = test["confusion_matrix"]
    matrix = pd.DataFrame(
        [
            [cm["true_negative"], cm["false_positive"]],
            [cm["false_negative"], cm["true_positive"]],
        ],
        index=["Actual retained", "Actual churned"],
        columns=["Predicted retained", "Predicted churn"],
    )
    st.dataframe(matrix, width="stretch")
    st.caption(
        f"Recall matters most here: {cm['false_negative']} real churners were missed. "
        "Lower the threshold to catch more of them at the cost of more false alarms."
    )

    with st.expander("Features seen by the model"):
        st.write(metrics.get("features", {}).get("model_columns", []))


# --------------------------------------------------------------------------- #
# Predict
# --------------------------------------------------------------------------- #
def render_predict() -> None:
    st.header("Score a customer")
    st.caption(
        "Only tenure, monthly charges and contract are required. Empty fields fall back "
        "to the training-time typical customer."
    )

    with st.form("predict-form"):
        left, middle, right = st.columns(3)
        tenure = left.number_input("Tenure (months)", min_value=0, max_value=120, value=3)
        monthly = middle.number_input("Monthly charges", min_value=1.0, value=1200.0, step=50.0)
        contract = right.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

        left, middle, right = st.columns(3)
        internet = left.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        payment = middle.selectbox(
            "Payment method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        paperless = right.selectbox("Paperless billing", ["Yes", "No"])

        left, middle, right = st.columns(3)
        online_security = left.selectbox("Online security", ["No", "Yes", "No internet service"])
        tech_support = middle.selectbox("Tech support", ["No", "Yes", "No internet service"])
        device_protection = right.selectbox(
            "Device protection", ["No", "Yes", "No internet service"]
        )

        submitted = st.form_submit_button("Predict churn risk", type="primary")

    if not submitted:
        return

    payload = {
        "tenure": int(tenure),
        "monthly_charges": float(monthly),
        "contract": contract,
        "internet_service": internet,
        "payment_method": payment,
        "paperless_billing": paperless,
        "online_security": online_security,
        "tech_support": tech_support,
        "device_protection": device_protection,
    }

    result = api_post(api_url, "/predict", payload)
    source = "API"
    if result is None:
        predictor = get_predictor()
        if predictor is None:
            st.error("No model available. Train one with `python scripts/train.py`.")
            return
        prediction = predictor.predict(payload)
        result = {
            **prediction.to_dict(),
            "explanations": [item.to_dict() for item in predictor.explain(payload)],
        }
        source = "local artifact"

    probability = result["churn_probability"]
    risk = result["risk_level"]
    emoji = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}.get(risk, "⚪")

    cols = st.columns(3)
    cols[0].metric("Churn probability", f"{probability:.1%}")
    cols[1].metric("Risk level", f"{emoji} {risk}")
    cols[2].metric("Decision", result["prediction"])
    st.caption(
        f"Model `{result['model_name']}` ({result['model_version']}) · "
        f"threshold {result['threshold']:.3f} · source: {source}"
    )

    st.progress(min(max(probability, 0.0), 1.0))

    explanations = result.get("explanations") or []
    if explanations:
        st.subheader("Why this score?")
        frame = pd.DataFrame(explanations)
        st.dataframe(
            frame[["label", "value", "direction", "impact"]].rename(
                columns={
                    "label": "Factor",
                    "value": "Value",
                    "direction": "Effect",
                    "impact": "Probability impact",
                }
            ),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "Impact is the change in churn probability relative to the typical customer, "
            "computed by switching one field at a time."
        )
    else:
        st.info("This customer matches the typical profile, so there are no strong drivers.")

    with st.expander("Raw response"):
        st.json(result)


# --------------------------------------------------------------------------- #
# Monitoring
# --------------------------------------------------------------------------- #
def render_monitoring() -> None:
    st.header("Production monitoring")

    summary = api_get(api_url, "/monitoring/summary")
    origin = "API"
    if not summary:
        summary = MonitoringStore(settings=SETTINGS).summary()
        origin = "local database"

    total = summary.get("total_predictions", 0)
    if not total:
        st.info(
            "No predictions logged yet. Send demo traffic with "
            "`python scripts/simulate_traffic.py --n 300`."
        )
        return

    st.caption(f"Source: {origin} · {summary.get('first_prediction_at')} → {summary.get('last_prediction_at')}")
    cols = st.columns(5)
    cols[0].metric("Predictions", f"{total:,}")
    cols[1].metric("Last 24h", f"{summary.get('predictions_last_24h', 0):,}")
    cols[2].metric("Flagged churn", f"{summary.get('churn_rate', 0):.1%}")
    cols[3].metric("Avg latency", f"{summary.get('avg_latency_ms') or 0:.1f} ms")
    cols[4].metric("p95 latency", f"{summary.get('p95_latency_ms') or 0:.1f} ms")

    left, right = st.columns(2)
    with left:
        st.subheader("Risk distribution")
        risk_series = pd.Series(summary.get("risk_distribution", {}), name="predictions")
        st.bar_chart(risk_series, horizontal=True)
    with right:
        st.subheader("Daily traffic")
        daily = pd.DataFrame(summary.get("daily", []))
        if not daily.empty:
            st.line_chart(daily, x="date", y="count", height=260)
            st.line_chart(daily, x="date", y="avg_churn_probability", height=200)
        else:
            st.info("No daily history yet.")

    st.subheader("Input drift (PSI)")
    drift = api_get(api_url, "/monitoring/drift", params={"limit": 500})
    if not drift:
        reference = load_reference_stats()
        if reference:
            store = MonitoringStore(settings=SETTINGS)
            drift = compute_drift(reference, store.payloads(limit=500))
        else:
            st.info("Reference statistics missing; train the model first.")

    if drift:
        status = drift.get("status")
        message = {
            "stable": "Inputs look stable - no action needed.",
            "moderate": "Moderate drift - review recent campaigns and pricing changes.",
            "significant": "Significant drift - schedule retraining and validate performance.",
            "insufficient_data": "Not enough live samples to measure drift yet.",
        }.get(status, "")
        if status in ("moderate", "significant"):
            st.warning(message)
        elif status == "insufficient_data":
            st.info(message)
        else:
            st.success(message)

        drift_features = pd.DataFrame(drift.get("features", []))
        if not drift_features.empty:
            st.dataframe(
                drift_features.rename(
                    columns={"feature": "Feature", "type": "Type", "psi": "PSI", "status": "Status"}
                ),
                width="stretch",
                hide_index=True,
            )
        st.caption(
            f"samples: {drift.get('n_samples', 0)} · overall PSI {drift.get('overall_psi')} · "
            f"max PSI {drift.get('max_psi')}. PSI < 0.10 is stable, > 0.25 needs action."
        )

    with st.expander("Recent predictions"):
        if origin == "API":
            rows = api_get(api_url, "/monitoring/predictions", params={"limit": 50}) or []
        else:
            rows = MonitoringStore(settings=SETTINGS).recent(limit=50)
        if rows:
            frame = pd.DataFrame(rows)
            columns = [c for c in ("created_at", "model_version", "prediction", "risk_level", "churn_probability", "latency_ms") if c in frame]
            st.dataframe(frame[columns], width="stretch", height=380, hide_index=True)

    st.caption(
        "Drift only measures input distribution change. Real accuracy drift needs churn "
        "outcomes joined back to these predictions after the observation window."
    )


PAGES = {
    "Overview": render_overview,
    "Model performance": render_performance,
    "Predict customer": render_predict,
    "Monitoring": render_monitoring,
}

PAGES[page]()
