"""Feature 7: FastAPI endpoints and prediction logging."""

from __future__ import annotations

from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from retainiq.api.main import create_app


@pytest.fixture()
def client(trained_settings, tmp_path):
    settings = replace(
        trained_settings,
        database_url=f"sqlite:///{(tmp_path / 'monitoring.db').as_posix()}",
    )
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


def test_root_and_health(client):
    root = client.get("/")
    assert root.status_code == 200
    assert root.json()["docs"] == "/docs"

    health = client.get("/health")
    assert health.status_code == 200
    body = health.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["database"] == "ok"
    assert body["predictions_logged"] == 0


def test_model_info(client):
    response = client.get("/model-info")

    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "logistic_regression"
    assert body["threshold"] == pytest.approx(body["threshold"])
    assert body["registry"]
    assert body["features"]


def test_predict_returns_probability_and_explanations(client):
    response = client.post(
        "/predict",
        json={
            "tenure": 3,
            "monthly_charges": 1200,
            "contract": "Month-to-month",
            "internet_service": "Fiber optic",
            "payment_method": "Electronic check",
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["prediction"] in {"Churn", "No Churn"}
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_level"] in {"Low", "Medium", "High"}
    assert isinstance(body["explanations"], list)
    assert "X-Process-Time-Ms" in response.headers


def test_predict_persists_log(client):
    client.post("/predict", json={"tenure": 5, "monthly_charges": 90, "contract": "One year"})
    client.post("/predict", json={"tenure": 40, "monthly_charges": 60, "contract": "Two year"})

    summary = client.get("/monitoring/summary").json()
    assert summary["total_predictions"] == 2
    assert summary["risk_distribution"]["Low"] + summary["risk_distribution"]["Medium"] + summary["risk_distribution"]["High"] == 2

    recent = client.get("/monitoring/predictions", params={"limit": 10}).json()
    assert len(recent) == 2
    assert recent[0]["payload"]["tenure"] == 40


def test_predict_validates_payload(client):
    response = client.post(
        "/predict",
        json={"tenure": -5, "monthly_charges": 100, "contract": "Monthly"},
    )

    assert response.status_code == 422


def test_predict_rejects_unknown_fields(client):
    response = client.post(
        "/predict",
        json={"tenure": 5, "monthly_charges": 90, "contract": "One year", "shoe_size": 9},
    )

    assert response.status_code == 422


def test_drift_endpoint_handles_low_volume(client):
    client.post("/predict", json={"tenure": 5, "monthly_charges": 90, "contract": "One year"})

    response = client.get("/monitoring/drift")

    assert response.status_code == 200
    assert response.json()["status"] == "insufficient_data"


def test_sample_endpoint(client):
    response = client.post("/predict/sample")

    assert response.status_code == 200
    assert response.json()["latency_ms"] >= 0


def test_health_reports_missing_model(tmp_path):
    from retainiq.config import get_settings

    empty_settings = replace(
        get_settings().with_artifact_dir(tmp_path / "empty"),
        database_url=f"sqlite:///{(tmp_path / 'empty.db').as_posix()}",
    )
    app = create_app(empty_settings)
    with TestClient(app) as test_client:
        health = test_client.get("/health").json()
        assert health["status"] == "degraded"
        assert health["model_loaded"] is False

        response = test_client.post("/predict/sample")
        assert response.status_code == 503
