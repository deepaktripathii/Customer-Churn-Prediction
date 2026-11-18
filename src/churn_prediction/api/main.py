"""FastAPI application: prediction service and monitoring endpoints (Feature 7).

Run locally::

    uvicorn churn_prediction.api.main:app --reload

The app degrades gracefully: if no model artifact exists yet, ``/health``
reports ``model_loaded: false`` and ``/predict`` returns HTTP 503 with an
actionable message instead of crashing the process.
"""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from churn_prediction import __version__
from churn_prediction.api.schemas import (
    SAMPLE_CUSTOMER,
    CustomerFeatures,
    DriftResponse,
    HealthResponse,
    ModelInfoResponse,
    MonitoringSummary,
    PredictionLogEntry,
    PredictionResponse,
)
from churn_prediction.config import Settings, get_settings
from churn_prediction.logging_config import get_logger
from churn_prediction.models.predictor import ChurnPredictor, ModelNotTrainedError
from churn_prediction.models.registry import describe_models
from churn_prediction.monitoring.drift import compute_drift
from churn_prediction.monitoring.store import MonitoringStore

logger = get_logger(__name__)

DESCRIPTION = """
Customer Churn Prediction turns a churn model into a production service.

* `POST /predict` - score a customer and explain which fields drive the risk
* `GET /model-info` - served model, threshold, metrics and the model registry
* `GET /monitoring/*` - traffic, latency, risk distribution and PSI drift

Every prediction is persisted to the monitoring store together with its
payload, so the dashboard can show live traffic and distribution drift.
"""


def _serialisable_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("Unable to read metrics file %s", path)
        return {}


def create_app(settings: Settings | None = None) -> FastAPI:
    """Application factory (used by uvicorn and by the test-suite)."""
    cfg = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        cfg.ensure_directories()
        app.state.settings = cfg
        app.state.store = MonitoringStore(settings=cfg)
        try:
            app.state.predictor = ChurnPredictor(settings=cfg)
            app.state.model_error = None
            logger.info("Prediction service ready with model %s", app.state.predictor.version)
        except ModelNotTrainedError as exc:
            app.state.predictor = None
            app.state.model_error = str(exc)
            logger.warning("Prediction service started without a model: %s", exc)
        yield
        app.state.predictor = None

    app = FastAPI(
        title=cfg.api_title,
        version=cfg.api_version,
        description=DESCRIPTION,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):  # type: ignore[no-untyped-def]
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - started) * 1000:.2f}"
        return response

    def _predictor(request: Request) -> ChurnPredictor:
        predictor = getattr(request.app.state, "predictor", None)
        if predictor is None:
            detail = getattr(request.app.state, "model_error", None) or (
                "No trained model available."
            )
            raise HTTPException(
                status_code=503,
                detail=f"{detail} Train one with `python scripts/train.py`.",
            )
        return predictor

    def _store(request: Request) -> MonitoringStore:
        store = getattr(request.app.state, "store", None)
        if store is None:
            raise HTTPException(status_code=503, detail="Monitoring store is not available.")
        return store

    # ------------------------------------------------------------------ #
    # Basic endpoints
    # ------------------------------------------------------------------ #
    @app.get("/", tags=["meta"])
    def root() -> dict[str, str]:
        return {
            "name": "Customer Churn Prediction API",
            "version": __version__,
            "docs": "/docs",
            "health": "/health",
        }

    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    def health(request: Request) -> HealthResponse:
        predictor = getattr(request.app.state, "predictor", None)
        database_status = "ok"
        predictions = 0
        try:
            predictions = _store(request).count()
        except Exception:  # noqa: BLE001 - health must never raise
            database_status = "unavailable"

        return HealthResponse(
            status="ok" if predictor is not None and database_status == "ok" else "degraded",
            model_loaded=predictor is not None,
            model_name=predictor.model_name if predictor else None,
            model_version=predictor.version if predictor else None,
            database=database_status,
            predictions_logged=predictions,
            version=__version__,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    @app.get("/model-info", response_model=ModelInfoResponse, tags=["model"])
    def model_info(request: Request) -> ModelInfoResponse:
        predictor = _predictor(request)
        info = predictor.info()
        return ModelInfoResponse(
            **info,
            registry=describe_models(),
        )

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    @app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
    def predict(features: CustomerFeatures, request: Request) -> PredictionResponse:
        predictor = _predictor(request)
        payload = features.to_model_payload()

        try:
            prediction = predictor.predict(payload)
            explanations = predictor.explain(payload)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Prediction failed")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

        try:
            _store(request).log_prediction(
                payload=payload,
                churn_probability=prediction.churn_probability,
                prediction=prediction.prediction,
                risk_level=prediction.risk_level,
                model_name=prediction.model_name,
                model_version=prediction.model_version,
                latency_ms=prediction.latency_ms or 0.0,
            )
        except Exception:  # noqa: BLE001 - logging must not break a prediction
            logger.exception("Failed to persist prediction log")

        return PredictionResponse(
            prediction=prediction.prediction,
            churn_probability=prediction.churn_probability,
            risk_level=prediction.risk_level,
            threshold=prediction.threshold,
            model_name=prediction.model_name,
            model_version=prediction.model_version,
            latency_ms=prediction.latency_ms or 0.0,
            explanations=[item.to_dict() for item in explanations],
        )

    @app.post("/predict/sample", response_model=PredictionResponse, tags=["prediction"])
    def predict_sample(request: Request) -> PredictionResponse:
        """Score the documented sample customer (handy for smoke tests)."""
        return predict(CustomerFeatures(**SAMPLE_CUSTOMER), request)

    # ------------------------------------------------------------------ #
    # Monitoring
    # ------------------------------------------------------------------ #
    @app.get("/monitoring/summary", response_model=MonitoringSummary, tags=["monitoring"])
    def monitoring_summary(request: Request) -> MonitoringSummary:
        return MonitoringSummary(**_store(request).summary())

    @app.get(
        "/monitoring/predictions",
        response_model=list[PredictionLogEntry],
        tags=["monitoring"],
    )
    def monitoring_predictions(
        request: Request,
        limit: int = Query(default=50, ge=1, le=500),
    ) -> list[PredictionLogEntry]:
        return [PredictionLogEntry(**row) for row in _store(request).recent(limit)]

    @app.get("/monitoring/drift", response_model=DriftResponse, tags=["monitoring"])
    def monitoring_drift(
        request: Request,
        limit: int = Query(default=500, ge=1, le=5000),
        min_samples: int = Query(default=20, ge=5, le=500),
    ) -> DriftResponse:
        store = _store(request)
        reference_path = getattr(request.app.state.settings, "reference_stats_path", None)
        if reference_path is None or not Path(reference_path).exists():
            raise HTTPException(
                status_code=503,
                detail="Reference statistics missing. Train the model first.",
            )
        reference_stats = json.loads(Path(reference_path).read_text(encoding="utf-8"))
        result = compute_drift(
            reference_stats, store.payloads(limit=limit), min_samples=min_samples
        )
        return DriftResponse(**result)

    return app


app = create_app()


__all__ = ["app", "create_app"]
