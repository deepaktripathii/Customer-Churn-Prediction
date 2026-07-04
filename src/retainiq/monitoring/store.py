"""Prediction logging and monitoring aggregates (Feature 7).

Every prediction is persisted with its payload, probability, risk band and
latency. SQLite is the default so the project runs with zero infrastructure;
swapping ``RETAINIQ_DATABASE_URL`` for a PostgreSQL DSN is enough for a shared
deployment (SQLAlchemy handles the dialect).
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import DateTime, Float, Integer, String, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.types import JSON

from retainiq.config import Settings, get_settings
from retainiq.logging_config import get_logger

logger = get_logger(__name__)


class Base(DeclarativeBase):
    pass


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )
    model_name: Mapped[str] = mapped_column(String(64))
    model_version: Mapped[str] = mapped_column(String(64))
    prediction: Mapped[str] = mapped_column(String(16))
    risk_level: Mapped[str] = mapped_column(String(16), index=True)
    churn_probability: Mapped[float] = mapped_column(Float)
    latency_ms: Mapped[float] = mapped_column(Float)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)

    def to_dict(self) -> dict[str, Any]:
        created = self.created_at
        if created is not None and created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        return {
            "id": self.id,
            "created_at": created.isoformat() if created else None,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "prediction": self.prediction,
            "risk_level": self.risk_level,
            "churn_probability": round(float(self.churn_probability), 4),
            "latency_ms": round(float(self.latency_ms), 2),
            "payload": self.payload,
        }


def _jsonify(value: Any) -> Any:
    """Best effort conversion of numpy scalars / datetimes into JSON types."""
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


class MonitoringStore:
    """Thin repository around the prediction log table."""

    def __init__(
        self,
        database_url: str | None = None,
        *,
        settings: Settings | None = None,
        echo: bool = False,
    ) -> None:
        cfg = settings or get_settings()
        self.settings = cfg
        self.database_url = database_url or cfg.database_url

        if self.database_url.startswith("sqlite:///"):
            db_path = Path(self.database_url[len("sqlite:///") :])
            if str(db_path) != ":memory:":
                db_path.parent.mkdir(parents=True, exist_ok=True)

        connect_args = {"check_same_thread": False} if self.database_url.startswith("sqlite") else {}
        self.engine = create_engine(self.database_url, echo=echo, future=True, connect_args=connect_args)
        self._session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        Base.metadata.create_all(self.engine)

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #
    def log_prediction(
        self,
        *,
        payload: dict[str, Any],
        churn_probability: float,
        prediction: str,
        risk_level: str,
        model_name: str,
        model_version: str,
        latency_ms: float,
        created_at: datetime | None = None,
    ) -> dict[str, Any]:
        row = PredictionLog(
            created_at=created_at or datetime.now(timezone.utc),
            model_name=model_name,
            model_version=model_version,
            prediction=prediction,
            risk_level=risk_level,
            churn_probability=float(churn_probability),
            latency_ms=float(latency_ms),
            payload=_jsonify(dict(payload)),
        )
        with self._session_factory() as session:
            session.add(row)
            session.commit()
            session.refresh(row)
            return row.to_dict()

    def reset(self) -> None:
        with self._session_factory() as session:
            session.query(PredictionLog).delete()
            session.commit()

    # ------------------------------------------------------------------ #
    # Reads
    # ------------------------------------------------------------------ #
    def count(self) -> int:
        with self._session_factory() as session:
            return int(session.scalar(select(func.count(PredictionLog.id))) or 0)

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._session_factory() as session:
            rows = session.scalars(
                select(PredictionLog).order_by(PredictionLog.id.desc()).limit(limit)
            ).all()
        return [row.to_dict() for row in rows]

    def payloads(self, limit: int = 500) -> list[dict[str, Any]]:
        """Most recent raw payloads, used as input for drift detection."""
        with self._session_factory() as session:
            rows = session.execute(
                select(PredictionLog.payload)
                .order_by(PredictionLog.id.desc())
                .limit(limit)
            ).all()
        return [row[0] for row in rows if row[0]]

    def summary(self, *, recent_limit: int = 1000) -> dict[str, Any]:
        """Aggregated monitoring view for the dashboard."""
        with self._session_factory() as session:
            rows = session.scalars(
                select(PredictionLog).order_by(PredictionLog.id.desc()).limit(recent_limit)
            ).all()
            total = int(session.scalar(select(func.count(PredictionLog.id))) or 0)

        if not rows:
            return {
                "total_predictions": 0,
                "window_size": 0,
                "churn_predictions": 0,
                "churn_rate": None,
                "avg_churn_probability": None,
                "risk_distribution": {},
                "model_versions": {},
                "avg_latency_ms": None,
                "p95_latency_ms": None,
                "first_prediction_at": None,
                "last_prediction_at": None,
                "predictions_last_24h": 0,
                "daily": [],
            }

        probabilities = np.asarray([row.churn_probability for row in rows], dtype=float)
        latencies = np.asarray([row.latency_ms for row in rows], dtype=float)
        risk_counts = Counter(row.risk_level for row in rows)
        version_counts = Counter(row.model_version for row in rows)
        now = datetime.now(timezone.utc)

        def _as_utc(value: datetime) -> datetime:
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)

        last_24h = sum(1 for row in rows if _as_utc(row.created_at) >= now - timedelta(hours=24))

        daily_buckets: dict[str, list[float]] = {}
        for row in rows:
            key = _as_utc(row.created_at).date().isoformat()
            daily_buckets.setdefault(key, []).append(row.churn_probability)
        daily = [
            {
                "date": key,
                "count": len(values),
                "avg_churn_probability": round(float(np.mean(values)), 4),
            }
            for key, values in sorted(daily_buckets.items())
        ]

        return {
            "total_predictions": total,
            "window_size": len(rows),
            "churn_predictions": int((probabilities >= 0.5).sum()),
            "churn_rate": round(float(np.mean(probabilities >= 0.5)), 4),
            "avg_churn_probability": round(float(probabilities.mean()), 4),
            "risk_distribution": {
                level: int(risk_counts.get(level, 0)) for level in ("Low", "Medium", "High")
            },
            "model_versions": dict(version_counts),
            "avg_latency_ms": round(float(latencies.mean()), 2),
            "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
            "first_prediction_at": _as_utc(min(row.created_at for row in rows)).isoformat(),
            "last_prediction_at": _as_utc(max(row.created_at for row in rows)).isoformat(),
            "predictions_last_24h": int(last_24h),
            "daily": daily,
        }


def export_summary_json(store: MonitoringStore, path: str | Path) -> Path:
    """Write the summary to disk (handy for CI artefacts)."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(store.summary(), indent=2), encoding="utf-8")
    return destination


__all__ = ["MonitoringStore", "PredictionLog", "export_summary_json"]
