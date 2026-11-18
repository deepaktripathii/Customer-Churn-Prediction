"""Central configuration for the Customer Churn Prediction platform.

Everything that touches the filesystem, the database or the model registry is
resolved here so that scripts, the API, the dashboard and the test-suite all
agree on the same paths. Values can be overridden with environment variables
(see ``.env.example``); the ``Settings`` object is cached but can be reset with
:func:`reset_settings_cache`, which the tests use to point the project at
temporary directories.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

# --------------------------------------------------------------------------- #
# Constants shared across modules
# --------------------------------------------------------------------------- #
TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

#: Ordered risk bands used by the prediction service and the dashboard.
RISK_LEVELS = ("Low", "Medium", "High")

#: URL of the public IBM Telco Customer Churn dataset used by the project.
TELCO_DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

TELCO_DATASET_MIRRORS = (
    TELCO_DATASET_URL,
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/main/"
    "data/Telco-Customer-Churn.csv",
)


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    return Path(raw).expanduser().resolve() if raw else default


def _project_root() -> Path:
    """Repository root, i.e. the folder that contains ``src/``."""
    env_root = os.getenv("CHURN_PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    """Immutable view of the runtime configuration."""

    project_root: Path
    data_dir: Path
    raw_data_path: Path
    artifact_dir: Path
    model_dir: Path
    metrics_path: Path
    model_card_path: Path
    reference_stats_path: Path
    database_url: str
    default_model: str
    random_state: int = 42
    test_size: float = 0.2
    validation_size: float = 0.2
    cv_folds: int = 5
    low_risk_max: float = 0.35
    high_risk_min: float = 0.65
    api_title: str = "Customer Churn Prediction - Customer Retention Intelligence API"
    api_version: str = "1.0.0"

    # -- helpers ---------------------------------------------------------- #
    def with_artifact_dir(self, directory: Path | str) -> Settings:
        """Return a copy of the settings that writes artifacts elsewhere."""
        directory = Path(directory).expanduser().resolve()
        return replace(
            self,
            artifact_dir=directory,
            model_dir=directory / "models",
            metrics_path=directory / "metrics.json",
            model_card_path=directory / "model_card.json",
            reference_stats_path=directory / "reference_stats.json",
        )

    @property
    def monitoring_db_path(self) -> Path | None:
        """Filesystem path of the SQLite database, if SQLite is in use."""
        prefix = "sqlite:///"
        if self.database_url.startswith(prefix):
            return Path(self.database_url[len(prefix) :]).expanduser()
        return None

    def ensure_directories(self) -> None:
        for path in (self.data_dir / "raw", self.artifact_dir, self.model_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    root = _project_root()
    data_dir = _env_path("CHURN_DATA_DIR", root / "data")
    artifact_dir = _env_path("CHURN_ARTIFACT_DIR", root / "artifacts")
    default_db = f"sqlite:///{(artifact_dir / 'monitoring.db').as_posix()}"

    return Settings(
        project_root=root,
        data_dir=data_dir,
        raw_data_path=_env_path("CHURN_RAW_DATA_PATH", data_dir / "raw" / "telco_churn.csv"),
        artifact_dir=artifact_dir,
        model_dir=_env_path("CHURN_MODEL_DIR", artifact_dir / "models"),
        metrics_path=_env_path("CHURN_METRICS_PATH", artifact_dir / "metrics.json"),
        model_card_path=_env_path("CHURN_MODEL_CARD_PATH", artifact_dir / "model_card.json"),
        reference_stats_path=_env_path(
            "CHURN_REFERENCE_STATS_PATH", artifact_dir / "reference_stats.json"
        ),
        database_url=os.getenv("CHURN_DATABASE_URL", default_db),
        default_model=os.getenv("CHURN_DEFAULT_MODEL", "random_forest"),
    )


def reset_settings_cache() -> None:
    """Drop the cached settings (used by tests and long running notebooks)."""
    get_settings.cache_clear()


def risk_level(probability: float, settings: Settings | None = None) -> str:
    """Map a churn probability to a human friendly risk band.

    The defaults (0.35 / 0.65) are deliberately conservative: a customer is
    only flagged as *High* risk when the model is clearly confident, because a
    retention offer costs money and should not be sent to everyone.
    """
    cfg = settings or get_settings()
    if probability < cfg.low_risk_max:
        return "Low"
    if probability < cfg.high_risk_min:
        return "Medium"
    return "High"


__all__ = [
    "ID_COLUMN",
    "RISK_LEVELS",
    "Settings",
    "TARGET_COLUMN",
    "TELCO_DATASET_MIRRORS",
    "TELCO_DATASET_URL",
    "get_settings",
    "reset_settings_cache",
    "risk_level",
]
