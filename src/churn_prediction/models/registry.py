"""Model registry (Feature 4).

Each supported algorithm is described by a :class:`ModelSpec` so the training
script, the API and the dashboard can list models without importing sklearn
estimators themselves.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from churn_prediction.config import get_settings


@dataclass(frozen=True)
class ModelSpec:
    name: str
    display_name: str
    family: str
    description: str
    builder: Callable[[int], Any]
    requires_scaling: bool = True


def _build_logistic_regression(random_state: int) -> Any:
    return LogisticRegression(
        C=1.0,
        max_iter=2000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=random_state,
    )


def _build_random_forest(random_state: int) -> Any:
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=14,
        min_samples_leaf=4,
        min_samples_split=8,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=random_state,
    )


def _build_xgboost(random_state: int) -> Any:
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_weight=2,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=random_state,
    )


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "logistic_regression": ModelSpec(
        name="logistic_regression",
        display_name="Logistic Regression",
        family="linear",
        description=(
            "Interpretable baseline. Coefficients can be read as log-odds, which "
            "makes it ideal for explaining retention decisions to business users."
        ),
        builder=_build_logistic_regression,
    ),
    "random_forest": ModelSpec(
        name="random_forest",
        display_name="Random Forest",
        family="tree-ensemble",
        description=(
            "Bagged decision trees that capture non-linear interactions such as "
            "contract type x tenure without manual crossing."
        ),
        builder=_build_random_forest,
    ),
    "xgboost": ModelSpec(
        name="xgboost",
        display_name="XGBoost",
        family="gradient-boosting",
        description=(
            "Optional third model. Gradient boosted trees usually give the best "
            "ranking quality on tabular churn data at the cost of interpretability."
        ),
        builder=_build_xgboost,
    ),
}


def available_models(*, include_optional: bool = True) -> list[str]:
    """Names of the models that can be trained in the current environment."""
    names: list[str] = []
    for name, spec in MODEL_REGISTRY.items():
        if name == "xgboost" and not include_optional:
            continue
        try:
            spec.builder(get_settings().random_state)
        except ImportError:  # optional dependency missing
            continue
        names.append(name)
    return names


def get_model_spec(name: str) -> ModelSpec:
    try:
        return MODEL_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown model '{name}'. Available models: {', '.join(available_models())}"
        ) from exc


def describe_models() -> list[dict[str, str]]:
    """Serialisable descriptions for ``/model-info`` and the dashboard."""
    descriptions = []
    for name in available_models():
        spec = MODEL_REGISTRY[name]
        descriptions.append(
            {
                "name": spec.name,
                "display_name": spec.display_name,
                "family": spec.family,
                "description": spec.description,
            }
        )
    return descriptions


__all__ = [
    "MODEL_REGISTRY",
    "ModelSpec",
    "available_models",
    "describe_models",
    "get_model_spec",
]
