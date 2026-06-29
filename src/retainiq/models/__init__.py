"""Model package: registry, training, evaluation and inference."""

from retainiq.models.evaluate import (
    classification_metrics,
    comparison_table,
    evaluate_pipeline,
    evaluate_scores,
    pr_curve_points,
    roc_curve_points,
)
from retainiq.models.predictor import (
    ChurnPredictor,
    ModelNotTrainedError,
    Prediction,
)
from retainiq.models.registry import (
    MODEL_REGISTRY,
    ModelSpec,
    available_models,
    describe_models,
    get_model_spec,
)
from retainiq.models.train import (
    ModelRun,
    TrainingError,
    build_reference_stats,
    infer_defaults,
    optimal_threshold,
    save_artifacts,
    select_best,
    train_and_save,
    train_models,
)

__all__ = [
    "MODEL_REGISTRY",
    "ChurnPredictor",
    "ModelNotTrainedError",
    "ModelRun",
    "ModelSpec",
    "Prediction",
    "TrainingError",
    "available_models",
    "build_reference_stats",
    "classification_metrics",
    "comparison_table",
    "describe_models",
    "evaluate_pipeline",
    "evaluate_scores",
    "get_model_spec",
    "infer_defaults",
    "optimal_threshold",
    "pr_curve_points",
    "roc_curve_points",
    "save_artifacts",
    "select_best",
    "train_and_save",
    "train_models",
]
