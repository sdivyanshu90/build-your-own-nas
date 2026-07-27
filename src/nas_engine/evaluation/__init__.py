"""Candidate evaluation: budgets, measurement, and results."""

from nas_engine.evaluation.budget import TrainingBudget
from nas_engine.evaluation.evaluator import (
    CandidateEvaluator,
    EvaluationContext,
    EvaluationSettings,
)
from nas_engine.evaluation.latency import (
    LATENCY_WARNING,
    LatencyMeasurement,
    measure_latency,
)
from nas_engine.evaluation.model_size import (
    ModelSizeMeasurement,
    measure_model_size,
    save_model_weights,
)
from nas_engine.evaluation.result import (
    EvaluationFailure,
    EvaluationResult,
    FailureKind,
    classify_failure,
)

__all__ = [
    "LATENCY_WARNING",
    "CandidateEvaluator",
    "EvaluationContext",
    "EvaluationFailure",
    "EvaluationResult",
    "EvaluationSettings",
    "FailureKind",
    "LatencyMeasurement",
    "ModelSizeMeasurement",
    "TrainingBudget",
    "classify_failure",
    "measure_latency",
    "measure_model_size",
    "save_model_weights",
]
