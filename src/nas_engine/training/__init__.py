"""Training: optimisers, schedules, metrics, early stopping, checkpoints, and the loop.

This package depends on :mod:`nas_engine.datasets` and PyTorch. It knows nothing about
search strategies, candidates, or persistence — a search strategy that contained training
logic would be untestable without a GPU, and that separation is enforced here by the
import graph.
"""

from nas_engine.training.checkpointing import (
    CHECKPOINT_FORMAT_VERSION,
    TrainingCheckpoint,
    load_checkpoint,
    save_checkpoint,
)
from nas_engine.training.early_stopping import EarlyStopping, MonitorMode
from nas_engine.training.metrics import (
    EpochMetrics,
    MetricAggregator,
    accuracy,
    topk_accuracy,
)
from nas_engine.training.optimizers import OptimizerSettings, OptimizerType, build_optimizer
from nas_engine.training.schedulers import SchedulerSettings, SchedulerType, build_scheduler
from nas_engine.training.trainer import (
    Trainer,
    TrainingOutcome,
    TrainingSettings,
    evaluation_mode,
)

__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "EarlyStopping",
    "EpochMetrics",
    "MetricAggregator",
    "MonitorMode",
    "OptimizerSettings",
    "OptimizerType",
    "SchedulerSettings",
    "SchedulerType",
    "Trainer",
    "TrainingCheckpoint",
    "TrainingOutcome",
    "TrainingSettings",
    "accuracy",
    "build_optimizer",
    "build_scheduler",
    "evaluation_mode",
    "load_checkpoint",
    "save_checkpoint",
    "topk_accuracy",
]
