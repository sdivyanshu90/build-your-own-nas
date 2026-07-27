"""Orchestration: the candidate lifecycle, execution backends, retries, and the engine.

This is the top of the dependency graph. It imports from every other package and nothing
imports from it, except :mod:`nas_engine.persistence`, which uses only
:mod:`nas_engine.orchestration.lifecycle` — a leaf module with no back-dependencies, so
the graph stays acyclic.
"""

from nas_engine.orchestration.checkpoint import EngineState, SearchCheckpoint
from nas_engine.orchestration.engine import SearchEngine
from nas_engine.orchestration.executors import (
    EvaluationExecutor,
    EvaluationTask,
    ProcessPoolExecutorBackend,
    SequentialExecutor,
    build_executor,
)
from nas_engine.orchestration.lifecycle import (
    ALLOWED_TRANSITIONS,
    TERMINAL_STATES,
    CandidateState,
    CandidateStateMachine,
    TrialState,
    can_transition,
    validate_transition,
)
from nas_engine.orchestration.result import SearchResult, StopReason
from nas_engine.orchestration.retry import RetryDecision, RetryPolicy

__all__ = [
    "ALLOWED_TRANSITIONS",
    "TERMINAL_STATES",
    "CandidateState",
    "CandidateStateMachine",
    "EngineState",
    "EvaluationExecutor",
    "EvaluationTask",
    "ProcessPoolExecutorBackend",
    "RetryDecision",
    "RetryPolicy",
    "SearchCheckpoint",
    "SearchEngine",
    "SearchResult",
    "SequentialExecutor",
    "StopReason",
    "TrialState",
    "build_executor",
    "can_transition",
    "validate_transition",
]
