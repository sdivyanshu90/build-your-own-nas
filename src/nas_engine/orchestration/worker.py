"""The worker-process entry point for multiprocessing execution.

Constraints imposed by ``spawn``
---------------------------------
The default start method is ``spawn`` (see
:class:`~nas_engine.config.models.ConcurrencyConfig`), which means a worker is a *fresh
interpreter*. Nothing is inherited: no imported modules, no open database handles, no
loaded dataset, no logging configuration. Everything the worker needs must arrive as a
picklable payload and be rebuilt on arrival.

``fork`` would inherit all of it, and that is precisely why it is not the default: forking
a process that has already initialised CUDA or a threaded BLAS library produces a child
with a broken runtime, and the resulting hangs are extremely hard to diagnose.

Consequences, all handled here
-------------------------------
* **The payload is plain data.** Configuration, architecture, and budget cross the process
  boundary as dictionaries, never as live objects. No custom class needs to be picklable.
* **Expensive setup is cached per process.** Building the dataset and evaluator costs real
  time, so the first task in a worker builds them and every later task reuses them. The
  cache is keyed by configuration hash, so a worker handed a different configuration
  rebuilds rather than silently using the wrong one.
* **Seeds are derived, never shared.** The worker seeds itself from
  ``(master_seed, worker_id)``, and each candidate seeds itself from its architecture
  hash. Two workers therefore never draw the same weights, and a candidate's weights do
  not depend on which worker happened to run it.
* **No exception escapes.** An exception crossing a process boundary loses its traceback
  and can fail to unpickle. Failures are classified and returned as data.
"""

from __future__ import annotations

import os
from typing import Any

from nas_engine.architectures.canonical import from_canonical_dict
from nas_engine.evaluation.budget import TrainingBudget
from nas_engine.evaluation.evaluator import CandidateEvaluator, EvaluationContext
from nas_engine.evaluation.result import EvaluationFailure, EvaluationResult
from nas_engine.observability.logging import configure_logging
from nas_engine.utilities.seeding import derive_seed, seed_everything
from nas_engine.utilities.timing import utc_now

#: Per-process cache of the built evaluator, keyed by configuration hash. Module-level
#: state is acceptable here precisely because each worker is a separate interpreter: there
#: is no cross-process sharing and no lock is needed.
_WORKER_CACHE: dict[str, CandidateEvaluator] = {}


def _build_evaluator(config_payload: dict[str, Any]) -> tuple[CandidateEvaluator, str]:
    """Build (or fetch from cache) the evaluator for a configuration.

    Args:
        config_payload: Serialised :class:`~nas_engine.config.models.SearchConfig`.

    Returns:
        The evaluator and the configuration hash it was built for.

    Raises:
        ConfigurationError: If the payload is not a valid configuration.
        DatasetError: If the dataset cannot be built.
    """
    from nas_engine.config.models import SearchConfig
    from nas_engine.datasets.registry import build_dataset
    from nas_engine.models.builder import ModelBuilder
    from nas_engine.utilities.determinism import configure_determinism

    config = SearchConfig.from_mapping(config_payload)
    key = config.config_hash()
    cached = _WORKER_CACHE.get(key)
    if cached is not None:
        return cached, key

    configure_logging(level=config.logging.level, log_format=config.logging.format, force=True)
    configure_determinism(
        enabled=config.reproducibility.deterministic,
        warn_only=config.reproducibility.warn_only,
    )

    dataset = build_dataset(config.dataset.provider, **config.dataset.options)
    device = config.hardware.resolve_device()
    evaluator = CandidateEvaluator(
        dataset=dataset,
        loader_settings=config.dataset.build_loader_settings(),
        training_settings=config.training.build(epochs=config.budget.epochs),
        settings=config.evaluation.build(max_seconds=config.budget.max_seconds_per_evaluation),
        artifact_root=config.artifact_dir,
        device=device,
        seed=config.reproducibility.seed,
        model_builder=ModelBuilder(zero_init_residual=config.training.zero_init_residual),
    )
    _WORKER_CACHE[key] = evaluator
    return evaluator, key


def _safe_budget(raw: object) -> TrainingBudget:
    """Rebuild a budget for a failure report, never raising.

    The failure path must not itself fail. A malformed budget is one of the things that
    can cause the evaluation to fail in the first place, so parsing it again while
    reporting that failure would let an exception escape the worker — which is exactly the
    outcome this module exists to prevent.

    Args:
        raw: The serialised budget from the payload, possibly malformed or absent.

    Returns:
        The parsed budget, or a one-epoch placeholder.
    """
    if isinstance(raw, dict):
        try:
            return TrainingBudget.from_dict(raw)
        except Exception:  # noqa: S110 - see the docstring: this path cannot raise
            pass
    return TrainingBudget(epochs=1)


def evaluate_task(payload: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one candidate inside a worker process.

    This function is the target handed to
    :class:`concurrent.futures.ProcessPoolExecutor`, so it must be importable at module
    level and must accept and return only picklable plain data.

    Args:
        payload: Mapping with keys ``config``, ``spec``, ``budget``, ``candidate_id``,
            ``trial_id``, ``attempt``, and ``worker_id``.

    Returns:
        The serialised :class:`~nas_engine.evaluation.result.EvaluationResult`.
    """
    candidate_id = str(payload.get("candidate_id", "unknown"))
    trial_id = str(payload.get("trial_id", "unknown"))
    worker_id = str(payload.get("worker_id", os.getpid()))
    started_at = utc_now()

    try:
        evaluator, _ = _build_evaluator(payload["config"])
        spec = from_canonical_dict(payload["spec"])
        budget = TrainingBudget.from_dict(payload["budget"])

        # Seed the worker's global RNGs from a worker-specific stream. Per-candidate seeds
        # are derived separately inside the evaluator, so this only affects anything that
        # reads a global generator outside the evaluator's control.
        master_seed = int(payload.get("seed", 42))
        seed_everything(derive_seed(master_seed, f"worker:{worker_id}"))

        result = evaluator.evaluate(
            spec,
            budget,
            EvaluationContext(
                candidate_id=candidate_id,
                trial_id=trial_id,
                worker_id=worker_id,
                attempt=int(payload.get("attempt", 0)),
            ),
        )
        return result.to_dict()
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as error:
        import traceback

        failure = EvaluationFailure.from_exception(
            error, traceback_text=traceback.format_exc(limit=20)
        )
        fallback_budget = _safe_budget(payload.get("budget"))
        return EvaluationResult(
            candidate_id=candidate_id,
            architecture_hash=str(payload.get("architecture_hash", "unknown")),
            budget=fallback_budget,
            succeeded=False,
            failure=failure,
            started_at=started_at,
            completed_at=utc_now(),
            worker_id=worker_id,
        ).to_dict()


def reset_worker_cache() -> None:
    """Clear the per-process evaluator cache.

    Used by tests that build several evaluators in one interpreter and must not have one
    leak into the next.
    """
    _WORKER_CACHE.clear()


__all__ = ["evaluate_task", "reset_worker_cache"]
