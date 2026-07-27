"""Unit tests for orchestration primitives.

Covers: the candidate state machine and its transition table, the retry policy's decision
logic, search-checkpoint versioning and validation, executor task payloads, and the search
result object.
"""

from __future__ import annotations

import pytest

from nas_engine.architectures.spec import ArchitectureSpec
from nas_engine.evaluation.budget import TrainingBudget
from nas_engine.evaluation.result import EvaluationFailure, FailureKind
from nas_engine.exceptions import (
    CheckpointError,
    CheckpointVersionError,
    EvaluationTimeoutError,
    InvalidStateTransitionError,
    NonFiniteLossError,
    ResourceLimitError,
    TrainingError,
    WorkerError,
)
from nas_engine.objectives.ranking import RankedCandidate
from nas_engine.orchestration.checkpoint import (
    SEARCH_CHECKPOINT_FORMAT_VERSION,
    EngineState,
    SearchCheckpoint,
)
from nas_engine.orchestration.executors import EvaluationTask, SequentialExecutor
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
from nas_engine.orchestration.retry import RetryPolicy

pytestmark = pytest.mark.unit


class TestStateMachine:
    def test_every_state_appears_in_the_transition_table(self) -> None:
        assert set(ALLOWED_TRANSITIONS) == set(CandidateState)

    def test_terminal_states_have_no_outgoing_edges(self) -> None:
        for state in TERMINAL_STATES:
            assert ALLOWED_TRANSITIONS[state] == frozenset()
            assert state.is_terminal

    def test_the_happy_path_is_permitted(self) -> None:
        machine = CandidateStateMachine()
        machine.transition(CandidateState.VALIDATED)
        machine.transition(CandidateState.QUEUED)
        machine.transition(CandidateState.RUNNING)
        machine.transition(CandidateState.COMPLETED)
        assert machine.is_terminal

    def test_running_can_return_to_the_queue_for_a_retry(self) -> None:
        assert can_transition(CandidateState.RUNNING, CandidateState.QUEUED)

    def test_skipping_validation_is_forbidden(self) -> None:
        with pytest.raises(InvalidStateTransitionError, match="legal transitions"):
            validate_transition(CandidateState.PROPOSED, CandidateState.RUNNING)

    def test_leaving_a_terminal_state_is_forbidden(self) -> None:
        with pytest.raises(InvalidStateTransitionError, match="terminal state"):
            validate_transition(CandidateState.COMPLETED, CandidateState.QUEUED)

    def test_error_lists_the_legal_targets(self) -> None:
        with pytest.raises(InvalidStateTransitionError) as excinfo:
            validate_transition(CandidateState.VALIDATED, CandidateState.COMPLETED)
        assert "queued" in excinfo.value.details["allowed"]

    def test_active_states_are_identified(self) -> None:
        assert CandidateState.QUEUED.is_active
        assert CandidateState.RUNNING.is_active
        assert not CandidateState.COMPLETED.is_active

    def test_cancellation_is_reachable_from_every_live_state(self) -> None:
        for state in CandidateState:
            if state.is_terminal:
                continue
            assert can_transition(state, CandidateState.CANCELLED)

    def test_try_transition_reports_instead_of_raising(self) -> None:
        machine = CandidateStateMachine(state=CandidateState.COMPLETED)
        assert not machine.try_transition(CandidateState.QUEUED)
        assert machine.state is CandidateState.COMPLETED

    def test_history_records_reasons(self) -> None:
        machine = CandidateStateMachine()
        machine.transition(CandidateState.VALIDATED, reason="checks passed")
        assert "checks passed" in machine.describe_history()
        assert len(machine.history) == 2

    def test_trial_states_are_distinct(self) -> None:
        assert len({state.value for state in TrialState}) == len(list(TrialState))


class TestRetryPolicy:
    @staticmethod
    def _failure(error: BaseException) -> EvaluationFailure:
        return EvaluationFailure.from_exception(error)

    def test_permanent_failures_are_never_retried(self) -> None:
        policy = RetryPolicy(max_retries=5)
        decision = policy.decide(self._failure(NonFiniteLossError("diverged")), attempt=0)
        assert not decision.should_retry
        assert "permanent" in decision.reason

    def test_retriable_failures_are_retried_within_budget(self) -> None:
        policy = RetryPolicy(max_retries=2)
        decision = policy.decide(self._failure(TrainingError("transient")), attempt=0)
        assert decision.should_retry
        assert decision.attempts_remaining == 2

    def test_retries_stop_once_exhausted(self) -> None:
        policy = RetryPolicy(max_retries=1)
        decision = policy.decide(self._failure(TrainingError("transient")), attempt=1)
        assert not decision.should_retry
        assert "exhausted" in decision.reason

    def test_zero_retries_disables_retrying(self) -> None:
        policy = RetryPolicy(max_retries=0)
        assert not policy.decide(self._failure(WorkerError("died")), attempt=0).should_retry

    def test_timeout_retries_can_be_disabled(self) -> None:
        policy = RetryPolicy(max_retries=3, retry_on_timeout=False)
        decision = policy.decide(self._failure(EvaluationTimeoutError("slow")), attempt=0)
        assert not decision.should_retry
        assert "retry_on_timeout" in decision.reason

    def test_resource_retries_can_be_disabled(self) -> None:
        policy = RetryPolicy(max_retries=3, retry_on_resource_error=False)
        failure = EvaluationFailure(
            kind=FailureKind.RESOURCE,
            code="oom",
            message="out of memory",
            retriable=True,
            exception_type="MemoryError",
        )
        assert not policy.decide(failure, attempt=0).should_retry

    def test_resource_limit_errors_are_permanent(self) -> None:
        policy = RetryPolicy(max_retries=3)
        decision = policy.decide(self._failure(ResourceLimitError("too big")), attempt=0)
        assert not decision.should_retry

    def test_backoff_is_disabled_by_default(self) -> None:
        assert RetryPolicy().backoff_for(3) == 0.0

    def test_backoff_grows_exponentially_and_is_capped(self) -> None:
        policy = RetryPolicy(backoff_seconds=1.0, backoff_multiplier=2.0, max_backoff_seconds=5.0)
        assert policy.backoff_for(0) == 1.0
        assert policy.backoff_for(1) == 2.0
        assert policy.backoff_for(2) == 4.0
        assert policy.backoff_for(10) == 5.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_retries": -1},
            {"backoff_seconds": -1.0},
            {"backoff_multiplier": 0.5},
            {"max_backoff_seconds": -1.0},
        ],
    )
    def test_configuration_is_validated(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ValueError):
            RetryPolicy(**kwargs)  # type: ignore[arg-type]


class TestSearchCheckpoint:
    @staticmethod
    def _checkpoint() -> SearchCheckpoint:
        return SearchCheckpoint(
            search_id="s1",
            strategy_name="random_search",
            strategy_state={"version": 1, "proposed": 3},
            engine_state=EngineState(proposed=3, completed=2),
            config_hash="abc",
        )

    def test_round_trips_through_a_payload(self) -> None:
        original = self._checkpoint()
        restored = SearchCheckpoint.from_payload(original.to_payload())
        assert restored.search_id == "s1"
        assert restored.strategy_state == {"version": 1, "proposed": 3}
        assert restored.engine_state.completed == 2

    def test_engine_state_round_trips(self) -> None:
        state = EngineState(proposed=5, duplicates=1, failed=2, elapsed_seconds=3.5)
        assert EngineState.from_dict(state.to_dict()) == state

    def test_rejects_a_future_format(self) -> None:
        payload = self._checkpoint().to_payload()
        payload["format_version"] = SEARCH_CHECKPOINT_FORMAT_VERSION + 1
        with pytest.raises(CheckpointVersionError, match="newer than the supported"):
            SearchCheckpoint.from_payload(payload)

    def test_rejects_a_missing_version(self) -> None:
        payload = self._checkpoint().to_payload()
        del payload["format_version"]
        with pytest.raises(CheckpointError, match="format_version"):
            SearchCheckpoint.from_payload(payload)

    def test_rejects_missing_fields(self) -> None:
        payload = self._checkpoint().to_payload()
        del payload["strategy_state"]
        with pytest.raises(CheckpointError, match="missing required fields"):
            SearchCheckpoint.from_payload(payload)

    def test_rejects_a_non_mapping_payload(self) -> None:
        with pytest.raises(CheckpointError, match="not a mapping"):
            SearchCheckpoint.from_payload([1, 2, 3])  # type: ignore[arg-type]

    def test_rejects_a_non_mapping_strategy_state(self) -> None:
        payload = self._checkpoint().to_payload()
        payload["strategy_state"] = "not a mapping"
        with pytest.raises(CheckpointError, match="must be a mapping"):
            SearchCheckpoint.from_payload(payload)

    def test_strategy_mismatch_is_fatal(self) -> None:
        with pytest.raises(CheckpointError, match="not interchangeable"):
            self._checkpoint().validate_for(strategy_name="regularized_evolution")

    def test_configuration_change_is_only_a_warning(self) -> None:
        warnings = self._checkpoint().validate_for(
            strategy_name="random_search", config_hash="different"
        )
        assert len(warnings) == 1
        assert "configuration changed" in warnings[0]

    def test_matching_checkpoint_produces_no_warnings(self) -> None:
        assert (
            self._checkpoint().validate_for(strategy_name="random_search", config_hash="abc") == []
        )


class TestExecutorTasks:
    def test_payload_is_plain_json_compatible_data(self, sample_spec: ArchitectureSpec) -> None:
        task = EvaluationTask(
            candidate_id="c",
            trial_id="t",
            architecture_hash="h",
            spec=sample_spec,
            budget=TrainingBudget(epochs=2),
        )
        payload = task.to_payload({"project": {"name": "x"}}, seed=7)
        assert payload["seed"] == 7
        assert payload["candidate_id"] == "c"
        assert isinstance(payload["spec"], dict)
        assert isinstance(payload["budget"], dict)

    def test_sequential_executor_reports_its_mode(self) -> None:
        class _Stub:
            def evaluate(self, *args: object, **kwargs: object) -> object:
                raise AssertionError("not called")

        executor = SequentialExecutor(_Stub())  # type: ignore[arg-type]
        assert executor.mode == "sequential"
        assert executor.max_in_flight == 1
        assert executor.run_batch([]) == []
        executor.shutdown()


class TestSearchResult:
    @staticmethod
    def _candidate() -> RankedCandidate:
        return RankedCandidate(
            candidate_id="c",
            architecture_hash="0123456789abcdef",
            metrics={"validation_accuracy": 0.75, "trainable_parameters": 1234.0},
            rank=0,
            score=0.9,
            pareto_rank=0,
            crowding=float("inf"),
            feasible=True,
        )

    def test_summary_reports_the_best_candidate(self) -> None:
        result = SearchResult(
            search_id="s",
            status="completed",
            stop_reason=StopReason.BUDGET_EXHAUSTED,
            best=self._candidate(),
            pareto_front=(self._candidate(),),
            ranked=(self._candidate(),),
            engine_state=EngineState(completed=1),
        )
        text = result.summary()
        assert "0.7500" in text
        assert "1,234" in text
        assert result.succeeded
        assert result.best_accuracy == 0.75

    def test_summary_handles_an_empty_search(self) -> None:
        result = SearchResult(
            search_id="s",
            status="completed",
            stop_reason=StopReason.SPACE_EXHAUSTED,
            best=None,
            pareto_front=(),
            ranked=(),
            engine_state=EngineState(),
            warnings=("nothing completed",),
        )
        text = result.summary()
        assert "none" in text
        assert "nothing completed" in text
        assert not result.succeeded
        assert result.best_accuracy is None

    def test_serialises_to_plain_data(self) -> None:
        payload = SearchResult(
            search_id="s",
            status="completed",
            stop_reason=StopReason.TIME_LIMIT,
            best=self._candidate(),
            pareto_front=(),
            ranked=(self._candidate(),),
            engine_state=EngineState(),
        ).to_dict()
        assert payload["stop_reason"] == "time_limit"
        assert payload["stop_reason_description"]
        assert payload["best"]["candidate_id"] == "c"

    @pytest.mark.parametrize("reason", list(StopReason))
    def test_every_stop_reason_is_described(self, reason: StopReason) -> None:
        assert reason.describe()

    def test_crowding_infinity_is_serialised_as_null(self) -> None:
        assert self._candidate().to_dict()["crowding"] is None
