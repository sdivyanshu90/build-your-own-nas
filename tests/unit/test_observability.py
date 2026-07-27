"""Unit tests for logging, events, context propagation, counters, and the error taxonomy.

Covers: secret redaction, event severity mapping, ambient identifier context, counter and
duration aggregation, snapshot merging, and the exception hierarchy's retry semantics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from nas_engine.exceptions import (
    ArchitectureValidationError,
    CheckpointVersionError,
    ConstraintViolationError,
    DuplicateRecordError,
    EvaluationTimeoutError,
    NasEngineError,
    NonFiniteLossError,
    PersistenceError,
    RecordNotFoundError,
    SchemaVersionError,
    ShapeInferenceError,
    TrainingError,
    WorkerError,
)
from nas_engine.observability.context import (
    bind_context,
    candidate_context,
    current_context,
    search_context,
    worker_context,
)
from nas_engine.observability.events import Event, emit
from nas_engine.observability.logging import (
    REDACTED,
    configure_logging,
    get_logger,
    redact_mapping,
)
from nas_engine.observability.metrics import CounterRegistry, DurationSummary, MetricsSnapshot

pytestmark = pytest.mark.unit


class TestExceptionTaxonomy:
    def test_every_error_derives_from_the_base(self) -> None:
        for error_type in (
            ArchitectureValidationError,
            TrainingError,
            PersistenceError,
            WorkerError,
        ):
            assert issubclass(error_type, NasEngineError)

    def test_codes_are_distinct_per_category(self) -> None:
        codes = {
            ArchitectureValidationError.code,
            TrainingError.code,
            PersistenceError.code,
            WorkerError.code,
        }
        assert len(codes) == 4

    @pytest.mark.parametrize(
        ("error_type", "retriable"),
        [
            (ArchitectureValidationError, False),
            (ShapeInferenceError, False),
            (ConstraintViolationError, False),
            (NonFiniteLossError, False),
            (SchemaVersionError, False),
            (RecordNotFoundError, False),
            (DuplicateRecordError, False),
            (CheckpointVersionError, False),
            (TrainingError, True),
            (EvaluationTimeoutError, True),
            (PersistenceError, True),
            (WorkerError, True),
        ],
    )
    def test_retry_semantics_are_declared(
        self, error_type: type[NasEngineError], retriable: bool
    ) -> None:
        assert error_type.retriable is retriable

    def test_details_are_rendered_in_the_message(self) -> None:
        error = TrainingError("failed", details={"epoch": 3, "step": 12})
        assert "epoch=3" in str(error)
        assert "step=12" in str(error)

    def test_details_render_in_sorted_order(self) -> None:
        error = TrainingError("failed", details={"z": 1, "a": 2})
        assert str(error).index("a=2") < str(error).index("z=1")

    def test_message_alone_when_no_details(self) -> None:
        assert str(TrainingError("failed")) == "failed"

    def test_serialises_to_plain_data(self) -> None:
        payload = NonFiniteLossError("diverged", details={"epoch": 1}).to_dict()
        assert payload["code"] == "non_finite_loss_error"
        assert payload["retriable"] is False
        assert payload["details"] == {"epoch": 1}

    def test_subclass_relationships_match_the_documented_taxonomy(self) -> None:
        assert issubclass(NonFiniteLossError, TrainingError)
        assert issubclass(ShapeInferenceError, ArchitectureValidationError)
        assert issubclass(ConstraintViolationError, ArchitectureValidationError)
        assert issubclass(SchemaVersionError, PersistenceError)


class TestRedaction:
    @pytest.mark.parametrize(
        "key",
        ["password", "api_key", "APIKEY", "auth_token", "db_secret", "private_key", "passwd"],
    )
    def test_sensitive_keys_are_redacted(self, key: str) -> None:
        assert redact_mapping({key: "hunter2"})[key] == REDACTED

    def test_ordinary_keys_survive(self) -> None:
        assert redact_mapping({"learning_rate": 0.1}) == {"learning_rate": 0.1}

    def test_nested_mappings_are_redacted(self) -> None:
        redacted = redact_mapping({"outer": {"api_key": "x", "safe": 1}})
        assert redacted["outer"]["api_key"] == REDACTED
        assert redacted["outer"]["safe"] == 1

    def test_sequences_are_traversed(self) -> None:
        redacted = redact_mapping({"items": [{"token": "x"}, {"ok": 1}]})
        assert redacted["items"][0]["token"] == REDACTED
        assert redacted["items"][1]["ok"] == 1

    def test_tuples_keep_their_type(self) -> None:
        redacted = redact_mapping({"items": ({"token": "x"},)})
        assert isinstance(redacted["items"], tuple)

    def test_recursion_is_bounded(self) -> None:
        deep: dict[str, Any] = {"password": "x"}
        for _ in range(20):
            deep = {"level": deep}
        # Must terminate rather than recursing without bound.
        assert redact_mapping(deep, max_depth=3)

    def test_the_original_mapping_is_not_modified(self) -> None:
        original = {"password": "hunter2"}
        redact_mapping(original)
        assert original == {"password": "hunter2"}


class TestLoggingConfiguration:
    def test_unknown_level_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown log level"):
            configure_logging(level="LOUD", force=True)

    def test_console_and_json_formats_are_accepted(self) -> None:
        configure_logging(level="INFO", log_format="json", force=True)
        configure_logging(level="INFO", log_format="console", force=True)

    def test_log_file_is_created(self, tmp_path: Path) -> None:
        target = tmp_path / "logs" / "run.log"
        configure_logging(level="INFO", log_file=target, force=True)
        get_logger("test").info("hello")
        logging.shutdown()
        assert target.exists()
        configure_logging(level="INFO", force=True)

    def test_logger_is_usable_without_explicit_configuration(self) -> None:
        assert get_logger(__name__) is not None


class TestEvents:
    def test_event_values_are_dotted_names(self) -> None:
        for event in Event:
            assert "." in event.value

    def test_event_values_are_unique(self) -> None:
        assert len({event.value for event in Event}) == len(list(Event))

    def test_event_renders_as_its_value(self) -> None:
        assert f"{Event.SEARCH_STARTED}" == "search.started"

    def test_emitting_every_event_is_safe(self) -> None:
        for event in Event:
            emit(event, search_id="s", detail=1)

    def test_failures_are_logged_at_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        # structlog writes through its own stderr handler, which `caplog` does not see;
        # capturing the stream is the only way to assert on the rendered severity.
        configure_logging(level="DEBUG", log_format="json", force=True)
        emit(Event.SEARCH_FAILED, search_id="s")
        captured = capsys.readouterr().err
        assert '"level": "error"' in captured
        assert "search.failed" in captured

    def test_rejections_are_logged_at_warning(self, capsys: pytest.CaptureFixture[str]) -> None:
        configure_logging(level="DEBUG", log_format="json", force=True)
        emit(Event.CANDIDATE_REJECTED, search_id="s")
        assert '"level": "warning"' in capsys.readouterr().err

    def test_routine_events_are_logged_at_info(self, capsys: pytest.CaptureFixture[str]) -> None:
        configure_logging(level="DEBUG", log_format="json", force=True)
        emit(Event.CANDIDATE_PROPOSED, search_id="s")
        assert '"level": "info"' in capsys.readouterr().err

    def test_ambient_context_is_attached_to_events(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        configure_logging(level="DEBUG", log_format="json", force=True)
        with search_context("search-123", strategy="random_search"):
            emit(Event.CANDIDATE_PROPOSED)
        captured = capsys.readouterr().err
        assert "search-123" in captured
        assert "random_search" in captured

    def test_sensitive_fields_are_redacted_in_events(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        configure_logging(level="DEBUG", log_format="json", force=True)
        emit(Event.SEARCH_STARTED, api_token="hunter2")
        captured = capsys.readouterr().err
        assert "hunter2" not in captured
        assert REDACTED in captured


class TestContext:
    def test_context_starts_empty(self) -> None:
        assert current_context() == {}

    def test_binding_is_scoped_to_the_block(self) -> None:
        with bind_context(search_id="s1"):
            assert current_context()["search_id"] == "s1"
        assert current_context() == {}

    def test_nested_bindings_merge(self) -> None:
        with search_context("s1", strategy="random"), candidate_context("c1"):
            context = current_context()
            assert context["search_id"] == "s1"
            assert context["strategy"] == "random"
            assert context["candidate_id"] == "c1"

    def test_none_values_are_dropped(self) -> None:
        with bind_context(search_id="s1", trial_id=None):
            assert "trial_id" not in current_context()

    def test_worker_context_stringifies_the_id(self) -> None:
        with worker_context(3):
            assert current_context()["worker_id"] == "3"

    def test_exceptions_still_restore_the_context(self) -> None:
        with pytest.raises(RuntimeError), bind_context(search_id="s1"):
            raise RuntimeError("boom")
        assert current_context() == {}


class TestCounters:
    def test_counters_accumulate(self) -> None:
        registry = CounterRegistry()
        assert registry.increment("a") == 1
        assert registry.increment("a", 4) == 5
        assert registry.counter("a") == 5

    def test_unseen_counters_read_as_zero(self) -> None:
        assert CounterRegistry().counter("nope") == 0

    def test_negative_increments_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            CounterRegistry().increment("a", -1)

    def test_gauges_keep_the_latest_value(self) -> None:
        registry = CounterRegistry()
        registry.set_gauge("g", 1.0)
        registry.set_gauge("g", 2.0)
        assert registry.snapshot().gauges["g"] == 2.0

    def test_durations_are_summarised(self) -> None:
        registry = CounterRegistry()
        for value in (1.0, 3.0, 2.0):
            registry.observe_duration("d", value)
        summary = registry.snapshot().durations["d"]
        assert summary.count == 3
        assert summary.min_seconds == 1.0
        assert summary.max_seconds == 3.0
        assert summary.mean_seconds == pytest.approx(2.0)

    def test_negative_durations_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            CounterRegistry().observe_duration("d", -1.0)

    def test_empty_duration_summary_has_zero_mean(self) -> None:
        assert DurationSummary(0, 0.0, 0.0, 0.0).mean_seconds == 0.0

    def test_reset_clears_everything(self) -> None:
        registry = CounterRegistry()
        registry.increment("a")
        registry.set_gauge("g", 1.0)
        registry.observe_duration("d", 1.0)
        registry.reset()
        snapshot = registry.snapshot()
        assert snapshot.counters == {}
        assert snapshot.gauges == {}
        assert snapshot.durations == {}

    def test_snapshots_merge_by_summing_counters(self) -> None:
        first = MetricsSnapshot(counters={"a": 1}, gauges={"g": 1.0})
        second = MetricsSnapshot(counters={"a": 2, "b": 1}, gauges={"g": 5.0})
        merged = first.merge(second)
        assert merged.counters == {"a": 3, "b": 1}
        assert merged.gauges == {"g": 5.0}

    def test_snapshots_merge_duration_statistics(self) -> None:
        first = MetricsSnapshot(durations={"d": DurationSummary(1, 1.0, 1.0, 1.0)})
        second = MetricsSnapshot(durations={"d": DurationSummary(1, 5.0, 5.0, 5.0)})
        merged = first.merge(second).durations["d"]
        assert merged.count == 2
        assert merged.min_seconds == 1.0
        assert merged.max_seconds == 5.0

    def test_snapshot_serialises(self) -> None:
        registry = CounterRegistry()
        registry.increment("a")
        registry.observe_duration("d", 1.0)
        payload = registry.snapshot().to_dict()
        assert payload["counters"]["a"] == 1
        assert payload["durations"]["d"]["count"] == 1.0

    def test_registry_is_thread_safe(self) -> None:
        import threading

        registry = CounterRegistry()

        def worker() -> None:
            for _ in range(200):
                registry.increment("a")

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert registry.counter("a") == 800
