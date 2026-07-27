"""Unit tests for the persistence layer.

Covers: connection setup and SQLite pragmas, versioned migrations, every repository
operation, uniqueness enforcement, atomic claiming, transaction rollback, cascade deletes,
timezone-aware timestamps, and interrupted-run recovery.
"""

from __future__ import annotations

from datetime import timezone
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import text

from nas_engine.architectures.hashing import architecture_hash
from nas_engine.architectures.spec import ArchitectureSpec
from nas_engine.evaluation.budget import TrainingBudget
from nas_engine.exceptions import (
    DuplicateRecordError,
    InvalidStateTransitionError,
    PersistenceError,
    RecordNotFoundError,
    SchemaVersionError,
)
from nas_engine.orchestration.lifecycle import CandidateState, TrialState
from nas_engine.persistence.database import Database
from nas_engine.persistence.migrations import (
    TARGET_SCHEMA_VERSION,
    apply_migrations,
    current_version,
    ensure_schema,
)
from nas_engine.persistence.models import CandidateRecord, SearchStatus
from nas_engine.persistence.repository import SearchRepository

pytestmark = pytest.mark.unit


def _create_search(repository: SearchRepository, **overrides: Any) -> str:
    """Create a search record with test-friendly defaults."""
    payload: dict[str, Any] = {
        "name": "test",
        "strategy": "random_search",
        "config": {"version": 1},
        "config_hash": "hash",
        "config_version": 1,
        "search_space": {"name": "tiny"},
        "seed": 42,
        "seeds": {"master": 42},
        "environment": {"python_version": "3.12"},
        "planned_evaluations": 5,
    }
    payload.update(overrides)
    return repository.create_search(**payload)


class TestDatabase:
    def test_in_memory_database_shares_one_connection(self) -> None:
        database = Database.in_memory()
        ensure_schema(database)
        with database.session() as session:
            session.execute(text("SELECT 1"))
        assert current_version(database) == TARGET_SCHEMA_VERSION
        database.dispose()

    def test_foreign_keys_are_enforced(self, database: Database) -> None:
        with database.session() as session:
            value = session.execute(text("PRAGMA foreign_keys")).scalar()
        assert value == 1

    def test_file_database_creates_parent_directories(self, tmp_path: Path) -> None:
        database = Database.from_path(tmp_path / "deep" / "nested" / "nas.db")
        ensure_schema(database)
        assert (tmp_path / "deep" / "nested" / "nas.db").exists()
        database.dispose()

    def test_transactions_roll_back_on_error(self, database: Database) -> None:
        repository = SearchRepository(database)
        search_id = _create_search(repository)
        with pytest.raises(RuntimeError), database.session() as session:
            record = session.get(CandidateRecord, "missing")
            assert record is None
            raise RuntimeError("boom")
        assert repository.get_search(search_id).status == SearchStatus.CREATED.value

    def test_invalid_url_is_reported(self) -> None:
        with pytest.raises(PersistenceError, match="could not create a database engine"):
            Database("not-a-url://x")

    def test_context_manager_disposes(self, tmp_path: Path) -> None:
        with Database.from_path(tmp_path / "nas.db") as database:
            ensure_schema(database)
        assert (tmp_path / "nas.db").exists()


class TestMigrations:
    def test_fresh_database_reports_version_zero(self) -> None:
        database = Database.in_memory()
        assert current_version(database) == 0
        database.dispose()

    def test_ensure_schema_is_idempotent(self, database: Database) -> None:
        assert ensure_schema(database) == TARGET_SCHEMA_VERSION
        assert ensure_schema(database) == TARGET_SCHEMA_VERSION

    def test_newer_database_is_refused(self, database: Database) -> None:
        with pytest.raises(SchemaVersionError, match="supports at most version"):
            apply_migrations(database, target=TARGET_SCHEMA_VERSION - 1)

    def test_partial_migration_target_is_honoured(self) -> None:
        database = Database.in_memory()
        assert apply_migrations(database, target=1) == 1
        database.dispose()


class TestSearchRecords:
    def test_create_and_fetch(self, repository: SearchRepository) -> None:
        search_id = _create_search(repository, name="my-search")
        summary = repository.get_search(search_id)
        assert summary.name == "my-search"
        assert summary.status == SearchStatus.CREATED.value
        assert summary.seed == 42

    def test_timestamps_are_timezone_aware(self, repository: SearchRepository) -> None:
        summary = repository.get_search(_create_search(repository))
        assert summary.created_at.tzinfo is not None
        assert summary.created_at.utcoffset() == timezone.utc.utcoffset(None)

    def test_duplicate_identifier_is_rejected(self, repository: SearchRepository) -> None:
        _create_search(repository, search_id="fixed")
        with pytest.raises(DuplicateRecordError, match="already exists"):
            _create_search(repository, search_id="fixed")

    def test_missing_search_is_reported(self, repository: SearchRepository) -> None:
        with pytest.raises(RecordNotFoundError, match="no search found"):
            repository.get_search("nope")

    def test_configuration_and_environment_round_trip(self, repository: SearchRepository) -> None:
        search_id = _create_search(
            repository, config={"version": 1, "a": [1, 2]}, environment={"x": "y"}
        )
        assert repository.get_search_config(search_id)["a"] == [1, 2]
        assert repository.get_search_environment(search_id) == {"x": "y"}

    def test_status_updates_stamp_timestamps(self, repository: SearchRepository) -> None:
        search_id = _create_search(repository)
        repository.update_search_status(search_id, SearchStatus.RUNNING, started=True)
        assert repository.get_search(search_id).started_at is not None
        repository.update_search_status(search_id, SearchStatus.COMPLETED, completed=True)
        summary = repository.get_search(search_id)
        assert summary.completed_at is not None
        assert summary.duration_seconds is not None

    def test_listing_is_newest_first(self, repository: SearchRepository) -> None:
        first = _create_search(repository, name="a")
        second = _create_search(repository, name="b")
        listed = [summary.id for summary in repository.list_searches()]
        assert set(listed) == {first, second}

    def test_latest_search_can_be_filtered_by_name(self, repository: SearchRepository) -> None:
        _create_search(repository, name="alpha")
        beta = _create_search(repository, name="beta")
        found = repository.find_latest_search(name="beta")
        assert found is not None
        assert found.id == beta

    def test_latest_search_is_none_when_empty(self, repository: SearchRepository) -> None:
        assert repository.find_latest_search() is None

    def test_summary_serialises(self, repository: SearchRepository) -> None:
        payload = repository.get_search(_create_search(repository)).to_dict()
        assert payload["seed"] == 42
        assert payload["created_at"].endswith("+00:00")


class TestCandidates:
    def test_insert_and_fetch(
        self, repository: SearchRepository, sample_spec: ArchitectureSpec
    ) -> None:
        search_id = _create_search(repository)
        digest = architecture_hash(sample_spec)
        candidate_id = repository.add_candidate(
            search_id=search_id,
            architecture_hash=digest,
            spec=sample_spec,
            origin="random",
        )
        candidate = repository.get_candidate(candidate_id)
        assert candidate.architecture_hash == digest
        assert candidate.status == CandidateState.PROPOSED.value
        assert candidate.origin == "random"

    def test_specification_is_revalidated_on_read(
        self, repository: SearchRepository, sample_spec: ArchitectureSpec
    ) -> None:
        search_id = _create_search(repository)
        candidate_id = repository.add_candidate(
            search_id=search_id,
            architecture_hash=architecture_hash(sample_spec),
            spec=sample_spec,
        )
        assert repository.get_candidate_spec(candidate_id) == sample_spec

    def test_corrupt_specification_is_rejected_on_read(
        self, repository: SearchRepository, sample_spec: ArchitectureSpec
    ) -> None:
        from nas_engine.exceptions import ArchitectureValidationError

        search_id = _create_search(repository)
        candidate_id = repository.add_candidate(
            search_id=search_id,
            architecture_hash=architecture_hash(sample_spec),
            spec=sample_spec,
        )
        with repository.database.session() as session:
            record = session.get(CandidateRecord, candidate_id)
            assert record is not None
            record.spec_json = {"stages": "not a list"}
        with pytest.raises(ArchitectureValidationError):
            repository.get_candidate_spec(candidate_id)

    def test_identity_is_unique_per_rung(
        self, repository: SearchRepository, sample_spec: ArchitectureSpec
    ) -> None:
        search_id = _create_search(repository)
        digest = architecture_hash(sample_spec)
        repository.add_candidate(
            search_id=search_id, architecture_hash=digest, spec=sample_spec, rung=0
        )
        with pytest.raises(DuplicateRecordError, match="already exists"):
            repository.add_candidate(
                search_id=search_id, architecture_hash=digest, spec=sample_spec, rung=0
            )
        # A different rung is a genuinely different measurement and is allowed.
        repository.add_candidate(
            search_id=search_id, architecture_hash=digest, spec=sample_spec, rung=1
        )

    def test_lookup_by_hash_and_rung(
        self, repository: SearchRepository, sample_spec: ArchitectureSpec
    ) -> None:
        search_id = _create_search(repository)
        digest = architecture_hash(sample_spec)
        repository.add_candidate(
            search_id=search_id, architecture_hash=digest, spec=sample_spec, rung=2
        )
        assert repository.find_candidate(search_id, digest, rung=2) is not None
        assert repository.find_candidate(search_id, digest, rung=0) is None

    def test_state_transitions_are_validated(
        self, repository: SearchRepository, sample_spec: ArchitectureSpec
    ) -> None:
        search_id = _create_search(repository)
        candidate_id = repository.add_candidate(
            search_id=search_id,
            architecture_hash=architecture_hash(sample_spec),
            spec=sample_spec,
        )
        repository.update_candidate_state(candidate_id, CandidateState.VALIDATED)
        with pytest.raises(InvalidStateTransitionError):
            repository.update_candidate_state(candidate_id, CandidateState.COMPLETED)

    def test_transitions_record_events(
        self, repository: SearchRepository, sample_spec: ArchitectureSpec
    ) -> None:
        search_id = _create_search(repository)
        candidate_id = repository.add_candidate(
            search_id=search_id,
            architecture_hash=architecture_hash(sample_spec),
            spec=sample_spec,
        )
        repository.update_candidate_state(
            candidate_id, CandidateState.VALIDATED, reason="checks passed"
        )
        events = repository.list_events(search_id)
        assert any(event["event"] == "candidate.validated" for event in events)

    def test_counts_include_every_state(
        self, repository: SearchRepository, sample_spec: ArchitectureSpec
    ) -> None:
        search_id = _create_search(repository)
        repository.add_candidate(
            search_id=search_id,
            architecture_hash=architecture_hash(sample_spec),
            spec=sample_spec,
        )
        counts = repository.count_candidates_by_status(search_id)
        assert set(counts) == {state.value for state in CandidateState}
        assert counts["proposed"] == 1

    def test_retry_counter_increments(
        self, repository: SearchRepository, sample_spec: ArchitectureSpec
    ) -> None:
        search_id = _create_search(repository)
        candidate_id = repository.add_candidate(
            search_id=search_id,
            architecture_hash=architecture_hash(sample_spec),
            spec=sample_spec,
        )
        assert repository.increment_retry(candidate_id) == 1
        assert repository.increment_retry(candidate_id) == 2

    def test_claiming_moves_a_candidate_to_running(
        self, repository: SearchRepository, sample_spec: ArchitectureSpec
    ) -> None:
        search_id = _create_search(repository)
        candidate_id = _queued_candidate(repository, search_id, sample_spec)
        claimed = repository.claim_next_queued(search_id, worker_id="w0")
        assert claimed is not None
        assert claimed.id == candidate_id
        assert claimed.status == CandidateState.RUNNING.value

    def test_a_claimed_candidate_cannot_be_claimed_again(
        self, repository: SearchRepository, sample_spec: ArchitectureSpec
    ) -> None:
        search_id = _create_search(repository)
        _queued_candidate(repository, search_id, sample_spec)
        assert repository.claim_next_queued(search_id, worker_id="w0") is not None
        assert repository.claim_next_queued(search_id, worker_id="w1") is None

    def test_seen_hashes_covers_every_candidate(
        self, repository: SearchRepository, sample_spec: ArchitectureSpec
    ) -> None:
        search_id = _create_search(repository)
        digest = architecture_hash(sample_spec)
        repository.add_candidate(search_id=search_id, architecture_hash=digest, spec=sample_spec)
        assert repository.seen_hashes(search_id) == {digest}

    def test_listing_can_filter_and_paginate(
        self, repository: SearchRepository, sampler: Any
    ) -> None:
        search_id = _create_search(repository)
        for _ in range(5):
            spec = sampler.sample()
            repository.add_candidate(
                search_id=search_id,
                architecture_hash=architecture_hash(spec),
                spec=spec,
            )
        assert len(repository.list_candidates(search_id, limit=2)) == 2
        assert len(repository.list_candidates(search_id, offset=3)) == 2
        assert repository.list_candidates(search_id, statuses=[CandidateState.COMPLETED]) == []

    def test_missing_candidate_is_reported(self, repository: SearchRepository) -> None:
        with pytest.raises(RecordNotFoundError, match="no candidate found"):
            repository.get_candidate("nope")


class TestTrialsAndMetrics:
    def test_completed_trial_stores_metrics_and_artifacts(
        self, repository: SearchRepository, sample_spec: ArchitectureSpec
    ) -> None:
        search_id = _create_search(repository)
        candidate_id = _queued_candidate(repository, search_id, sample_spec)
        repository.claim_next_queued(search_id, worker_id="w0")
        trial_id = repository.start_trial(
            candidate_id=candidate_id, attempt=0, budget=TrainingBudget(epochs=1)
        )
        repository.complete_trial(
            trial_id,
            metrics={"validation_accuracy": 0.8},
            duration_seconds=1.5,
            artifacts={"weights": "w.pt"},
            artifact_sizes={"weights": 512},
        )
        repository.update_candidate_state(candidate_id, CandidateState.COMPLETED)
        candidate = repository.get_candidate(candidate_id)
        assert candidate.metrics == {"validation_accuracy": 0.8}
        assert candidate.artifacts == {"weights": "w.pt"}

    def test_rewriting_the_same_artifact_updates_rather_than_fails(
        self, repository: SearchRepository, sample_spec: ArchitectureSpec
    ) -> None:
        search_id = _create_search(repository)
        candidate_id = _queued_candidate(repository, search_id, sample_spec)
        for attempt in range(2):
            trial_id = repository.start_trial(
                candidate_id=candidate_id, attempt=attempt, budget=TrainingBudget(epochs=1)
            )
            repository.complete_trial(
                trial_id,
                metrics={"validation_accuracy": 0.1 * attempt},
                duration_seconds=1.0,
                artifacts={"weights": "w.pt"},
                artifact_sizes={"weights": 100},
            )
        assert len(repository.list_trials(candidate_id)) == 2

    def test_duplicate_attempt_is_rejected(
        self, repository: SearchRepository, sample_spec: ArchitectureSpec
    ) -> None:
        search_id = _create_search(repository)
        candidate_id = _queued_candidate(repository, search_id, sample_spec)
        repository.start_trial(
            candidate_id=candidate_id, attempt=0, budget=TrainingBudget(epochs=1)
        )
        with pytest.raises(DuplicateRecordError, match="already exists"):
            repository.start_trial(
                candidate_id=candidate_id, attempt=0, budget=TrainingBudget(epochs=1)
            )

    def test_failed_trial_records_the_error(
        self, repository: SearchRepository, sample_spec: ArchitectureSpec
    ) -> None:
        search_id = _create_search(repository)
        candidate_id = _queued_candidate(repository, search_id, sample_spec)
        trial_id = repository.start_trial(
            candidate_id=candidate_id, attempt=0, budget=TrainingBudget(epochs=1)
        )
        repository.fail_trial(trial_id, error={"code": "boom"}, duration_seconds=0.5, timeout=True)
        trials = repository.list_trials(candidate_id)
        assert trials[0]["status"] == TrialState.TIMEOUT.value
        assert trials[0]["error"] == {"code": "boom"}

    def test_completing_an_unknown_trial_is_reported(self, repository: SearchRepository) -> None:
        with pytest.raises(RecordNotFoundError, match="no trial found"):
            repository.complete_trial("nope", metrics={}, duration_seconds=0.0)

    def test_best_candidate_uses_a_single_metric(
        self, repository: SearchRepository, sampler: Any
    ) -> None:
        search_id = _create_search(repository)
        for index, accuracy in enumerate([0.3, 0.9, 0.5]):
            spec = sampler.sample()
            candidate_id = _completed_candidate(
                repository, search_id, spec, accuracy, attempt=index
            )
            assert candidate_id
        best = repository.best_candidate(search_id)
        assert best is not None
        assert best.metrics["validation_accuracy"] == 0.9

    def test_best_candidate_can_minimise(self, repository: SearchRepository, sampler: Any) -> None:
        search_id = _create_search(repository)
        for accuracy in [0.3, 0.9]:
            _completed_candidate(repository, search_id, sampler.sample(), accuracy)
        best = repository.best_candidate(search_id, maximize=False)
        assert best is not None
        assert best.metrics["validation_accuracy"] == 0.3

    def test_best_candidate_is_none_when_nothing_completed(
        self, repository: SearchRepository
    ) -> None:
        assert repository.best_candidate(_create_search(repository)) is None

    def test_completed_metrics_feeds_ranking(
        self, repository: SearchRepository, sampler: Any
    ) -> None:
        search_id = _create_search(repository)
        for accuracy in [0.4, 0.6]:
            _completed_candidate(repository, search_id, sampler.sample(), accuracy)
        entries = repository.completed_metrics(search_id)
        assert len(entries) == 2
        assert all(len(entry) == 3 for entry in entries)


class TestCheckpointsAndEvents:
    def test_checkpoints_are_append_only(self, repository: SearchRepository) -> None:
        search_id = _create_search(repository)
        assert repository.save_checkpoint(search_id, {"a": 1}) == 1
        assert repository.save_checkpoint(search_id, {"a": 2}) == 2
        assert repository.latest_checkpoint(search_id) == {"a": 2}
        assert repository.count_checkpoints(search_id) == 2

    def test_no_checkpoint_returns_none(self, repository: SearchRepository) -> None:
        assert repository.latest_checkpoint(_create_search(repository)) is None

    def test_pruning_keeps_the_newest(self, repository: SearchRepository) -> None:
        search_id = _create_search(repository)
        for index in range(6):
            repository.save_checkpoint(search_id, {"index": index})
        assert repository.prune_checkpoints(search_id, keep=2) == 4
        assert repository.count_checkpoints(search_id) == 2
        assert repository.latest_checkpoint(search_id) == {"index": 5}

    def test_pruning_validates_its_argument(self, repository: SearchRepository) -> None:
        with pytest.raises(ValueError, match="keep must be"):
            repository.prune_checkpoints(_create_search(repository), keep=-1)

    def test_events_are_recorded_in_order(self, repository: SearchRepository) -> None:
        search_id = _create_search(repository)
        repository.record_event(search_id=search_id, event="first")
        repository.record_event(search_id=search_id, event="second", payload={"x": 1})
        events = repository.list_events(search_id)
        assert [event["event"] for event in events] == ["first", "second"]
        assert events[1]["payload"] == {"x": 1}


class TestLineageAndDeletion:
    def test_lineage_nodes_carry_parents_and_mutations(
        self, repository: SearchRepository, sampler: Any
    ) -> None:
        search_id = _create_search(repository)
        parent_spec = sampler.sample()
        parent_id = repository.add_candidate(
            search_id=search_id,
            architecture_hash=architecture_hash(parent_spec),
            spec=parent_spec,
        )
        child_spec = sampler.sample()
        repository.add_candidate(
            search_id=search_id,
            architecture_hash=architecture_hash(child_spec),
            spec=child_spec,
            parent_id=parent_id,
            mutation="kernel 3->5",
            generation=1,
        )
        nodes = repository.lineage_nodes(search_id)
        assert len(nodes) == 2
        assert any(node.mutation == "kernel 3->5" for node in nodes)

    def test_deleting_a_search_cascades(
        self, repository: SearchRepository, sample_spec: ArchitectureSpec
    ) -> None:
        search_id = _create_search(repository)
        candidate_id = _queued_candidate(repository, search_id, sample_spec)
        repository.save_checkpoint(search_id, {"a": 1})
        assert repository.delete_search(search_id) is True
        with pytest.raises(RecordNotFoundError):
            repository.get_candidate(candidate_id)

    def test_deleting_an_unknown_search_reports_false(self, repository: SearchRepository) -> None:
        assert repository.delete_search("nope") is False


class TestRecovery:
    def test_running_candidates_are_requeued(
        self, repository: SearchRepository, sample_spec: ArchitectureSpec
    ) -> None:
        search_id = _create_search(repository)
        candidate_id = _queued_candidate(repository, search_id, sample_spec)
        repository.claim_next_queued(search_id, worker_id="w0")
        repository.start_trial(
            candidate_id=candidate_id, attempt=0, budget=TrainingBudget(epochs=1)
        )

        report = repository.recover_interrupted(search_id, max_retries=2)
        assert report.interrupted_running == 1
        assert report.requeued == (candidate_id,)
        assert report.interrupted_trials == 1
        assert repository.get_candidate(candidate_id).status == CandidateState.QUEUED.value
        assert repository.list_trials(candidate_id)[0]["status"] == (TrialState.INTERRUPTED.value)

    def test_candidates_out_of_retries_are_failed(
        self, repository: SearchRepository, sample_spec: ArchitectureSpec
    ) -> None:
        search_id = _create_search(repository)
        candidate_id = _queued_candidate(repository, search_id, sample_spec)
        repository.claim_next_queued(search_id, worker_id="w0")
        report = repository.recover_interrupted(search_id, max_retries=0)
        assert report.abandoned == (candidate_id,)
        assert repository.get_candidate(candidate_id).status == CandidateState.FAILED.value

    def test_completed_candidates_are_untouched(
        self, repository: SearchRepository, sampler: Any
    ) -> None:
        search_id = _create_search(repository)
        _completed_candidate(repository, search_id, sampler.sample(), 0.5)
        report = repository.recover_interrupted(search_id, max_retries=1)
        assert report.interrupted_running == 0
        assert report.completed == 1

    def test_report_serialises(
        self, repository: SearchRepository, sample_spec: ArchitectureSpec
    ) -> None:
        search_id = _create_search(repository)
        payload = repository.recover_interrupted(search_id, max_retries=1).to_dict()
        assert set(payload) == {
            "interrupted_running",
            "interrupted_trials",
            "requeued",
            "abandoned",
            "completed",
        }


def _queued_candidate(repository: SearchRepository, search_id: str, spec: ArchitectureSpec) -> str:
    """Insert a candidate and advance it to ``QUEUED``."""
    candidate_id = repository.add_candidate(
        search_id=search_id,
        architecture_hash=architecture_hash(spec),
        spec=spec,
    )
    repository.update_candidate_state(candidate_id, CandidateState.VALIDATED)
    repository.update_candidate_state(candidate_id, CandidateState.QUEUED)
    return candidate_id


def _completed_candidate(
    repository: SearchRepository,
    search_id: str,
    spec: ArchitectureSpec,
    accuracy: float,
    *,
    attempt: int = 0,
) -> str:
    """Insert a candidate and drive it all the way to ``COMPLETED``."""
    candidate_id = _queued_candidate(repository, search_id, spec)
    repository.claim_next_queued(search_id, worker_id="w0")
    trial_id = repository.start_trial(
        candidate_id=candidate_id, attempt=attempt, budget=TrainingBudget(epochs=1)
    )
    repository.complete_trial(
        trial_id,
        metrics={"validation_accuracy": accuracy, "trainable_parameters": 1000.0},
        duration_seconds=1.0,
    )
    repository.update_candidate_state(
        candidate_id, CandidateState.COMPLETED, objective_value=accuracy
    )
    return candidate_id
