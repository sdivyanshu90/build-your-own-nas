"""Integration tests across component boundaries.

Each test here exercises a *seam* — a place where two components must agree — rather than
either component in isolation:

* configuration → search space and strategy construction;
* architecture specification → model builder → forward pass;
* static shape inference → what PyTorch actually computes;
* trainer → evaluator → metric mapping;
* strategy → engine → repository;
* engine → checkpoint → resume;
* repository → report generator.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from nas_engine.architectures.hashing import architecture_hash
from nas_engine.architectures.shapes import infer_shapes
from nas_engine.config.models import SearchConfig
from nas_engine.datasets.base import DatasetBundle
from nas_engine.datasets.loaders import LoaderSettings, build_dataloaders
from nas_engine.evaluation.budget import TrainingBudget
from nas_engine.evaluation.evaluator import (
    CandidateEvaluator,
    EvaluationContext,
    EvaluationSettings,
)
from nas_engine.models.builder import ModelBuilder
from nas_engine.objectives.ranking import rank_candidates
from nas_engine.orchestration.engine import SearchEngine
from nas_engine.orchestration.lifecycle import CandidateState
from nas_engine.persistence.database import Database
from nas_engine.reporting.report import ReportGenerator
from nas_engine.search.registry import build_strategy
from nas_engine.search_space.sampler import ArchitectureSampler
from nas_engine.training.optimizers import OptimizerSettings
from nas_engine.training.trainer import Trainer, TrainingSettings

pytestmark = pytest.mark.integration


class TestConfigurationToComponents:
    def test_configuration_builds_a_search_space(self, smoke_config: SearchConfig) -> None:
        space = smoke_config.search_space.build(input_size=16, num_classes=4)
        assert space.input_size == 16
        assert space.num_classes == 4
        space.require_non_empty()

    def test_configuration_builds_every_strategy(self, smoke_config: SearchConfig) -> None:
        space = smoke_config.search_space.build(input_size=16, num_classes=4)
        for name in ("random_search", "regularized_evolution", "successive_halving"):
            strategy = build_strategy(
                name,
                space=space,
                seed=1,
                budget=smoke_config.budget.build_budget(),
                max_evaluations=smoke_config.budget.max_evaluations,
                native_resolution=16,
            )
            assert strategy.name == name

    def test_dataset_shape_flows_into_the_search_space(self, smoke_config: SearchConfig) -> None:
        from nas_engine.datasets.registry import build_dataset

        bundle = build_dataset(smoke_config.dataset.provider, **smoke_config.dataset.options)
        space = smoke_config.search_space.build(
            input_size=bundle.input_size,
            num_classes=bundle.num_classes,
            input_channels=bundle.input_channels,
        )
        spec = ArchitectureSampler(space, seed=0).sample()
        assert spec.input_size == bundle.input_size
        assert spec.num_classes == bundle.num_classes


class TestShapeInferenceMatchesTorch:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7])
    def test_every_intermediate_shape_agrees(self, seed: int, smoke_config: SearchConfig) -> None:
        space = smoke_config.search_space.build(input_size=16, num_classes=4)
        spec = ArchitectureSampler(space, seed=seed).sample()
        model = ModelBuilder(initialize=False).build(spec)
        static = infer_shapes(spec)
        runtime = dict(model.feature_shapes(torch.zeros(1, 3, 16, 16)))

        assert runtime["stem"][1:] == static.layers[0].output_shape.as_tuple()
        assert runtime[f"stages.{spec.num_stages - 1}"][1:] == (static.features_shape.as_tuple())
        assert runtime["head"] == (1, static.output_features)

    def test_the_analytic_cost_model_tracks_the_builder(self, smoke_config: SearchConfig) -> None:
        space = smoke_config.search_space.build(input_size=16, num_classes=4)
        sampler = ArchitectureSampler(space, seed=42)
        builder = ModelBuilder(initialize=False)
        for _ in range(12):
            _, summary = builder.build_and_summarize(sampler.sample())
            assert summary.matches_analytic_estimate


class TestTrainerAndEvaluator:
    def test_evaluator_metrics_match_a_direct_training_run(
        self, synthetic_bundle: DatasetBundle, tmp_path: Path, smoke_config: SearchConfig
    ) -> None:
        space = smoke_config.search_space.build(input_size=16, num_classes=4)
        spec = ArchitectureSampler(space, seed=3).sample()
        settings = TrainingSettings(
            epochs=1, optimizer=OptimizerSettings(learning_rate=0.01), topk=2
        )

        evaluator = CandidateEvaluator(
            dataset=synthetic_bundle,
            loader_settings=LoaderSettings(batch_size=32),
            training_settings=settings,
            settings=EvaluationSettings(measure_latency=False, save_weights=False),
            artifact_root=tmp_path,
            device="cpu",
            seed=17,
        )
        result = evaluator.evaluate(
            spec, TrainingBudget(epochs=1), EvaluationContext(candidate_id="c", trial_id="t")
        )
        assert result.succeeded

        # Reproduce the same run outside the evaluator, seeding exactly as it does.
        from nas_engine.utilities.seeding import derive_seed, seed_everything

        seed_everything(evaluator.candidate_seed(spec, TrainingBudget(epochs=1)))
        loaders = build_dataloaders(
            synthetic_bundle,
            LoaderSettings(batch_size=32),
            seed=derive_seed(17, "loaders"),
        )
        outcome = Trainer(settings, device="cpu").fit(
            ModelBuilder().build(spec), loaders, architecture_hash=architecture_hash(spec)
        )
        assert outcome.best_validation_accuracy == pytest.approx(
            result.metrics["validation_accuracy"]
        )


class TestEngineAndRepository:
    def test_a_search_persists_every_candidate(self, smoke_config: SearchConfig) -> None:
        engine = SearchEngine(smoke_config, configure_process=False)
        try:
            result = engine.run()
            counts = engine.repository.count_candidates_by_status(result.search_id)
            assert counts["completed"] == result.engine_state.completed
            assert sum(counts.values()) == result.engine_state.accepted + (
                result.engine_state.invalid + result.engine_state.pruned
            )
        finally:
            engine.close()

    def test_trials_and_metrics_are_recorded(self, smoke_config: SearchConfig) -> None:
        engine = SearchEngine(smoke_config, configure_process=False)
        try:
            result = engine.run()
            candidates = engine.repository.list_candidates(
                result.search_id, statuses=[CandidateState.COMPLETED]
            )
            assert candidates
            for candidate in candidates:
                assert candidate.metrics["validation_accuracy"] >= 0.0
                assert candidate.trial_count >= 1
                assert candidate.artifacts["weights"]
        finally:
            engine.close()

    def test_the_engine_writes_checkpoints(self, smoke_config: SearchConfig) -> None:
        engine = SearchEngine(smoke_config, configure_process=False)
        try:
            result = engine.run()
            payload = engine.repository.latest_checkpoint(result.search_id)
            assert payload is not None
            assert payload["strategy_name"] == "random_search"
            assert payload["engine_state"]["completed"] == result.engine_state.completed
        finally:
            engine.close()

    def test_checkpoints_are_pruned_to_the_configured_depth(self, config_factory: object) -> None:
        config = config_factory(  # type: ignore[operator]
            budget={"max_evaluations": 5, "epochs": 1},
            persistence={"checkpoint_every": 1, "keep_checkpoints": 2},
        )
        engine = SearchEngine(config, configure_process=False)
        try:
            result = engine.run()
            assert engine.repository.count_checkpoints(result.search_id) <= 2
        finally:
            engine.close()

    def test_events_record_the_search_lifecycle(self, smoke_config: SearchConfig) -> None:
        engine = SearchEngine(smoke_config, configure_process=False)
        try:
            result = engine.run()
            events = engine.repository.list_events(result.search_id)
            names = {event["event"] for event in events}
            assert any(name.startswith("candidate.") for name in names)
        finally:
            engine.close()


class TestCheckpointAndResume:
    def test_resume_continues_rather_than_restarting(self, config_factory: object) -> None:
        first_config = config_factory(budget={"max_evaluations": 2, "epochs": 1})  # type: ignore[operator]
        engine = SearchEngine(first_config, configure_process=False)
        try:
            first = engine.run()
        finally:
            engine.close()

        second_config = config_factory(budget={"max_evaluations": 4, "epochs": 1})  # type: ignore[operator]
        resumed_engine = SearchEngine(second_config, configure_process=False)
        try:
            second = resumed_engine.resume(first.search_id)
            assert second.resumed
            assert second.search_id == first.search_id
            assert second.engine_state.completed == 4
            assert second.engine_state.duplicates == 0
        finally:
            resumed_engine.close()

    def test_resume_reuses_the_same_database_rows(self, config_factory: object) -> None:
        config = config_factory(budget={"max_evaluations": 2, "epochs": 1})  # type: ignore[operator]
        engine = SearchEngine(config, configure_process=False)
        try:
            first = engine.run()
            hashes_before = engine.repository.seen_hashes(first.search_id)
        finally:
            engine.close()

        wider = config_factory(budget={"max_evaluations": 4, "epochs": 1})  # type: ignore[operator]
        resumed = SearchEngine(wider, configure_process=False)
        try:
            second = resumed.resume(first.search_id)
            hashes_after = resumed.repository.seen_hashes(second.search_id)
            assert hashes_before <= hashes_after
            assert len(hashes_after) > len(hashes_before)
        finally:
            resumed.close()


class TestReportingFromPersistedResults:
    def test_a_report_can_be_generated_from_the_database_alone(
        self, smoke_config: SearchConfig, tmp_path: Path
    ) -> None:
        engine = SearchEngine(smoke_config, configure_process=False)
        try:
            result = engine.run()
        finally:
            engine.close()

        # Re-open the database from scratch: the report generator must not need the engine.
        database = Database(smoke_config.database_url)
        try:
            from nas_engine.persistence.repository import SearchRepository

            generator = ReportGenerator(
                SearchRepository(database),
                objectives=smoke_config.objectives.build_objectives(),
                constraints=smoke_config.objectives.build_constraints(),
                output_dir=tmp_path / "reports",
                artifact_root=smoke_config.artifact_dir,
            )
            artifacts = generator.generate(result.search_id)
            assert artifacts.markdown.is_file()
            assert artifacts.csv.is_file()
            assert artifacts.json.is_file()
            text = artifacts.markdown.read_text(encoding="utf-8")
            assert "## Known limitations" in text
            assert "## Leaderboard" in text
        finally:
            database.dispose()

    def test_ranking_recomputed_from_the_database_matches_the_engine(
        self, smoke_config: SearchConfig
    ) -> None:
        engine = SearchEngine(smoke_config, configure_process=False)
        try:
            result = engine.run()
            population = engine.repository.completed_metrics(result.search_id)
            recomputed = rank_candidates(
                population,
                smoke_config.objectives.build_objectives(),
                constraints=smoke_config.objectives.build_constraints(),
            )
            assert [c.candidate_id for c in recomputed.ranked] == [
                c.candidate_id for c in result.ranked
            ]
        finally:
            engine.close()
