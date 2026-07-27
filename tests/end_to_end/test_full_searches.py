"""End-to-end searches on CPU with synthetic data.

These are the tests that answer "does the whole thing work?". Each runs a complete search
through the public API or the CLI, on a tiny synthetic dataset, with no network access and
no GPU. Together they cover the acceptance criteria: every strategy runs, results persist,
an interrupted search resumes, reports and exports are produced, and the winning
architecture can be reloaded and used for inference.

Runtime is kept to a few seconds per test by using a one-epoch budget on a 96-example
dataset. The genuinely slower variants are marked ``slow`` and excluded from the default
run; the nightly workflow includes them.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

from nas_engine.config.models import SearchConfig
from nas_engine.orchestration.engine import SearchEngine
from nas_engine.orchestration.lifecycle import CandidateState
from nas_engine.orchestration.result import StopReason
from nas_engine.reporting.report import ReportGenerator

pytestmark = pytest.mark.e2e


def _generate_report(config: SearchConfig, engine: SearchEngine, search_id: str) -> object:
    """Generate a report for a finished search."""
    generator = ReportGenerator(
        engine.repository,
        objectives=config.objectives.build_objectives(),
        constraints=config.objectives.build_constraints(),
        output_dir=config.report_dir,
        artifact_root=config.artifact_dir,
    )
    return generator.generate(search_id)


class TestRandomSearchEndToEnd:
    def test_completes_and_produces_a_winner(self, config_factory: object) -> None:
        config = config_factory(  # type: ignore[operator]
            algorithm={"name": "random_search"},
            budget={"max_evaluations": 4, "epochs": 1},
        )
        engine = SearchEngine(config, configure_process=False)
        try:
            result = engine.run()
            assert result.stop_reason is StopReason.BUDGET_EXHAUSTED
            assert result.engine_state.completed == 4
            assert result.best is not None
            assert result.pareto_front
            assert result.status == "completed"
        finally:
            engine.close()

    def test_all_candidates_are_unique(self, config_factory: object) -> None:
        config = config_factory(  # type: ignore[operator]
            budget={"max_evaluations": 5, "epochs": 1}
        )
        engine = SearchEngine(config, configure_process=False)
        try:
            result = engine.run()
            hashes = [candidate.architecture_hash for candidate in result.ranked]
            assert len(set(hashes)) == len(hashes)
        finally:
            engine.close()


class TestEvolutionEndToEnd:
    def test_population_fills_and_ages(self, config_factory: object) -> None:
        config = config_factory(  # type: ignore[operator]
            algorithm={
                "name": "regularized_evolution",
                "params": {"population_size": 3, "tournament_size": 2},
            },
            budget={"max_evaluations": 7, "epochs": 1},
        )
        engine = SearchEngine(config, configure_process=False)
        try:
            result = engine.run()
            assert result.engine_state.completed == 7
            population = result.strategy_statistics["population"]
            assert population["size"] == 3
            assert result.strategy_statistics["retired"] >= 1
        finally:
            engine.close()

    def test_lineage_is_recorded(self, config_factory: object) -> None:
        from nas_engine.architectures.lineage import LineageGraph

        config = config_factory(  # type: ignore[operator]
            algorithm={
                "name": "regularized_evolution",
                "params": {"population_size": 2, "tournament_size": 2},
            },
            budget={"max_evaluations": 6, "epochs": 1},
        )
        engine = SearchEngine(config, configure_process=False)
        try:
            result = engine.run()
            graph = LineageGraph.from_nodes(engine.repository.lineage_nodes(result.search_id))
            statistics = graph.statistics()
            assert statistics["nodes"] == 6
            assert statistics["mutated_nodes"] >= 1
            assert statistics["max_depth"] >= 2
        finally:
            engine.close()


class TestSuccessiveHalvingEndToEnd:
    def test_the_ladder_is_climbed(self, config_factory: object) -> None:
        config = config_factory(  # type: ignore[operator]
            algorithm={
                "name": "successive_halving",
                "params": {
                    "initial_candidates": 4,
                    "num_rungs": 3,
                    "reduction_factor": 2.0,
                },
            },
            budget={"max_evaluations": 20, "epochs": 1},
        )
        engine = SearchEngine(config, configure_process=False)
        try:
            result = engine.run()
            rungs = result.strategy_statistics["rungs"]
            assert [rung["planned"] for rung in rungs] == [4, 2, 1]
            assert [rung["completed"] for rung in rungs] == [4, 2, 1]
            assert result.engine_state.completed == 7
        finally:
            engine.close()

    def test_the_same_architecture_is_re_evaluated_at_a_higher_rung(
        self, config_factory: object
    ) -> None:
        config = config_factory(  # type: ignore[operator]
            algorithm={
                "name": "successive_halving",
                "params": {
                    "initial_candidates": 4,
                    "num_rungs": 2,
                    "reduction_factor": 2.0,
                },
            },
            budget={"max_evaluations": 20, "epochs": 1},
        )
        engine = SearchEngine(config, configure_process=False)
        try:
            result = engine.run()
            candidates = engine.repository.list_candidates(result.search_id)
            rungs = {candidate.rung for candidate in candidates}
            assert rungs == {0, 1}
            # A promoted architecture appears once per rung, not as a rejected duplicate.
            assert result.engine_state.duplicates == 0
        finally:
            engine.close()


class TestInterruptionAndResume:
    def test_a_crashed_evaluation_is_recovered_on_resume(self, config_factory: object) -> None:
        from nas_engine.persistence.models import CandidateRecord

        config = config_factory(  # type: ignore[operator]
            budget={"max_evaluations": 3, "epochs": 1}, retry={"max_retries": 1}
        )
        engine = SearchEngine(config, configure_process=False)
        try:
            first = engine.run()
        finally:
            engine.close()

        # Simulate a process that died mid-evaluation by forcing a completed candidate
        # back into RUNNING, exactly the state a crash leaves behind.
        recovery_engine = SearchEngine(config, configure_process=False)
        try:
            victim = recovery_engine.repository.list_candidates(first.search_id)[0]
            with recovery_engine.repository.database.session() as session:
                record = session.get(CandidateRecord, victim.id)
                assert record is not None
                record.status = CandidateState.RUNNING.value

            second = recovery_engine.resume(first.search_id)
            assert second.resumed
            assert any("recovered" in warning for warning in second.warnings)
            counts = recovery_engine.repository.count_candidates_by_status(first.search_id)
            assert counts["running"] == 0
            assert counts["completed"] >= 3
        finally:
            recovery_engine.close()

    def test_resuming_a_finished_search_is_a_no_op(self, config_factory: object) -> None:
        config = config_factory(budget={"max_evaluations": 2, "epochs": 1})  # type: ignore[operator]
        engine = SearchEngine(config, configure_process=False)
        try:
            first = engine.run()
        finally:
            engine.close()

        again = SearchEngine(config, configure_process=False)
        try:
            second = again.resume(first.search_id)
            assert second.engine_state.completed == first.engine_state.completed
            assert second.engine_state.proposed == first.engine_state.proposed
        finally:
            again.close()


class TestArtifactsAndExports:
    def test_the_full_artifact_set_is_produced(self, config_factory: object) -> None:
        config = config_factory(  # type: ignore[operator]
            budget={"max_evaluations": 4, "epochs": 1},
            evaluation={
                "measure_latency": True,
                "latency_repeats": 2,
                "latency_timed_iterations": 2,
                "save_weights": True,
            },
        )
        engine = SearchEngine(config, configure_process=False)
        try:
            result = engine.run()
            artifacts = _generate_report(config, engine, result.search_id)
        finally:
            engine.close()

        assert artifacts.markdown.is_file()  # type: ignore[attr-defined]
        assert artifacts.csv.is_file()  # type: ignore[attr-defined]
        assert artifacts.json.is_file()  # type: ignore[attr-defined]
        assert set(artifacts.plots) >= {  # type: ignore[attr-defined]
            "accuracy_vs_parameters",
            "search_progress",
        }

    def test_csv_export_lists_every_candidate(self, config_factory: object) -> None:
        config = config_factory(budget={"max_evaluations": 4, "epochs": 1})  # type: ignore[operator]
        engine = SearchEngine(config, configure_process=False)
        try:
            result = engine.run()
            artifacts = _generate_report(config, engine, result.search_id)
        finally:
            engine.close()

        rows = list(csv.DictReader(artifacts.csv.open(encoding="utf-8")))  # type: ignore[attr-defined]
        assert len(rows) == 4
        assert {"rank", "architecture_hash", "validation_accuracy"} <= set(rows[0])

    def test_json_export_is_self_describing(self, config_factory: object) -> None:
        config = config_factory(budget={"max_evaluations": 3, "epochs": 1})  # type: ignore[operator]
        engine = SearchEngine(config, configure_process=False)
        try:
            result = engine.run()
            artifacts = _generate_report(config, engine, result.search_id)
        finally:
            engine.close()

        payload = json.loads(artifacts.json.read_text(encoding="utf-8"))  # type: ignore[attr-defined]
        assert payload["search"]["id"] == result.search_id
        assert payload["configuration"]["algorithm"]["name"] == "random_search"
        assert payload["environment"]["torch_version"]
        assert payload["limitations"]

    def test_report_names_are_deterministic(self, config_factory: object) -> None:
        config = config_factory(budget={"max_evaluations": 3, "epochs": 1})  # type: ignore[operator]
        engine = SearchEngine(config, configure_process=False)
        try:
            result = engine.run()
            first = _generate_report(config, engine, result.search_id)
            second = _generate_report(config, engine, result.search_id)
        finally:
            engine.close()

        assert first.markdown == second.markdown  # type: ignore[attr-defined]
        assert first.csv == second.csv  # type: ignore[attr-defined]


class TestBestModelReload:
    def test_the_winner_reloads_and_performs_inference(self, config_factory: object) -> None:
        config = config_factory(budget={"max_evaluations": 3, "epochs": 1})  # type: ignore[operator]
        engine = SearchEngine(config, configure_process=False)
        try:
            result = engine.run()
            spec, model = engine.load_best_model(result.search_id)
            model.eval()
            with torch.no_grad():
                logits = model(
                    torch.randn(4, spec.input_channels, spec.input_size, spec.input_size)
                )
            assert logits.shape == (4, spec.num_classes)
            assert torch.isfinite(logits).all()
        finally:
            engine.close()

    def test_reloaded_weights_match_the_saved_ones(self, config_factory: object) -> None:
        config = config_factory(budget={"max_evaluations": 2, "epochs": 1})  # type: ignore[operator]
        engine = SearchEngine(config, configure_process=False)
        try:
            result = engine.run()
            assert result.best is not None
            candidate = engine.repository.get_candidate(result.best.candidate_id)
            weights_path = engine.artifact_root / candidate.artifacts["weights"]
            saved = torch.load(weights_path, map_location="cpu", weights_only=True)
            _, model = engine.load_best_model(result.search_id)
            for key, value in model.state_dict().items():
                assert torch.equal(value, saved[key])
        finally:
            engine.close()

    def test_the_winner_can_be_scored_on_the_held_out_test_split(
        self, config_factory: object
    ) -> None:
        config = config_factory(budget={"max_evaluations": 2, "epochs": 1})  # type: ignore[operator]
        engine = SearchEngine(config, configure_process=False)
        try:
            result = engine.run()
            assert result.best is not None
            candidate = engine.repository.get_candidate(result.best.candidate_id)
            spec = engine.repository.get_candidate_spec(candidate.id)
            weights = engine.artifact_root / candidate.artifacts["weights"]
            metrics = engine.evaluator.evaluate_on_test(spec, weights_path=weights)
            assert 0.0 <= metrics["test_accuracy"] <= 1.0
            assert metrics["test_examples"] == 48
        finally:
            engine.close()


class TestCommandLineEndToEnd:
    def test_the_documented_command_sequence_works(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from nas_engine.cli import app
        from tests.conftest import build_smoke_config

        runner = CliRunner()
        config_path = tmp_path / "config.yaml"
        build_smoke_config(tmp_path / "out").to_yaml(config_path)
        base = ["--config", str(config_path)]

        assert runner.invoke(app, ["validate-config", *base]).exit_code == 0
        assert runner.invoke(app, ["search", *base]).exit_code == 0
        assert runner.invoke(app, ["status", *base]).exit_code == 0
        assert runner.invoke(app, ["list-candidates", *base]).exit_code == 0
        assert runner.invoke(app, ["best", *base]).exit_code == 0
        assert runner.invoke(app, ["pareto", *base]).exit_code == 0
        assert runner.invoke(app, ["export", *base, "--format", "csv"]).exit_code == 0
        assert runner.invoke(app, ["export", *base, "--format", "json"]).exit_code == 0
        assert runner.invoke(app, ["report", *base]).exit_code == 0
        assert runner.invoke(app, ["evaluate", *base]).exit_code == 0
        assert runner.invoke(app, ["resume", *base]).exit_code == 0

    def test_json_output_is_parseable_for_every_query_command(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from nas_engine.cli import app
        from tests.conftest import build_smoke_config

        runner = CliRunner()
        config_path = tmp_path / "config.yaml"
        build_smoke_config(tmp_path / "out").to_yaml(config_path)
        base = ["--config", str(config_path)]
        runner.invoke(app, ["search", *base])

        for command in ("status", "list-candidates", "best", "pareto"):
            result = runner.invoke(app, [command, *base, "--json"])
            assert result.exit_code == 0, command
            json.loads(result.output)

    def test_show_candidate_accepts_a_hash_prefix(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from nas_engine.cli import app
        from tests.conftest import build_smoke_config

        runner = CliRunner()
        config_path = tmp_path / "config.yaml"
        build_smoke_config(tmp_path / "out").to_yaml(config_path)
        base = ["--config", str(config_path)]
        runner.invoke(app, ["search", *base])

        best = json.loads(runner.invoke(app, ["best", *base, "--json"]).output)
        prefix = best["best"]["architecture_hash"][:10]
        result = runner.invoke(app, ["show-candidate", prefix, *base, "--json"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["architecture"]["hash"].startswith(prefix)


@pytest.mark.slow
class TestLargerSearches:
    def test_a_longer_evolution_run_improves_on_its_first_candidate(
        self, config_factory: object
    ) -> None:
        config = config_factory(  # type: ignore[operator]
            algorithm={
                "name": "regularized_evolution",
                "params": {"population_size": 5, "tournament_size": 3},
            },
            budget={"max_evaluations": 16, "epochs": 2},
            dataset={
                "provider": "synthetic",
                "batch_size": 32,
                "options": {
                    "num_classes": 4,
                    "input_size": 16,
                    "train_samples": 256,
                    "validation_samples": 128,
                    "test_samples": 128,
                    "noise_scale": 0.4,
                    "seed": 1234,
                },
            },
        )
        engine = SearchEngine(config, configure_process=False)
        try:
            result = engine.run()
            assert result.engine_state.completed == 16
            assert result.best is not None
            accuracies = [candidate.metrics["validation_accuracy"] for candidate in result.ranked]
            assert max(accuracies) > min(accuracies)
        finally:
            engine.close()

    def test_multiprocessing_produces_the_same_kind_of_result(self, config_factory: object) -> None:
        config = config_factory(  # type: ignore[operator]
            budget={"max_evaluations": 4, "epochs": 1},
            concurrency={"mode": "multiprocessing", "workers": 2, "start_method": "spawn"},
        )
        engine = SearchEngine(config, configure_process=False)
        try:
            result = engine.run()
            assert result.engine_state.completed == 4
            assert result.engine_state.failed == 0
            assert result.best is not None
        finally:
            engine.close()
