"""Determinism tests for sequential CPU execution.

What is guaranteed
------------------
Given the same configuration, seed, dataset, library versions, and *sequential* execution
mode on the same machine, a search reproduces:

* the order in which candidates are proposed;
* every architecture hash;
* every mutation decision;
* the strategy's internal state;
* the final ranking.

What is **not** guaranteed, and is deliberately not asserted here:

* bit-identical floating-point metrics across different CPUs, BLAS builds, or PyTorch
  versions — reduction order differs and floating-point addition is not associative;
* identical results under multiprocessing — observation order varies, so an adaptive
  strategy sees a different sequence and may propose differently;
* identical latency measurements — they depend on machine load.

``docs/concepts/reproducibility.md`` explains the distinction between reproducibility,
determinism, and statistical repeatability that these tests operationalise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from nas_engine.architectures.hashing import architecture_hash
from nas_engine.evaluation.budget import TrainingBudget
from nas_engine.models.builder import build_model
from nas_engine.orchestration.engine import SearchEngine
from nas_engine.search.evolution import RegularizedEvolution
from nas_engine.search.random_search import RandomSearch
from nas_engine.search_space.mutation import MutationOperator
from nas_engine.search_space.presets import default_cnn_space, tiny_cnn_space
from nas_engine.search_space.sampler import ArchitectureSampler
from nas_engine.utilities.seeding import seed_everything
from tests.conftest import build_smoke_config

pytestmark = [pytest.mark.regression, pytest.mark.determinism]


class TestSamplingDeterminism:
    def test_proposal_order_is_reproducible(self) -> None:
        space = default_cnn_space()

        def run() -> list[str]:
            sampler = ArchitectureSampler(space, seed=2024)
            return [architecture_hash(sampler.sample()) for _ in range(15)]

        assert run() == run()

    def test_mutation_decisions_are_reproducible(self) -> None:
        space = default_cnn_space()
        parent = ArchitectureSampler(space, seed=1).sample()

        def run() -> list[tuple[str, str]]:
            mutator = MutationOperator(space, seed=99)
            current = parent
            trace: list[tuple[str, str]] = []
            for _ in range(10):
                result = mutator.mutate(current)
                trace.append((result.operator, result.description))
                current = result.child
            return trace

        assert run() == run()

    def test_strategy_state_is_reproducible(self) -> None:
        space = tiny_cnn_space()

        def run() -> dict[str, Any]:
            strategy = RandomSearch(
                space, seed=7, max_evaluations=6, budget=TrainingBudget(epochs=1)
            )
            strategy.propose(6)
            state = strategy.state_dict()
            return {"seen": state["seen"], "proposed": state["proposed"]}

        assert run() == run()

    def test_evolution_population_is_reproducible(self) -> None:
        from nas_engine.evaluation.result import EvaluationResult
        from nas_engine.search.strategy import Observation

        space = tiny_cnn_space()

        def run() -> list[str]:
            strategy = RegularizedEvolution(
                space,
                seed=11,
                max_evaluations=12,
                budget=TrainingBudget(epochs=1),
                population_size=4,
                tournament_size=2,
            )
            for index in range(12):
                for proposal in strategy.propose(1):
                    digest = architecture_hash(proposal.spec)
                    strategy.observe(
                        Observation(
                            candidate_id=digest,
                            architecture_hash=digest,
                            spec=proposal.spec,
                            result=EvaluationResult(
                                candidate_id=digest,
                                architecture_hash=digest,
                                budget=proposal.budget,
                            ),
                            # A deterministic pseudo-fitness keeps selection reproducible
                            # without running any training.
                            objective_value=(index * 37 % 11) / 11,
                            parent_id=proposal.parent_id,
                        )
                    )
            return [member.architecture_hash for member in strategy.population]

        assert run() == run()


class TestModelDeterminism:
    def test_weights_are_reproducible_for_a_given_seed(self) -> None:
        spec = ArchitectureSampler(tiny_cnn_space(), seed=5).sample()

        seed_everything(31)
        first = {key: value.clone() for key, value in build_model(spec).state_dict().items()}
        seed_everything(31)
        second = build_model(spec).state_dict()
        assert all(torch.equal(first[key], second[key]) for key in first)

    def test_different_seeds_produce_different_weights(self) -> None:
        spec = ArchitectureSampler(tiny_cnn_space(), seed=5).sample()
        seed_everything(1)
        first = build_model(spec).state_dict()
        seed_everything(2)
        second = build_model(spec).state_dict()
        assert any(not torch.equal(first[key], second[key]) for key in first)


class TestSearchDeterminism:
    def test_two_identical_sequential_searches_agree_exactly(self, tmp_path: Path) -> None:
        def run(directory: Path) -> dict[str, Any]:
            config = build_smoke_config(
                directory,
                budget={"max_evaluations": 4, "epochs": 1},
                reproducibility={"seed": 4242, "deterministic": True},
            )
            engine = SearchEngine(config, configure_process=False)
            try:
                result = engine.run()
                return {
                    "hashes": [c.architecture_hash for c in result.ranked],
                    "accuracies": [c.metrics["validation_accuracy"] for c in result.ranked],
                    "parameters": [c.metrics["trainable_parameters"] for c in result.ranked],
                    "best": result.best.architecture_hash if result.best else None,
                    "pareto": [c.architecture_hash for c in result.pareto_front],
                }
            finally:
                engine.close()

        first = run(tmp_path / "a")
        second = run(tmp_path / "b")
        assert first["hashes"] == second["hashes"]
        assert first["accuracies"] == second["accuracies"]
        assert first["parameters"] == second["parameters"]
        assert first["best"] == second["best"]
        assert first["pareto"] == second["pareto"]

    def test_a_different_seed_explores_differently(self, tmp_path: Path) -> None:
        def run(directory: Path, seed: int) -> list[str]:
            config = build_smoke_config(
                directory,
                budget={"max_evaluations": 4, "epochs": 1},
                reproducibility={"seed": seed, "deterministic": True},
            )
            engine = SearchEngine(config, configure_process=False)
            try:
                return sorted(c.architecture_hash for c in engine.run().ranked)
            finally:
                engine.close()

        assert run(tmp_path / "a", 1) != run(tmp_path / "b", 2)

    def test_resume_reaches_the_same_state_as_an_uninterrupted_run(self, tmp_path: Path) -> None:
        # An uninterrupted four-evaluation run.
        whole_config = build_smoke_config(
            tmp_path / "whole",
            budget={"max_evaluations": 4, "epochs": 1},
            reproducibility={"seed": 77, "deterministic": True},
        )
        whole_engine = SearchEngine(whole_config, configure_process=False)
        try:
            whole = whole_engine.run()
            whole_hashes = sorted(c.architecture_hash for c in whole.ranked)
        finally:
            whole_engine.close()

        # The same run split in two, with the second half resumed from the checkpoint.
        split_dir = tmp_path / "split"
        first_config = build_smoke_config(
            split_dir,
            budget={"max_evaluations": 2, "epochs": 1},
            reproducibility={"seed": 77, "deterministic": True},
        )
        first_engine = SearchEngine(first_config, configure_process=False)
        try:
            first = first_engine.run()
        finally:
            first_engine.close()

        second_config = build_smoke_config(
            split_dir,
            budget={"max_evaluations": 4, "epochs": 1},
            reproducibility={"seed": 77, "deterministic": True},
        )
        second_engine = SearchEngine(second_config, configure_process=False)
        try:
            second = second_engine.resume(first.search_id)
            split_hashes = sorted(c.architecture_hash for c in second.ranked)
        finally:
            second_engine.close()

        # The strategy's generator state is checkpointed, so the resumed half continues
        # the same stream rather than replaying it.
        assert split_hashes == whole_hashes

    def test_metrics_are_reproducible_within_one_machine(self, tmp_path: Path) -> None:
        def run(directory: Path) -> list[float]:
            config = build_smoke_config(
                directory,
                budget={"max_evaluations": 3, "epochs": 2},
                reproducibility={"seed": 909, "deterministic": True},
            )
            engine = SearchEngine(config, configure_process=False)
            try:
                return [
                    candidate.metrics["validation_accuracy"] for candidate in engine.run().ranked
                ]
            finally:
                engine.close()

        assert run(tmp_path / "a") == run(tmp_path / "b")


class TestDocumentedNonDeterminism:
    def test_latency_is_not_asserted_to_be_reproducible(self) -> None:
        """Latency varies with machine load, so nothing in the suite pins its value.

        This test documents the decision rather than testing behaviour: if a future change
        adds a latency equality assertion, the reasoning here explains why that assertion
        would be wrong.
        """
        from nas_engine.evaluation.latency import LATENCY_WARNING

        assert "hardware" in LATENCY_WARNING
        assert "same machine" in LATENCY_WARNING

    def test_the_determinism_report_lists_its_caveats(self) -> None:
        from nas_engine.utilities.determinism import configure_determinism

        report = configure_determinism(enabled=True, warn_only=True)
        try:
            payload = report.to_dict()
            assert "warnings" in payload
            assert payload["requested"] is True
        finally:
            configure_determinism(enabled=False)
