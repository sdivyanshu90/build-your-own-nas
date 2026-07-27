"""Golden-fixture regression tests.

These tests pin values that must not drift silently:

* canonical architecture JSON and its hash — a change invalidates every stored hash and
  makes historical results incomparable;
* analytic parameter counts and MACs — the objective values a search optimises;
* built model output shapes;
* Pareto-front outcomes for hand-checked cases;
* the candidate state-transition table;
* the structure of a generated report and its exports.

Fixtures are **intentionally versioned**. Each file carries a ``fixture_version``; changing
a golden value requires bumping it and recording why. A test failure here is not
necessarily a bug — it is a demand for a deliberate decision.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from nas_engine.architectures.canonical import from_canonical_dict, to_canonical_dict
from nas_engine.architectures.cost import compute_cost
from nas_engine.architectures.hashing import architecture_hash
from nas_engine.architectures.spec import ARCHITECTURE_SCHEMA_VERSION
from nas_engine.models.builder import ModelBuilder, count_parameters
from nas_engine.objectives.pareto import ObjectiveVector, pareto_front
from nas_engine.orchestration.engine import SearchEngine
from nas_engine.orchestration.lifecycle import ALLOWED_TRANSITIONS
from nas_engine.reporting.report import ReportGenerator
from tests.conftest import FIXTURE_DIR

pytestmark = pytest.mark.regression


def _load(name: str) -> dict[str, Any]:
    """Read one fixture file."""
    payload = json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


ARCHITECTURES = _load("architectures.json")
TRANSITIONS = _load("state_transitions.json")
PARETO = _load("pareto_cases.json")
REPORT = _load("report_structure.json")


class TestArchitectureFixtures:
    def test_the_fixture_targets_the_current_schema(self) -> None:
        assert ARCHITECTURES["architecture_schema_version"] == ARCHITECTURE_SCHEMA_VERSION

    @pytest.mark.parametrize("case", ARCHITECTURES["cases"], ids=lambda case: case["name"])
    def test_stored_architectures_still_parse(self, case: dict[str, Any]) -> None:
        spec = from_canonical_dict(case["architecture"])
        assert spec.num_classes == case["expected_output_shape"][1]

    @pytest.mark.parametrize("case", ARCHITECTURES["cases"], ids=lambda case: case["name"])
    def test_hashes_are_unchanged(self, case: dict[str, Any]) -> None:
        spec = from_canonical_dict(case["architecture"])
        assert architecture_hash(spec) == case["expected_hash"], (
            "The architecture hash changed. Every stored hash in every existing database "
            "is now wrong. If this change is intentional, bump fixture_version in "
            "tests/fixtures/architectures.json and document the migration."
        )

    @pytest.mark.parametrize("case", ARCHITECTURES["cases"], ids=lambda case: case["name"])
    def test_canonical_form_round_trips_to_the_same_bytes(self, case: dict[str, Any]) -> None:
        spec = from_canonical_dict(case["architecture"])
        assert to_canonical_dict(spec) == case["architecture"]

    @pytest.mark.parametrize("case", ARCHITECTURES["cases"], ids=lambda case: case["name"])
    def test_parameter_counts_are_unchanged(self, case: dict[str, Any]) -> None:
        cost = compute_cost(from_canonical_dict(case["architecture"]))
        assert cost.trainable_parameters == case["expected_trainable_parameters"]
        assert cost.non_trainable_parameters == case["expected_non_trainable_parameters"]

    @pytest.mark.parametrize("case", ARCHITECTURES["cases"], ids=lambda case: case["name"])
    def test_mac_counts_are_unchanged(self, case: dict[str, Any]) -> None:
        cost = compute_cost(from_canonical_dict(case["architecture"]))
        assert cost.multiply_accumulates == case["expected_multiply_accumulates"]

    @pytest.mark.parametrize("case", ARCHITECTURES["cases"], ids=lambda case: case["name"])
    def test_total_stride_is_unchanged(self, case: dict[str, Any]) -> None:
        spec = from_canonical_dict(case["architecture"])
        assert spec.total_stride == case["expected_total_stride"]

    @pytest.mark.parametrize("case", ARCHITECTURES["cases"], ids=lambda case: case["name"])
    def test_built_models_match_the_recorded_shape_and_count(self, case: dict[str, Any]) -> None:
        spec = from_canonical_dict(case["architecture"])
        model = ModelBuilder(initialize=False).build(spec)
        batch, classes = case["expected_output_shape"]
        output = model(torch.zeros(batch, spec.input_channels, spec.input_size, spec.input_size))
        assert list(output.shape) == [batch, classes]
        trainable, _ = count_parameters(model)
        assert trainable == case["expected_trainable_parameters"]


class TestStateMachineFixture:
    def test_the_transition_table_is_unchanged(self) -> None:
        current = {
            state.value: sorted(target.value for target in targets)
            for state, targets in ALLOWED_TRANSITIONS.items()
        }
        assert current == TRANSITIONS["transitions"], (
            "The candidate state machine changed. Recovery logic depends on these edges; "
            "review docs/architecture/component-design.md before updating the fixture."
        )


class TestParetoFixture:
    @pytest.mark.parametrize("case", PARETO["cases"], ids=lambda case: case["name"])
    def test_fronts_match_the_hand_checked_answers(self, case: dict[str, Any]) -> None:
        vectors = [
            ObjectiveVector(name, tuple(values), tuple(values))
            for name, values in sorted(case["vectors"].items())
        ]
        front = sorted(vector.candidate_id for vector in pareto_front(vectors))
        assert front == sorted(case["front"])


class TestReportStructureFixture:
    def test_a_generated_report_contains_every_required_section(
        self, config_factory: object, tmp_path: Path
    ) -> None:
        config = config_factory(  # type: ignore[operator]
            budget={"max_evaluations": 3, "epochs": 1},
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
            generator = ReportGenerator(
                engine.repository,
                objectives=config.objectives.build_objectives(),
                constraints=config.objectives.build_constraints(),
                output_dir=tmp_path,
                artifact_root=config.artifact_dir,
            )
            artifacts = generator.generate(result.search_id)
        finally:
            engine.close()

        text = artifacts.markdown.read_text(encoding="utf-8")
        position = -1
        for heading in REPORT["required_sections"]:
            found = text.find(heading)
            assert found >= 0, f"report is missing the section {heading!r}"
            assert found > position, f"section {heading!r} appears out of order"
            position = found

        payload = json.loads(artifacts.json.read_text(encoding="utf-8"))
        assert set(REPORT["required_json_keys"]) <= set(payload)

        with artifacts.csv.open(encoding="utf-8") as handle:
            header = next(csv.reader(handle))
        assert header[: len(REPORT["required_csv_columns"])] == REPORT["required_csv_columns"]


class TestFixtureIntegrity:
    @pytest.mark.parametrize("payload", [ARCHITECTURES, TRANSITIONS, PARETO, REPORT])
    def test_every_fixture_declares_a_version(self, payload: dict[str, Any]) -> None:
        assert payload["fixture_version"] >= 1

    @pytest.mark.parametrize("payload", [ARCHITECTURES, TRANSITIONS, PARETO, REPORT])
    def test_every_fixture_explains_itself(self, payload: dict[str, Any]) -> None:
        assert payload["note"]

    def test_fixture_hashes_are_distinct(self) -> None:
        hashes = [case["expected_hash"] for case in ARCHITECTURES["cases"]]
        assert len(set(hashes)) == len(hashes)
