"""Unit tests for reporting, exports, and figures.

Covers: CSV column ordering and formula-injection defence, JSON export, figure generation
and skip reasons, and the Markdown report's required sections.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from nas_engine.objectives.ranking import RankedCandidate
from nas_engine.reporting.exporters import (
    BASE_COLUMNS,
    collect_metric_columns,
    export_candidates_csv,
    export_json,
    export_rows_csv,
    sanitize_cell,
)
from nas_engine.reporting.plots import (
    generate_plots,
    plot_accuracy_vs_latency,
    plot_accuracy_vs_parameters,
    plot_population_statistics,
    plot_search_progress,
)
from nas_engine.reporting.report import KNOWN_LIMITATIONS

pytestmark = pytest.mark.unit


def _candidate(
    index: int,
    *,
    accuracy: float,
    parameters: float,
    latency: float | None = None,
    front: bool = False,
) -> RankedCandidate:
    """Build a ranked candidate for report and plot tests."""
    metrics = {"validation_accuracy": accuracy, "trainable_parameters": parameters}
    if latency is not None:
        metrics["latency_median_ms"] = latency
    return RankedCandidate(
        candidate_id=f"c{index}",
        architecture_hash=f"{index:032x}",
        metrics=metrics,
        rank=index,
        score=accuracy,
        pareto_rank=0 if front else 1,
        crowding=0.0,
        feasible=True,
    )


class TestSanitisation:
    @pytest.mark.parametrize("prefix", ["=", "+", "-", "@", "\t", "\r"])
    def test_formula_prefixes_are_neutralised(self, prefix: str) -> None:
        assert sanitize_cell(f"{prefix}cmd").startswith("'")

    def test_ordinary_text_is_unchanged(self) -> None:
        assert sanitize_cell("safe") == "safe"

    def test_numbers_pass_through(self) -> None:
        assert sanitize_cell(-5) == -5
        assert sanitize_cell(3.5) == 3.5


class TestCsvExport:
    def test_columns_include_the_base_set_and_metrics(self, tmp_path: Path) -> None:
        candidates = [_candidate(0, accuracy=0.9, parameters=100.0, latency=1.0)]
        path = export_candidates_csv(candidates, tmp_path / "out.csv")
        with path.open(encoding="utf-8") as handle:
            header = next(csv.reader(handle))
        assert header[: len(BASE_COLUMNS)] == list(BASE_COLUMNS)
        assert "validation_accuracy" in header
        assert "latency_median_ms" in header

    def test_rows_are_written_in_rank_order(self, tmp_path: Path) -> None:
        candidates = [
            _candidate(2, accuracy=0.1, parameters=10.0),
            _candidate(0, accuracy=0.9, parameters=100.0),
            _candidate(1, accuracy=0.5, parameters=50.0),
        ]
        path = export_candidates_csv(candidates, tmp_path / "out.csv")
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        assert [row["rank"] for row in rows] == ["0", "1", "2"]

    def test_export_is_byte_stable(self, tmp_path: Path) -> None:
        candidates = [_candidate(0, accuracy=0.9, parameters=100.0)]
        first = export_candidates_csv(candidates, tmp_path / "a.csv").read_bytes()
        second = export_candidates_csv(candidates, tmp_path / "b.csv").read_bytes()
        assert first == second

    def test_missing_metrics_become_empty_cells(self, tmp_path: Path) -> None:
        candidates = [
            _candidate(0, accuracy=0.9, parameters=100.0, latency=1.0),
            _candidate(1, accuracy=0.5, parameters=50.0),
        ]
        path = export_candidates_csv(candidates, tmp_path / "out.csv")
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        assert rows[1]["latency_median_ms"] == ""

    def test_metric_columns_are_sorted(self) -> None:
        candidates = [
            _candidate(0, accuracy=0.9, parameters=100.0, latency=1.0),
            _candidate(1, accuracy=0.5, parameters=50.0),
        ]
        assert collect_metric_columns(candidates) == sorted(collect_metric_columns(candidates))

    def test_empty_population_still_writes_a_header(self, tmp_path: Path) -> None:
        path = export_candidates_csv([], tmp_path / "empty.csv")
        assert path.read_text(encoding="utf-8").strip() == ",".join(BASE_COLUMNS)

    def test_arbitrary_rows_can_be_exported(self, tmp_path: Path) -> None:
        path = export_rows_csv([{"a": 1, "b": "=cmd"}], ["a", "b"], tmp_path / "rows.csv")
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        assert rows[0]["b"] == "'=cmd"


class TestJsonExport:
    def test_writes_readable_json(self, tmp_path: Path) -> None:
        path = export_json({"a": [1, 2], "b": {"c": 3}}, tmp_path / "out.json")
        assert json.loads(path.read_text(encoding="utf-8")) == {"a": [1, 2], "b": {"c": 3}}

    def test_non_serialisable_payload_is_reported(self, tmp_path: Path) -> None:
        from nas_engine.exceptions import ReportingError

        with pytest.raises(ReportingError, match="could not write JSON export"):
            export_json({"a": object()}, tmp_path / "out.json")


class TestPlots:
    def test_accuracy_versus_parameters_is_drawn(self, tmp_path: Path) -> None:
        candidates = [
            _candidate(0, accuracy=0.9, parameters=1000.0, front=True),
            _candidate(1, accuracy=0.5, parameters=100.0, front=True),
            _candidate(2, accuracy=0.4, parameters=5000.0),
        ]
        path = plot_accuracy_vs_parameters(candidates, tmp_path / "plot.png")
        assert path is not None
        assert path.stat().st_size > 1000

    def test_accuracy_versus_latency_needs_latency(self, tmp_path: Path) -> None:
        without = [_candidate(index, accuracy=0.5, parameters=100.0) for index in range(3)]
        assert plot_accuracy_vs_latency(without, tmp_path / "a.png") is None
        with_latency = [
            _candidate(index, accuracy=0.5, parameters=100.0, latency=float(index + 1))
            for index in range(3)
        ]
        assert plot_accuracy_vs_latency(with_latency, tmp_path / "b.png") is not None

    def test_single_candidate_is_not_plotted(self, tmp_path: Path) -> None:
        assert (
            plot_accuracy_vs_parameters(
                [_candidate(0, accuracy=0.9, parameters=100.0)], tmp_path / "plot.png"
            )
            is None
        )

    def test_progress_plot_needs_two_points(self, tmp_path: Path) -> None:
        assert plot_search_progress([(0, 0.5)], tmp_path / "p.png") is None
        assert plot_search_progress([(0, 0.5), (1, 0.7), (2, 0.6)], tmp_path / "p.png") is not None

    def test_population_plot_needs_two_generations(self, tmp_path: Path) -> None:
        assert plot_population_statistics([(0, 1.0, 0.5, 0.0)], tmp_path / "g.png") is None
        assert (
            plot_population_statistics([(0, 1.0, 0.5, 0.0), (1, 1.2, 0.7, 0.2)], tmp_path / "g.png")
            is not None
        )

    def test_generate_plots_uses_deterministic_filenames(self, tmp_path: Path) -> None:
        candidates = [
            _candidate(
                index,
                accuracy=0.1 * index,
                parameters=100.0 * (index + 1),
                latency=float(index + 1),
            )
            for index in range(3)
        ]
        result = generate_plots(candidates, [(0, 0.1), (1, 0.2)], tmp_path, prefix="searchid")
        assert set(result.paths) == {
            "accuracy_vs_parameters",
            "accuracy_vs_latency",
            "search_progress",
        }
        for name, path in result.paths.items():
            assert path.name == f"searchid_{name}.png"

    def test_regeneration_overwrites_in_place(self, tmp_path: Path) -> None:
        candidates = [
            _candidate(index, accuracy=0.1 * index, parameters=100.0 * (index + 1))
            for index in range(3)
        ]
        generate_plots(candidates, [(0, 0.1), (1, 0.2)], tmp_path, prefix="s")
        generate_plots(candidates, [(0, 0.1), (1, 0.2)], tmp_path, prefix="s")
        assert len(list(tmp_path.glob("s_*.png"))) == 2

    def test_skipped_plots_explain_themselves(self, tmp_path: Path) -> None:
        result = generate_plots([], [], tmp_path, prefix="s")
        assert result.paths == {}
        assert set(result.skipped) == {
            "accuracy_vs_parameters",
            "accuracy_vs_latency",
            "search_progress",
        }
        assert all(reason for reason in result.skipped.values())

    def test_population_plot_is_included_when_requested(self, tmp_path: Path) -> None:
        result = generate_plots(
            [], [], tmp_path, prefix="s", population=[(0, 1.0, 0.5, 0.0), (1, 1.1, 0.6, 0.1)]
        )
        assert "population_statistics" in result.paths

    def test_result_serialises(self, tmp_path: Path) -> None:
        payload = generate_plots([], [], tmp_path, prefix="s").to_dict()
        assert set(payload) == {"paths", "skipped"}


class TestKnownLimitations:
    def test_the_standing_caveats_are_present(self) -> None:
        text = " ".join(KNOWN_LIMITATIONS).lower()
        assert "globally optimal" in text
        assert "noisy estimate" in text
        assert "test split" in text
        assert "not comparable across machines" in text
        assert "cost of the search" in text
        assert "search-space design" in text

    def test_there_are_enough_of_them_to_be_useful(self) -> None:
        assert len(KNOWN_LIMITATIONS) >= 6
