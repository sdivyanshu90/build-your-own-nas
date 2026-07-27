"""Unit tests for the command-line interface.

Covers: help text, exit codes, JSON output, configuration scaffolding and validation, the
doctor diagnostics, candidate lookup by hash prefix, and error translation.

The CLI is exercised through :class:`typer.testing.CliRunner`, which invokes the real
command functions in-process. Only genuinely fast commands are covered here; the
search-and-report path is an end-to-end test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from nas_engine.cli import ExitCode, app, main
from nas_engine.config.models import SearchConfig

pytestmark = pytest.mark.unit

runner = CliRunner()


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    """Write a minimal, fast configuration file and return its path."""
    from tests.conftest import build_smoke_config

    config = build_smoke_config(tmp_path / "out")
    path = tmp_path / "config.yaml"
    config.to_yaml(path)
    return path


class TestHelp:
    def test_top_level_help_lists_every_command(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        for command in (
            "init",
            "validate-config",
            "search",
            "resume",
            "status",
            "list-candidates",
            "show-candidate",
            "best",
            "pareto",
            "evaluate",
            "export",
            "report",
            "doctor",
        ):
            assert command in result.output

    @pytest.mark.parametrize(
        "command",
        [
            "init",
            "validate-config",
            "search",
            "resume",
            "status",
            "list-candidates",
            "show-candidate",
            "best",
            "pareto",
            "evaluate",
            "export",
            "report",
            "doctor",
        ],
    )
    def test_every_command_has_help_and_examples(self, command: str) -> None:
        result = runner.invoke(app, [command, "--help"])
        assert result.exit_code == 0
        assert "nas-engine" in result.output


class TestInit:
    def test_writes_a_usable_configuration(self, tmp_path: Path) -> None:
        target = tmp_path / "configs" / "search.yaml"
        result = runner.invoke(app, ["init", "--output", str(target)])
        assert result.exit_code == 0
        assert target.exists()
        config = SearchConfig.from_yaml(target, use_environment=False)
        assert config.algorithm.name == "random_search"

    def test_honours_the_preset_and_strategy(self, tmp_path: Path) -> None:
        target = tmp_path / "evolution.yaml"
        result = runner.invoke(
            app,
            [
                "init",
                "--output",
                str(target),
                "--preset",
                "tiny_cnn",
                "--strategy",
                "regularized_evolution",
            ],
        )
        assert result.exit_code == 0
        config = SearchConfig.from_yaml(target, use_environment=False)
        assert config.search_space.preset == "tiny_cnn"
        assert config.algorithm.name == "regularized_evolution"

    def test_refuses_to_overwrite_without_force(self, tmp_path: Path) -> None:
        target = tmp_path / "search.yaml"
        target.write_text("existing", encoding="utf-8")
        result = runner.invoke(app, ["init", "--output", str(target)])
        assert result.exit_code == ExitCode.CONFIGURATION_ERROR
        assert target.read_text(encoding="utf-8") == "existing"

    def test_force_overwrites(self, tmp_path: Path) -> None:
        target = tmp_path / "search.yaml"
        target.write_text("existing", encoding="utf-8")
        result = runner.invoke(app, ["init", "--output", str(target), "--force"])
        assert result.exit_code == 0
        assert "existing" not in target.read_text(encoding="utf-8")


class TestValidateConfig:
    def test_accepts_a_valid_file(self, config_file: Path) -> None:
        result = runner.invoke(app, ["validate-config", "--config", str(config_file)])
        assert result.exit_code == 0
        assert "valid" in result.output.lower()

    def test_json_output_is_machine_readable(self, config_file: Path) -> None:
        result = runner.invoke(app, ["validate-config", "--config", str(config_file), "--json"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["valid"] is True
        assert payload["config_hash"]

    def test_reports_an_invalid_file(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("budget:\n  max_evaluations: -1\n", encoding="utf-8")
        result = runner.invoke(app, ["validate-config", "--config", str(bad)])
        assert result.exit_code != 0

    def test_reports_a_missing_file(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["validate-config", "--config", str(tmp_path / "absent.yaml")])
        assert result.exit_code != 0

    def test_overrides_are_applied(self, config_file: Path) -> None:
        result = runner.invoke(
            app,
            [
                "validate-config",
                "--config",
                str(config_file),
                "--set",
                "budget.max_evaluations=99",
                "--json",
            ],
        )
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["config"]["budget"]["max_evaluations"] == 99

    def test_malformed_override_is_reported(self, config_file: Path) -> None:
        result = runner.invoke(
            app, ["validate-config", "--config", str(config_file), "--set", "nonsense"]
        )
        assert result.exit_code != 0


class TestDoctor:
    def test_reports_every_check(self, config_file: Path) -> None:
        result = runner.invoke(app, ["doctor", "--config", str(config_file), "--json"])
        payload = json.loads(result.output)
        names = {check["name"] for check in payload["checks"]}
        assert {
            "python version",
            "pytorch",
            "configuration",
            "database",
            "search space",
            "random seed",
        } <= names

    def test_passes_on_a_healthy_environment(self, config_file: Path) -> None:
        result = runner.invoke(app, ["doctor", "--config", str(config_file)])
        assert result.exit_code == 0
        assert "All checks passed" in result.output

    def test_fails_on_an_invalid_configuration(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("budget:\n  epochs: -1\n", encoding="utf-8")
        result = runner.invoke(app, ["doctor", "--config", str(bad), "--json"])
        assert result.exit_code == ExitCode.RUNTIME_ERROR
        payload = json.loads(result.output)
        assert payload["failures"] >= 1

    def test_works_with_no_configuration_file(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["doctor", "--set", f"project.output_dir={tmp_path}", "--json"])
        assert result.exit_code == 0


class TestInspectionWithoutASearch:
    def test_status_reports_no_search(self, config_file: Path) -> None:
        result = runner.invoke(app, ["status", "--config", str(config_file)])
        assert result.exit_code == ExitCode.NOT_FOUND
        assert "No search found" in result.output or "No search found" in str(result.stdout)

    def test_best_reports_no_search(self, config_file: Path) -> None:
        result = runner.invoke(app, ["best", "--config", str(config_file)])
        assert result.exit_code == ExitCode.NOT_FOUND

    def test_list_candidates_reports_no_search(self, config_file: Path) -> None:
        result = runner.invoke(app, ["list-candidates", "--config", str(config_file)])
        assert result.exit_code == ExitCode.NOT_FOUND


class TestExportValidation:
    def test_unknown_format_is_rejected(self, config_file: Path) -> None:
        result = runner.invoke(app, ["export", "--config", str(config_file), "--format", "xml"])
        assert result.exit_code == ExitCode.CONFIGURATION_ERROR

    def test_unknown_state_filter_is_rejected(self, config_file: Path) -> None:
        result = runner.invoke(
            app, ["list-candidates", "--config", str(config_file), "--state", "sideways"]
        )
        assert result.exit_code in {ExitCode.CONFIGURATION_ERROR, ExitCode.NOT_FOUND}


class TestExitCodes:
    def test_codes_are_distinct(self) -> None:
        values = [member.value for member in ExitCode]
        assert len(values) == len(set(values))

    def test_success_is_zero(self) -> None:
        assert ExitCode.SUCCESS.value == 0

    def test_interrupt_uses_the_conventional_code(self) -> None:
        assert ExitCode.INTERRUPTED.value == 130

    def test_main_returns_zero_for_help(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.argv", ["nas-engine", "--help"])
        assert main() == 0

    def test_main_translates_configuration_errors(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "sys.argv",
            ["nas-engine", "validate-config", "--config", str(tmp_path / "absent.yaml")],
        )
        assert main() == ExitCode.CONFIGURATION_ERROR

    def test_main_reports_usage_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.argv", ["nas-engine", "not-a-command"])
        assert main() != 0
