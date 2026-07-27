"""Unit tests for the configuration system.

Covers: field validation and actionable errors, cross-section consistency checks, the
four-layer precedence chain, deep merging, environment and command-line parsing, YAML
safety, path resolution, configuration hashing, and version compatibility.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nas_engine.config.loader import (
    ENV_PREFIX,
    MAX_CONFIG_BYTES,
    assign_path,
    check_config_compatibility,
    deep_merge,
    dump_yaml,
    load_config,
    parse_environment,
    parse_overrides,
    parse_scalar,
    read_yaml,
)
from nas_engine.config.models import (
    CONFIG_VERSION,
    ConcurrencyConfig,
    DatasetConfig,
    HardwareConfig,
    SearchConfig,
    SearchSpaceConfig,
)
from nas_engine.exceptions import ConfigurationError, ConfigVersionError

pytestmark = pytest.mark.unit


class TestDefaults:
    def test_empty_configuration_is_valid(self) -> None:
        config = SearchConfig()
        assert config.version == CONFIG_VERSION
        assert config.algorithm.name == "random_search"

    def test_describe_covers_the_key_sections(self) -> None:
        text = SearchConfig().describe()
        for label in ("project", "dataset", "algorithm", "budget", "seed", "config hash"):
            assert label in text

    def test_hash_is_stable(self) -> None:
        assert SearchConfig().config_hash() == SearchConfig().config_hash()

    def test_hash_changes_with_content(self) -> None:
        first = SearchConfig()
        second = SearchConfig.from_mapping({"budget": {"max_evaluations": 99}})
        assert first.config_hash() != second.config_hash()

    def test_serialises_paths_as_strings(self) -> None:
        payload = SearchConfig().to_dict()
        assert isinstance(payload["project"]["output_dir"], str)


class TestValidation:
    def test_unknown_fields_are_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="Extra inputs are not permitted"):
            SearchConfig.from_mapping({"nonsense": 1})

    def test_error_names_the_field_and_value(self) -> None:
        with pytest.raises(ConfigurationError) as excinfo:
            SearchConfig.from_mapping({"budget": {"max_evaluations": -5}})
        message = str(excinfo.value)
        assert "budget.max_evaluations" in message
        assert "received -5" in message

    def test_all_problems_are_reported_together(self) -> None:
        with pytest.raises(ConfigurationError) as excinfo:
            SearchConfig.from_mapping({"budget": {"max_evaluations": 0, "epochs": 0}, "typo": 1})
        problems = excinfo.value.details["problems"]
        assert len(problems) >= 3

    def test_future_version_is_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="newer than the supported version"):
            SearchConfig.from_mapping({"version": CONFIG_VERSION + 1})

    def test_mixed_precision_on_cpu_is_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="mixed_precision requires a CUDA"):
            SearchConfig.from_mapping(
                {"training": {"mixed_precision": True}, "hardware": {"device": "cpu"}}
            )

    def test_checkpoints_without_weights_are_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="save_weights"):
            SearchConfig.from_mapping(
                {"evaluation": {"save_training_checkpoints": True, "save_weights": False}}
            )

    def test_workers_without_multiprocessing_are_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="no effect with"):
            SearchConfig.from_mapping({"concurrency": {"mode": "sequential", "workers": 4}})

    def test_persistent_workers_need_workers(self) -> None:
        with pytest.raises(ConfigurationError, match="persistent_workers requires"):
            SearchConfig.from_mapping({"dataset": {"persistent_workers": True, "num_workers": 0}})

    def test_unknown_preset_is_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="not a known preset"):
            SearchConfig.from_mapping({"search_space": {"preset": "nope"}})

    def test_section_models_raise_pydantic_errors_directly(self) -> None:
        # Direct construction bypasses the loader's error translation; only the loader
        # promises the friendly ConfigurationError wrapper.
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ConcurrencyConfig(mode="sequential", workers=4)
        with pytest.raises(ValidationError):
            DatasetConfig(persistent_workers=True, num_workers=0)
        with pytest.raises(ValidationError):
            SearchSpaceConfig(preset="nope")

    def test_bad_search_space_override_is_reported(self) -> None:
        config = SearchSpaceConfig(preset="tiny_cnn", overrides={"num_stages": (0,)})
        with pytest.raises(ConfigurationError, match="invalid search space"):
            config.build()

    def test_valid_search_space_override_is_applied(self) -> None:
        config = SearchSpaceConfig(preset="tiny_cnn", overrides={"stage_channels": (32, 64)})
        assert config.build().stage_channels == (32, 64)


class TestDeviceResolution:
    def test_auto_resolves_to_something_usable(self) -> None:
        assert HardwareConfig(device="auto").resolve_device().type in {"cpu", "cuda", "mps"}

    def test_cpu_is_always_available(self) -> None:
        assert HardwareConfig(device="cpu").resolve_device().type == "cpu"

    def test_unavailable_accelerator_is_an_error_not_a_fallback(self) -> None:
        import torch

        if torch.cuda.is_available():  # pragma: no cover - depends on the host
            pytest.skip("this host has CUDA, so the failure path cannot be exercised")
        with pytest.raises(ConfigurationError, match="CUDA is not available"):
            HardwareConfig(device="cuda").resolve_device()

    def test_nonsense_device_is_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="not a valid device"):
            HardwareConfig(device="quantum").resolve_device()


class TestPaths:
    def test_relative_paths_resolve_under_the_output_directory(self, tmp_path: Path) -> None:
        config = SearchConfig.from_mapping({"project": {"output_dir": str(tmp_path)}})
        assert config.artifact_dir == tmp_path / "candidates"
        assert config.report_dir == tmp_path / "reports"
        assert str(tmp_path) in config.database_url

    def test_absolute_paths_are_respected(self, tmp_path: Path) -> None:
        elsewhere = tmp_path / "elsewhere"
        config = SearchConfig.from_mapping(
            {
                "project": {"output_dir": str(tmp_path)},
                "persistence": {"artifact_dir": str(elsewhere)},
            }
        )
        assert config.artifact_dir == elsewhere

    def test_explicit_database_url_wins(self, tmp_path: Path) -> None:
        config = SearchConfig.from_mapping(
            {
                "project": {"output_dir": str(tmp_path)},
                "persistence": {"database_url": "sqlite+pysqlite:///:memory:"},
            }
        )
        assert config.database_url == "sqlite+pysqlite:///:memory:"

    def test_log_file_resolves_under_the_output_directory(self, tmp_path: Path) -> None:
        config = SearchConfig.from_mapping(
            {"project": {"output_dir": str(tmp_path)}, "logging": {"file": "run.log"}}
        )
        assert config.log_file == tmp_path / "run.log"

    def test_log_file_defaults_to_none(self) -> None:
        assert SearchConfig().log_file is None


class TestMerging:
    def test_deep_merge_preserves_siblings(self) -> None:
        base = {"a": {"b": 1, "c": 2}}
        merged = deep_merge(base, {"a": {"b": 9}})
        assert merged == {"a": {"b": 9, "c": 2}}

    def test_deep_merge_does_not_mutate_its_inputs(self) -> None:
        base = {"a": {"b": 1}}
        deep_merge(base, {"a": {"b": 2}})
        assert base == {"a": {"b": 1}}

    def test_lists_are_replaced_not_concatenated(self) -> None:
        assert deep_merge({"a": [1, 2]}, {"a": [3]}) == {"a": [3]}

    def test_assign_path_creates_intermediate_sections(self) -> None:
        target: dict[str, object] = {}
        assign_path(target, ["a", "b", "c"], 1)
        assert target == {"a": {"b": {"c": 1}}}

    def test_assign_path_rejects_scalar_traversal(self) -> None:
        with pytest.raises(ConfigurationError, match="already a scalar value"):
            assign_path({"a": 1}, ["a", "b"], 2)

    def test_assign_path_rejects_an_empty_path(self) -> None:
        with pytest.raises(ConfigurationError, match="empty key path"):
            assign_path({}, [], 1)


class TestScalarParsing:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("3", 3),
            ("3.5", 3.5),
            ("true", True),
            ("false", False),
            ("null", None),
            ("[1, 2]", [1, 2]),
            ("hello", "hello"),
        ],
    )
    def test_yaml_scalars_are_parsed(self, text: str, expected: object) -> None:
        assert parse_scalar(text) == expected

    def test_unparseable_text_stays_a_string(self) -> None:
        assert parse_scalar("{unbalanced") == "{unbalanced"


class TestEnvironmentOverrides:
    def test_nested_variables_are_parsed(self) -> None:
        overrides = parse_environment({f"{ENV_PREFIX}TRAINING__OPTIMIZER__LEARNING_RATE": "0.01"})
        assert overrides == {"training": {"optimizer": {"learning_rate": 0.01}}}

    def test_unrelated_variables_are_ignored(self) -> None:
        assert parse_environment({"PATH": "/usr/bin"}) == {}

    def test_bare_prefix_is_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="no field path"):
            parse_environment({ENV_PREFIX: "x"})

    def test_underscores_inside_field_names_are_preserved(self) -> None:
        overrides = parse_environment({f"{ENV_PREFIX}BUDGET__MAX_EVALUATIONS": "7"})
        assert overrides == {"budget": {"max_evaluations": 7}}


class TestCommandLineOverrides:
    def test_dotted_assignments_are_parsed(self) -> None:
        assert parse_overrides(["budget.max_evaluations=7"]) == {"budget": {"max_evaluations": 7}}

    def test_multiple_assignments_merge(self) -> None:
        overrides = parse_overrides(["budget.epochs=2", "budget.max_evaluations=4"])
        assert overrides == {"budget": {"epochs": 2, "max_evaluations": 4}}

    def test_missing_equals_is_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="not of the form"):
            parse_overrides(["budget.epochs"])

    def test_empty_key_is_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="empty key"):
            parse_overrides(["=3"])

    def test_values_containing_equals_are_preserved(self) -> None:
        overrides = parse_overrides(["persistence.database_url=sqlite:///a=b"])
        assert overrides["persistence"]["database_url"] == "sqlite:///a=b"


class TestYamlLoading:
    def test_reads_a_mapping(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text("budget:\n  epochs: 2\n", encoding="utf-8")
        assert read_yaml(path) == {"budget": {"epochs": 2}}

    def test_empty_file_is_an_empty_mapping(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.yaml"
        path.write_text("", encoding="utf-8")
        assert read_yaml(path) == {}

    def test_missing_file_is_reported(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match="configuration file not found"):
            read_yaml(tmp_path / "absent.yaml")

    def test_non_mapping_document_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "list.yaml"
        path.write_text("- 1\n- 2\n", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="must contain a YAML mapping"):
            read_yaml(path)

    def test_malformed_yaml_is_reported(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("a: [1, 2\n", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="not valid YAML"):
            read_yaml(path)

    def test_oversized_file_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "big.yaml"
        path.write_text("a: " + "x" * 200, encoding="utf-8")
        # The production limit is 1 MiB; shrinking it for one test exercises the guard
        # without writing a megabyte to disk. monkeypatch restores it afterwards.
        monkeypatch.setattr("nas_engine.config.loader.MAX_CONFIG_BYTES", 10)
        with pytest.raises(ConfigurationError, match="byte limit"):
            read_yaml(path)

    def test_python_object_tags_are_refused(self, tmp_path: Path) -> None:
        # `yaml.safe_load` refuses `!!python/...` tags. If this ever regressed, a
        # configuration file would become executable code.
        path = tmp_path / "evil.yaml"
        path.write_text("a: !!python/object/apply:os.system ['echo pwned']\n", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="not valid YAML"):
            read_yaml(path)

    def test_round_trips_through_yaml(self, tmp_path: Path) -> None:
        original = SearchConfig.from_mapping({"budget": {"max_evaluations": 11}})
        path = tmp_path / "out.yaml"
        dump_yaml(original, path)
        restored = SearchConfig.from_yaml(path, use_environment=False)
        assert restored.config_hash() == original.config_hash()

    def test_to_yaml_writes_the_file(self, tmp_path: Path) -> None:
        path = tmp_path / "written.yaml"
        text = SearchConfig().to_yaml(path)
        assert path.read_text(encoding="utf-8") == text


class TestPrecedence:
    def test_defaults_apply_with_no_sources(self) -> None:
        assert load_config(None, use_environment=False).budget.max_evaluations == 12

    def test_file_overrides_defaults(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text("budget:\n  max_evaluations: 5\n", encoding="utf-8")
        assert load_config(path, use_environment=False).budget.max_evaluations == 5

    def test_environment_overrides_the_file(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text("budget:\n  max_evaluations: 5\n", encoding="utf-8")
        config = load_config(path, environ={f"{ENV_PREFIX}BUDGET__MAX_EVALUATIONS": "9"})
        assert config.budget.max_evaluations == 9

    def test_command_line_overrides_everything(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text("budget:\n  max_evaluations: 5\n", encoding="utf-8")
        config = load_config(
            path,
            environ={f"{ENV_PREFIX}BUDGET__MAX_EVALUATIONS": "9"},
            overrides=["budget.max_evaluations=13"],
        )
        assert config.budget.max_evaluations == 13

    def test_overrides_do_not_erase_sibling_fields(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text(
            "training:\n  optimizer:\n    learning_rate: 0.5\n    weight_decay: 0.25\n",
            encoding="utf-8",
        )
        config = load_config(
            path, use_environment=False, overrides=["training.optimizer.learning_rate=0.1"]
        )
        assert config.training.optimizer.learning_rate == 0.1
        assert config.training.optimizer.weight_decay == 0.25

    def test_error_message_names_the_sources(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text("budget:\n  max_evaluations: 0\n", encoding="utf-8")
        with pytest.raises(ConfigurationError) as excinfo:
            load_config(path, use_environment=False)
        assert "defaults <" in str(excinfo.value)


class TestCompatibility:
    def test_identical_configurations_report_no_differences(self) -> None:
        config = SearchConfig()
        assert check_config_compatibility(config.to_dict(), config) == []

    def test_algorithm_change_is_flagged_first(self) -> None:
        stored = SearchConfig().to_dict()
        current = SearchConfig.from_mapping({"algorithm": {"name": "regularized_evolution"}})
        differences = check_config_compatibility(stored, current)
        assert differences
        assert "search strategy" in differences[0]

    def test_incidental_change_is_still_reported(self) -> None:
        stored = SearchConfig().to_dict()
        current = SearchConfig.from_mapping({"logging": {"level": "DEBUG"}})
        differences = check_config_compatibility(stored, current)
        assert any("logging" in difference for difference in differences)

    def test_future_stored_version_is_rejected(self) -> None:
        stored = SearchConfig().to_dict()
        stored["version"] = CONFIG_VERSION + 1
        with pytest.raises(ConfigVersionError, match="upgrade nas-engine"):
            check_config_compatibility(stored, SearchConfig())

    def test_corrupt_stored_version_is_rejected(self) -> None:
        stored = SearchConfig().to_dict()
        stored["version"] = 0
        with pytest.raises(ConfigVersionError, match="not a valid configuration version"):
            check_config_compatibility(stored, SearchConfig())


class TestConversion:
    def test_training_settings_are_produced(self) -> None:
        settings = SearchConfig().training.build(epochs=7)
        assert settings.epochs == 7
        assert settings.optimizer.learning_rate > 0

    def test_evaluation_settings_are_produced(self) -> None:
        settings = SearchConfig().evaluation.build(max_seconds=30.0)
        assert settings.max_evaluation_seconds == 30.0

    def test_loader_settings_are_produced(self) -> None:
        settings = SearchConfig().dataset.build_loader_settings()
        assert settings.batch_size == 64

    def test_budget_is_produced(self) -> None:
        budget = SearchConfig().budget.build_budget()
        assert budget.epochs == 3
        assert budget.rung == 0

    def test_objectives_and_constraints_are_produced(self) -> None:
        config = SearchConfig.from_mapping(
            {
                "objectives": {
                    "objectives": [{"metric": "validation_accuracy", "direction": "maximize"}],
                    "constraints": [
                        {
                            "metric": "trainable_parameters",
                            "operator": "le",
                            "threshold": 1000,
                        }
                    ],
                }
            }
        )
        assert len(config.objectives.build_objectives().objectives) == 1
        assert len(config.objectives.build_constraints().constraints) == 1

    def test_effective_in_flight_respects_the_mode(self) -> None:
        assert ConcurrencyConfig(mode="sequential").effective_in_flight == 1
        assert ConcurrencyConfig(mode="multiprocessing", workers=3).effective_in_flight == 3
        assert (
            ConcurrencyConfig(
                mode="multiprocessing", workers=3, max_in_flight=2
            ).effective_in_flight
            == 2
        )


def test_production_size_limit_is_documented() -> None:
    """The production guard exists and is a sane size."""
    assert MAX_CONFIG_BYTES == 1024 * 1024
