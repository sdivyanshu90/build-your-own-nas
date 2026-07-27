"""Tests for the public API surface and the project's hard constraints.

Two things are enforced here:

1. **The public API is stable and complete.** Everything ``nas_engine.__all__`` advertises
   must be importable, and the three-line quick start in the README must work.
2. **The test suite never reaches outside the machine.** No network, no GPU requirement,
   no CIFAR-10 download, no cloud credentials. This is checked mechanically rather than by
   convention, because a single accidental download makes CI flaky for everyone.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

import pytest

import nas_engine

pytestmark = pytest.mark.unit


class TestPublicApi:
    def test_every_exported_name_exists(self) -> None:
        missing = [name for name in nas_engine.__all__ if not hasattr(nas_engine, name)]
        assert not missing, f"__all__ advertises missing names: {missing}"

    def test_exports_are_sorted(self) -> None:
        assert list(nas_engine.__all__) == sorted(nas_engine.__all__)

    def test_the_headline_entry_points_are_exported(self) -> None:
        for name in ("SearchConfig", "SearchEngine", "SearchResult", "ArchitectureSpec"):
            assert name in nas_engine.__all__

    def test_the_extension_points_are_exported(self) -> None:
        for name in (
            "SearchStrategy",
            "register_strategy",
            "DatasetProvider",
            "register_provider",
            "Objective",
            "MetricConstraint",
        ):
            assert name in nas_engine.__all__

    def test_the_error_taxonomy_root_is_exported(self) -> None:
        assert "NasEngineError" in nas_engine.__all__

    def test_a_version_is_reported(self) -> None:
        assert isinstance(nas_engine.__version__, str)
        assert nas_engine.__version__

    def test_the_documented_quick_start_works(self, tmp_path: Path) -> None:
        from nas_engine import SearchConfig, SearchEngine

        config = SearchConfig.from_mapping(
            {
                "project": {"name": "quickstart", "output_dir": str(tmp_path)},
                "dataset": {
                    "provider": "synthetic",
                    "batch_size": 32,
                    "options": {
                        "num_classes": 3,
                        "input_size": 16,
                        "train_samples": 48,
                        "validation_samples": 24,
                        "test_samples": 24,
                    },
                },
                "search_space": {"preset": "tiny_cnn"},
                "budget": {"max_evaluations": 2, "epochs": 1},
                "evaluation": {"measure_latency": False},
                "logging": {"level": "ERROR"},
                "hardware": {"device": "cpu"},
            }
        )
        engine = SearchEngine(config, configure_process=False)
        try:
            result = engine.run()
            assert result.best is not None
            assert result.summary()
        finally:
            engine.close()


class TestModuleGraph:
    def test_every_module_imports_cleanly(self) -> None:
        package = nas_engine
        failures: list[str] = []
        for info in pkgutil.walk_packages(package.__path__, prefix="nas_engine."):
            try:
                importlib.import_module(info.name)
            except Exception as exc:
                failures.append(f"{info.name}: {type(exc).__name__}: {exc}")
        assert not failures, "modules failed to import:\n" + "\n".join(failures)

    def test_the_domain_does_not_import_the_orchestrator(self) -> None:
        # Import direction is a load-bearing design property: architectures, search spaces,
        # models, and objectives must stay usable without the engine.
        import ast

        source_root = Path(nas_engine.__file__).parent
        offenders: list[str] = []
        leaf_packages = ("architectures", "search_space", "models", "objectives")
        for package in leaf_packages:
            for path in (source_root / package).rglob("*.py"):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    module = None
                    if isinstance(node, ast.ImportFrom) and node.module:
                        module = node.module
                    elif isinstance(node, ast.Import):
                        module = node.names[0].name
                    if module and module.startswith(
                        (
                            "nas_engine.orchestration",
                            "nas_engine.persistence",
                            "nas_engine.cli",
                            "nas_engine.reporting",
                        )
                    ):
                        offenders.append(f"{path.name} imports {module}")
        assert not offenders, "leaf packages must not depend on higher layers: " + str(offenders)

    def test_no_module_uses_a_wildcard_import(self) -> None:
        import ast

        source_root = Path(nas_engine.__file__).parent
        offenders: list[str] = []
        for path in source_root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and any(
                    alias.name == "*" for alias in node.names
                ):
                    offenders.append(str(path.relative_to(source_root)))
        assert not offenders, f"wildcard imports found in {offenders}"


class TestNoExternalDependencies:
    def test_the_default_dataset_needs_no_network(self) -> None:
        from nas_engine.datasets.synthetic import SyntheticDatasetProvider

        bundle = SyntheticDatasetProvider(
            num_classes=2,
            input_size=8,
            train_samples=8,
            validation_samples=8,
            test_samples=8,
        ).build()
        assert bundle.split_sizes()["train"] == 8

    def test_cifar10_never_downloads_without_permission(self, tmp_path: Path) -> None:
        from nas_engine.datasets.cifar10 import Cifar10Provider
        from nas_engine.exceptions import DatasetError

        provider = Cifar10Provider(root=tmp_path / "cifar", download=False)
        with pytest.raises(DatasetError, match="download=false"):
            provider.build()

    def test_no_test_module_imports_a_network_client(self) -> None:
        import ast

        banned = {"requests", "urllib.request", "httpx", "boto3", "google.cloud", "socket"}
        offenders: list[str] = []
        test_root = Path(__file__).parent.parent
        for path in test_root.rglob("test_*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for name in names:
                    if any(name == item or name.startswith(f"{item}.") for item in banned):
                        offenders.append(f"{path.name} imports {name}")
        assert not offenders, f"tests must not reach the network: {offenders}"

    def test_no_test_moves_tensors_to_a_gpu(self) -> None:
        # An AST check rather than a text search: a test that *asserts CUDA is rejected*
        # legitimately contains the string "cuda", but no test may actually call `.cuda()`.
        import ast

        offenders: list[str] = []
        for path in Path(__file__).parent.parent.rglob("test_*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "cuda"
                ):
                    offenders.append(f"{path.name}:{node.lineno}")
        assert not offenders, f"tests must not require a GPU: {offenders}"

    def test_the_default_test_device_is_cpu(self, tmp_path: Path) -> None:
        from tests.conftest import build_smoke_config

        config = build_smoke_config(tmp_path)
        assert config.hardware.device == "cpu"
        assert config.hardware.resolve_device().type == "cpu"

    def test_no_test_reads_the_process_environment_directly(self) -> None:
        # Reading the ambient environment makes a test depend on the developer's shell.
        # `monkeypatch.setenv` and explicit mappings are the sanctioned alternatives; the
        # only exception is conftest, which strips NAS_ENGINE variables for isolation.
        import ast

        offenders: list[str] = []
        for path in Path(__file__).parent.parent.rglob("test_*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Attribute)
                    and node.attr in {"environ", "getenv"}
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "os"
                ):
                    offenders.append(f"{path.name}:{node.lineno}")
        assert not offenders, f"tests must not read os.environ directly: {offenders}"
