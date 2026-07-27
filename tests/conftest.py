"""Shared pytest fixtures.

Principles for this suite
-------------------------
* **No network, no GPU, no external services.** Every fixture is local and synthetic.
  ``tests/unit/test_no_external_dependencies.py`` enforces this mechanically.
* **Deterministic.** Every seed is explicit. Nothing depends on wall-clock time, on
  dictionary iteration order, or on the developer's environment.
* **Fast by default.** The session-scoped dataset is deliberately tiny. Anything that
  genuinely takes seconds is marked ``slow`` and excluded from the default run.
* **Environment-isolated.** ``NAS_ENGINE__*`` variables in a developer's shell would
  silently change configuration under test, so they are stripped for every test.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from nas_engine.architectures.spec import (
    ArchitectureSpec,
    BlockSpec,
    HeadSpec,
    StageSpec,
    StemSpec,
)
from nas_engine.architectures.types import (
    ActivationType,
    NormalizationType,
    OperationType,
    PoolingType,
)
from nas_engine.config.models import SearchConfig
from nas_engine.datasets.base import DatasetBundle
from nas_engine.datasets.synthetic import SyntheticDatasetProvider
from nas_engine.persistence.database import Database
from nas_engine.persistence.migrations import ensure_schema
from nas_engine.persistence.repository import SearchRepository
from nas_engine.search_space.presets import (
    default_cnn_space,
    micro_cnn_space,
    tiny_cnn_space,
)
from nas_engine.search_space.sampler import ArchitectureSampler
from nas_engine.search_space.space import SearchSpace

#: Seed used by every fixture that needs one, so failures are reproducible.
FIXTURE_SEED = 1234

#: Directory holding golden regression fixtures.
FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove ``NAS_ENGINE__*`` variables so a developer's shell cannot alter tests."""
    for name in list(os.environ):
        if name.startswith("NAS_ENGINE__"):
            monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="session")
def default_space() -> SearchSpace:
    """The demonstration CNN space at 32x32."""
    return default_cnn_space()


@pytest.fixture(scope="session")
def tiny_space() -> SearchSpace:
    """A small space that samples quickly and still contains real variety."""
    return tiny_cnn_space()


@pytest.fixture(scope="session")
def micro_space() -> SearchSpace:
    """A near-exhaustible space, used to trigger exhaustion behaviour."""
    return micro_cnn_space()


@pytest.fixture
def sampler(tiny_space: SearchSpace) -> ArchitectureSampler:
    """A seeded sampler over :func:`tiny_space`."""
    return ArchitectureSampler(tiny_space, seed=FIXTURE_SEED)


@pytest.fixture
def sample_spec(sampler: ArchitectureSampler) -> ArchitectureSpec:
    """One valid architecture drawn from the tiny space."""
    return sampler.sample()


@pytest.fixture
def manual_spec() -> ArchitectureSpec:
    """A hand-written architecture with known structure.

    Hand-written rather than sampled so that tests asserting exact parameter counts and
    hashes do not silently change when the sampler changes.
    """
    return ArchitectureSpec(
        input_channels=3,
        input_size=16,
        num_classes=4,
        stem=StemSpec(
            out_channels=8,
            kernel_size=3,
            stride=1,
            normalization=NormalizationType.BATCH,
            activation=ActivationType.RELU,
        ),
        stages=(
            StageSpec(
                blocks=(
                    BlockSpec(
                        operation=OperationType.CONV,
                        kernel_size=3,
                        out_channels=16,
                        stride=1,
                        normalization=NormalizationType.BATCH,
                        activation=ActivationType.RELU,
                    ),
                    BlockSpec(
                        operation=OperationType.DW_SEP_CONV,
                        kernel_size=3,
                        expansion_ratio=2.0,
                        out_channels=16,
                        stride=1,
                        use_residual=True,
                        normalization=NormalizationType.BATCH,
                        activation=ActivationType.RELU,
                    ),
                )
            ),
            StageSpec(
                blocks=(
                    BlockSpec(
                        operation=OperationType.MAX_POOL,
                        kernel_size=3,
                        out_channels=16,
                        stride=2,
                    ),
                )
            ),
        ),
        head=HeadSpec(pooling=PoolingType.AVG, hidden_units=0, dropout=0.0),
    )


@pytest.fixture(scope="session")
def synthetic_bundle() -> DatasetBundle:
    """A tiny synthetic dataset shared across the session.

    Session-scoped because generating it costs a few milliseconds that would otherwise be
    paid by every test. It is immutable in practice: the tensors are never written to.
    """
    return SyntheticDatasetProvider(
        num_classes=4,
        input_channels=3,
        input_size=16,
        train_samples=96,
        validation_samples=48,
        test_samples=48,
        seed=FIXTURE_SEED,
        noise_scale=0.4,
    ).build()


@pytest.fixture
def database() -> Iterator[Database]:
    """An in-memory database with the schema applied."""
    handle = Database.in_memory()
    ensure_schema(handle)
    try:
        yield handle
    finally:
        handle.dispose()


@pytest.fixture
def repository(database: Database) -> SearchRepository:
    """A repository over the in-memory database."""
    return SearchRepository(database)


@pytest.fixture
def file_database(tmp_path: Path) -> Iterator[Database]:
    """A file-backed database, for tests that need persistence across handles."""
    handle = Database.from_path(tmp_path / "nas.db")
    ensure_schema(handle)
    try:
        yield handle
    finally:
        handle.dispose()


def build_smoke_config(output_dir: Path, **overrides: Any) -> SearchConfig:
    """Build the minimal configuration used by integration and end-to-end tests.

    Args:
        output_dir: Directory for the database, artifacts, and reports.
        **overrides: Section mappings merged over the defaults.

    Returns:
        A validated configuration.
    """
    from nas_engine.config.loader import deep_merge

    base: dict[str, Any] = {
        "project": {"name": "test-search", "output_dir": str(output_dir)},
        "dataset": {
            "provider": "synthetic",
            "batch_size": 32,
            "num_workers": 0,
            "options": {
                "num_classes": 4,
                "input_channels": 3,
                "input_size": 16,
                "train_samples": 96,
                "validation_samples": 48,
                "test_samples": 48,
                "noise_scale": 0.4,
                "seed": FIXTURE_SEED,
            },
        },
        "search_space": {"preset": "tiny_cnn"},
        "algorithm": {"name": "random_search"},
        "budget": {"max_evaluations": 3, "epochs": 1},
        "training": {"optimizer": {"learning_rate": 0.005}, "topk": 2},
        "evaluation": {"measure_latency": False, "save_weights": True},
        "logging": {"level": "ERROR"},
        "hardware": {"device": "cpu"},
        "concurrency": {"mode": "sequential", "workers": 1},
        "reproducibility": {"seed": 7, "deterministic": False},
        "retry": {"max_retries": 0},
    }
    merged = deep_merge(base, overrides)
    return SearchConfig.from_mapping(merged)


@pytest.fixture
def smoke_config(tmp_path: Path) -> SearchConfig:
    """A minimal, fast, CPU-only search configuration."""
    return build_smoke_config(tmp_path)


@pytest.fixture
def config_factory(tmp_path: Path) -> Any:
    """Return a callable producing configurations rooted at ``tmp_path``."""

    def factory(**overrides: Any) -> SearchConfig:
        return build_smoke_config(tmp_path, **overrides)

    return factory
