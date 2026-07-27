"""Unit tests for the cross-cutting utilities.

Covers: stable hashing, canonical JSON, path validation and traversal defence, monotonic
timing, timezone-aware timestamps, seed derivation and isolation, determinism reporting,
and environment capture.
"""

from __future__ import annotations

import random
from datetime import timezone
from pathlib import Path

import pytest

from nas_engine.exceptions import NasEngineError, UnsafePathError
from nas_engine.utilities.determinism import configure_determinism
from nas_engine.utilities.environment import collect_environment
from nas_engine.utilities.hashing import stable_hash, stable_hash_bytes, stable_json_hash
from nas_engine.utilities.json_io import (
    canonical_json_dumps,
    read_json,
    read_json_bytes,
    write_json,
)
from nas_engine.utilities.paths import (
    ensure_directory,
    is_within,
    resolve_under_root,
    safe_filename,
)
from nas_engine.utilities.seeding import (
    SeedBundle,
    dataloader_worker_init,
    derive_seed,
    rng_state_from_json,
    rng_state_to_json,
    seed_everything,
    torch_generator,
)
from nas_engine.utilities.timing import Stopwatch, utc_now, utc_now_iso

pytestmark = pytest.mark.unit


class TestHashing:
    def test_digest_is_stable_across_calls(self) -> None:
        assert stable_hash("abc") == stable_hash("abc")

    def test_digest_length_matches_requested_size(self) -> None:
        assert len(stable_hash("abc", digest_bytes=8)) == 16
        assert len(stable_hash("abc", digest_bytes=16)) == 32

    def test_single_character_change_changes_digest(self) -> None:
        assert stable_hash("abc") != stable_hash("abd")

    def test_rejects_out_of_range_digest_size(self) -> None:
        with pytest.raises(ValueError, match="digest_bytes must be in"):
            stable_hash_bytes(b"x", digest_bytes=0)
        with pytest.raises(ValueError, match="digest_bytes must be in"):
            stable_hash_bytes(b"x", digest_bytes=65)

    def test_json_hash_ignores_key_order(self) -> None:
        assert stable_json_hash({"a": 1, "b": 2}) == stable_json_hash({"b": 2, "a": 1})

    def test_json_hash_distinguishes_values(self) -> None:
        assert stable_json_hash({"a": 1}) != stable_json_hash({"a": 2})


class TestCanonicalJson:
    def test_sorts_keys_and_omits_whitespace(self) -> None:
        assert canonical_json_dumps({"b": 1, "a": 2}) == '{"a":2,"b":1}'

    def test_rejects_non_finite_floats(self) -> None:
        with pytest.raises(NasEngineError, match="not canonically JSON-serialisable"):
            canonical_json_dumps({"x": float("nan")})

    def test_rejects_unserialisable_objects(self) -> None:
        with pytest.raises(NasEngineError):
            canonical_json_dumps({"x": object()})


class TestJsonIo:
    def test_round_trips_through_a_file(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "value.json"
        write_json(target, {"a": [1, 2, 3]})
        assert read_json(target) == {"a": [1, 2, 3]}

    def test_write_is_atomic_leaving_no_temporary_file(self, tmp_path: Path) -> None:
        target = tmp_path / "value.json"
        write_json(target, {"a": 1})
        assert [path.name for path in tmp_path.iterdir()] == ["value.json"]

    def test_rejects_oversized_payload(self) -> None:
        with pytest.raises(NasEngineError, match="exceeds the limit"):
            read_json_bytes(b'{"a": 1}', max_bytes=2)

    def test_rejects_oversized_file(self, tmp_path: Path) -> None:
        target = tmp_path / "big.json"
        target.write_text('{"a": "' + "x" * 100 + '"}', encoding="utf-8")
        with pytest.raises(NasEngineError, match="exceeds the limit"):
            read_json(target, max_bytes=10)

    def test_reports_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(NasEngineError, match="not found"):
            read_json(tmp_path / "absent.json")

    def test_reports_malformed_json(self, tmp_path: Path) -> None:
        target = tmp_path / "bad.json"
        target.write_text("{not json", encoding="utf-8")
        with pytest.raises(NasEngineError, match="not valid UTF-8 JSON"):
            read_json(target)


class TestSafeFilename:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("simple", "simple"),
            ("with space", "with_space"),
            ("../../etc/passwd", "etc_passwd"),
            ("a/b/c", "a_b_c"),
            ("...", "unnamed"),
            ("", "unnamed"),
            ("a;rm -rf /", "a_rm_-rf"),
        ],
    )
    def test_reduces_hostile_input(self, raw: str, expected: str) -> None:
        assert safe_filename(raw) == expected

    def test_truncates_long_names(self) -> None:
        assert len(safe_filename("x" * 500)) <= 120

    def test_avoids_windows_reserved_names(self) -> None:
        assert safe_filename("con") == "con_"
        assert safe_filename("LPT1") == "LPT1_"


class TestPathValidation:
    def test_accepts_paths_under_the_root(self, tmp_path: Path) -> None:
        assert resolve_under_root(tmp_path, "a", "b").parent.name == "a"

    def test_root_itself_counts_as_inside(self, tmp_path: Path) -> None:
        assert is_within(tmp_path, tmp_path)

    def test_rejects_traversal(self, tmp_path: Path) -> None:
        with pytest.raises(UnsafePathError, match="escapes its permitted root"):
            resolve_under_root(tmp_path, "..", "outside")

    def test_rejects_absolute_components(self, tmp_path: Path) -> None:
        with pytest.raises(UnsafePathError, match="absolute path component"):
            resolve_under_root(tmp_path, "/etc/passwd")

    def test_detects_sibling_directories(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        root.mkdir()
        (tmp_path / "sibling").mkdir()
        assert not is_within(tmp_path / "sibling", root)

    def test_ensure_directory_creates_and_resolves(self, tmp_path: Path) -> None:
        created = ensure_directory(tmp_path / "deep" / "nested")
        assert created.is_dir()

    def test_ensure_directory_rejects_a_file(self, tmp_path: Path) -> None:
        target = tmp_path / "file"
        target.write_text("x", encoding="utf-8")
        with pytest.raises(UnsafePathError, match="exists as a file"):
            ensure_directory(target)


class TestTiming:
    def test_timestamps_are_timezone_aware_utc(self) -> None:
        now = utc_now()
        assert now.tzinfo is not None
        assert now.utcoffset() == timezone.utc.utcoffset(None)

    def test_iso_timestamp_carries_an_offset(self) -> None:
        assert utc_now_iso().endswith("+00:00")

    def test_stopwatch_measures_non_negative_elapsed_time(self) -> None:
        with Stopwatch() as watch:
            sum(range(1000))
        assert watch.elapsed_seconds >= 0.0

    def test_stopwatch_freezes_after_stop(self) -> None:
        watch = Stopwatch().start()
        first = watch.stop()
        assert watch.elapsed_seconds == first

    def test_stopwatch_requires_start(self) -> None:
        watch = Stopwatch()
        with pytest.raises(RuntimeError, match="before start"):
            watch.stop()
        with pytest.raises(RuntimeError, match="before start"):
            _ = watch.elapsed_seconds


class TestSeeding:
    def test_derivation_is_deterministic(self) -> None:
        assert derive_seed(42, "strategy") == derive_seed(42, "strategy")

    def test_labels_produce_independent_streams(self) -> None:
        assert derive_seed(42, "strategy") != derive_seed(42, "sampler")

    def test_master_seed_changes_every_derived_seed(self) -> None:
        first = SeedBundle.from_master(1)
        second = SeedBundle.from_master(2)
        assert first.to_dict() != second.to_dict()

    def test_derived_seeds_are_in_numpy_range(self) -> None:
        assert all(0 <= value < 2**32 for value in SeedBundle.from_master(42).to_dict().values())

    def test_worker_bundles_are_isolated(self) -> None:
        bundle = SeedBundle.from_master(42)
        assert bundle.for_worker(0).to_dict() != bundle.for_worker(1).to_dict()

    def test_seed_everything_rejects_negative_seeds(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            seed_everything(-1)

    def test_seed_everything_makes_python_random_reproducible(self) -> None:
        seed_everything(11)
        first = [random.random() for _ in range(5)]
        seed_everything(11)
        assert [random.random() for _ in range(5)] == first

    def test_torch_generator_is_reproducible(self) -> None:
        import torch

        first = torch.randn(4, generator=torch_generator(3))
        second = torch.randn(4, generator=torch_generator(3))
        assert torch.equal(first, second)

    def test_rng_state_round_trips(self) -> None:
        source = random.Random(5)
        source.random()
        restored = rng_state_from_json(rng_state_to_json(source))
        assert restored.getstate() == source.getstate()

    def test_rng_state_restores_the_exact_position(self) -> None:
        source = random.Random(5)
        [source.random() for _ in range(3)]
        payload = rng_state_to_json(source)
        expected = source.random()
        assert rng_state_from_json(payload).random() == expected

    def test_rng_state_rejects_incomplete_payloads(self) -> None:
        with pytest.raises(ValueError, match="missing keys"):
            rng_state_from_json({"version": 3})

    def test_dataloader_worker_init_is_deterministic(self) -> None:
        dataloader_worker_init(0, base_seed=42)
        first = random.random()
        dataloader_worker_init(0, base_seed=42)
        assert random.random() == first

    def test_dataloader_workers_get_different_streams(self) -> None:
        dataloader_worker_init(0, base_seed=42)
        first = random.random()
        dataloader_worker_init(1, base_seed=42)
        assert random.random() != first


class TestDeterminism:
    def test_reports_disabled_state(self) -> None:
        report = configure_determinism(enabled=False)
        assert report.requested is False
        assert report.cudnn_benchmark is True
        assert report.warnings

    def test_reports_enabled_state(self) -> None:
        report = configure_determinism(enabled=True, warn_only=True)
        try:
            assert report.requested is True
            assert report.cudnn_deterministic is True
            assert report.cudnn_benchmark is False
            assert "requested" in report.to_dict()
        finally:
            configure_determinism(enabled=False)


class TestEnvironment:
    def test_captures_versions_and_platform(self) -> None:
        info = collect_environment()
        assert info.python_version
        assert info.torch_version
        assert info.cpu_count >= 1
        assert "cuda_available" in info.accelerator

    def test_summary_lines_are_human_readable(self) -> None:
        lines = collect_environment().summary_lines()
        assert len(lines) == 4
        assert any("PyTorch" in line for line in lines)

    def test_only_allow_listed_variables_are_captured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OMP_NUM_THREADS", "3")
        monkeypatch.setenv("MY_SECRET_TOKEN", "hunter2")
        captured = collect_environment().environment_variables
        assert captured.get("OMP_NUM_THREADS") == "3"
        assert "MY_SECRET_TOKEN" not in captured

    def test_serialises_to_plain_data(self) -> None:
        payload = collect_environment().to_dict()
        assert isinstance(payload["accelerator"], dict)
        assert isinstance(payload["environment_variables"], dict)
