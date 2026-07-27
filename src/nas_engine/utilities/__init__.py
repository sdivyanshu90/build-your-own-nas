"""Cross-cutting helpers with no dependencies on other ``nas_engine`` subpackages.

This package sits at the bottom of the dependency graph. Nothing in here may import
from :mod:`nas_engine.config`, :mod:`nas_engine.persistence`, or any other domain
package, which keeps the module dependency graph acyclic (see
``docs/architecture/component-design.md``).
"""

from nas_engine.utilities.determinism import DeterminismReport, configure_determinism
from nas_engine.utilities.environment import EnvironmentInfo, collect_environment
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

__all__ = [
    "DeterminismReport",
    "EnvironmentInfo",
    "SeedBundle",
    "Stopwatch",
    "canonical_json_dumps",
    "collect_environment",
    "configure_determinism",
    "dataloader_worker_init",
    "derive_seed",
    "ensure_directory",
    "is_within",
    "read_json",
    "read_json_bytes",
    "resolve_under_root",
    "rng_state_from_json",
    "rng_state_to_json",
    "safe_filename",
    "seed_everything",
    "stable_hash",
    "stable_hash_bytes",
    "stable_json_hash",
    "torch_generator",
    "utc_now",
    "utc_now_iso",
    "write_json",
]
