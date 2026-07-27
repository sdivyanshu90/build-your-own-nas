"""Centralised seed management.

Setting ``random.seed(42)`` seeds exactly one generator. A NAS run touches at least
six independent sources of randomness, and every one of them must be controlled for a
search to be reproducible:

======================================  ==========================================
Source                                  Controlled by
======================================  ==========================================
Python ``random``                       :func:`random.seed`
NumPy legacy global generator           :func:`numpy.random.seed`
PyTorch CPU RNG                         :func:`torch.manual_seed`
PyTorch CUDA RNGs (all devices)         :func:`torch.cuda.manual_seed_all`
DataLoader worker processes             ``worker_init_fn`` + a seeded generator
Search-strategy sampling                A private :class:`random.Random` instance
======================================  ==========================================

**Derived seeds, not shared generators.** Components never share one global
generator, because then the values any component receives would depend on how often
every *other* component drew — an invisible coupling that breaks reproducibility as
soon as unrelated code changes. Instead each component receives a seed *derived* from
the run's master seed and a stable string label (:func:`derive_seed`). Derivation is a
pure function of ``(master_seed, label)``, so a component's stream is identical
whether it runs first, last, or in a different process.

See ``docs/concepts/reproducibility.md`` for the distinction between reproducibility,
determinism, and statistical repeatability.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from nas_engine.utilities.hashing import stable_hash

#: Seeds are reduced into this range so they are valid for NumPy (``uint32``) and for
#: PyTorch/Python alike. NumPy's legacy ``seed`` rejects values >= 2**32.
_SEED_MODULUS: int = 2**32


def derive_seed(master_seed: int, label: str) -> int:
    """Derive a stable component seed from a master seed and a text label.

    The derivation hashes ``"{master_seed}:{label}"`` with BLAKE2b and reduces the
    digest modulo ``2**32``. Properties that matter:

    * **Deterministic.** The same pair always yields the same seed, in any process.
    * **Independent.** Labels that differ by one character produce unrelated seeds,
      so component streams do not correlate.
    * **Order-free.** A component's seed does not depend on when it was created.

    Args:
        master_seed: The run-level seed from configuration.
        label: Stable identifier for the component, e.g. ``"strategy"`` or
            ``"worker:3"``.

    Returns:
        An integer seed in ``[0, 2**32)``.
    """
    digest = stable_hash(f"{master_seed}:{label}", digest_bytes=8)
    return int(digest, 16) % _SEED_MODULUS


@dataclass(frozen=True)
class SeedBundle:
    """The set of derived seeds used by one search run.

    Attributes:
        master: The configured run seed; every other value derives from it.
        strategy: Seed for the search strategy's private generator.
        sampler: Seed for architecture sampling.
        mutation: Seed for mutation operators.
        data: Seed for dataset shuffling and splitting.
        training: Seed for weight initialisation and training-time randomness.
    """

    master: int
    strategy: int
    sampler: int
    mutation: int
    data: int
    training: int

    @classmethod
    def from_master(cls, master_seed: int) -> SeedBundle:
        """Build a bundle by deriving every component seed from ``master_seed``.

        Args:
            master_seed: The configured run seed.

        Returns:
            A fully populated :class:`SeedBundle`.
        """
        return cls(
            master=master_seed,
            strategy=derive_seed(master_seed, "strategy"),
            sampler=derive_seed(master_seed, "sampler"),
            mutation=derive_seed(master_seed, "mutation"),
            data=derive_seed(master_seed, "data"),
            training=derive_seed(master_seed, "training"),
        )

    def for_worker(self, worker_id: int) -> SeedBundle:
        """Return a bundle whose streams are isolated to one worker process.

        Every worker must draw from a *different* stream, otherwise concurrent workers
        would initialise identical weights and augment data identically, silently
        reducing the effective diversity of the search.

        Args:
            worker_id: Zero-based worker index.

        Returns:
            A bundle derived from ``master`` and the worker index.
        """
        return SeedBundle.from_master(derive_seed(self.master, f"worker:{worker_id}"))

    def to_dict(self) -> dict[str, int]:
        """Return the bundle as a plain dictionary for persistence and logging."""
        return {
            "master": self.master,
            "strategy": self.strategy,
            "sampler": self.sampler,
            "mutation": self.mutation,
            "data": self.data,
            "training": self.training,
        }


def seed_everything(seed: int, *, seed_cuda: bool = True) -> SeedBundle:
    """Seed every global random source and return the derived seed bundle.

    This affects *process-global* state, which is exactly why component code should
    prefer explicit local generators. It is still called once per process (and once
    per worker) so that any third-party code relying on global RNGs behaves
    reproducibly.

    Args:
        seed: The master seed.
        seed_cuda: Whether to seed CUDA generators when CUDA is available.

    Returns:
        The :class:`SeedBundle` derived from ``seed``.

    Raises:
        ValueError: If ``seed`` is negative.
    """
    if seed < 0:
        msg = f"seed must be non-negative, received {seed}"
        raise ValueError(msg)

    bundle = SeedBundle.from_master(seed)
    # PYTHONHASHSEED only takes effect at interpreter start-up; setting it here
    # documents intent and is inherited by child processes spawned later.
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed % _SEED_MODULUS)
    torch.manual_seed(seed)
    if seed_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return bundle


def torch_generator(seed: int) -> torch.Generator:
    """Create an explicitly seeded CPU :class:`torch.Generator`.

    Passing an explicit generator to :class:`~torch.utils.data.DataLoader` and to
    sampling utilities avoids depending on the global RNG, so shuffling order is
    unaffected by unrelated code that also draws random numbers.

    Args:
        seed: Seed for the generator.

    Returns:
        A CPU generator seeded with ``seed``.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def dataloader_worker_init(worker_id: int, *, base_seed: int) -> None:
    """Seed a DataLoader worker process deterministically.

    PyTorch seeds each worker's ``torch`` RNG automatically, but leaves Python's
    :mod:`random` and NumPy's global RNG untouched — a well-known source of duplicated
    augmentations across workers. This function closes that gap.

    Use with :func:`functools.partial`::

        worker_init_fn=functools.partial(dataloader_worker_init, base_seed=seed)

    Args:
        worker_id: Index supplied by PyTorch.
        base_seed: Seed shared by the DataLoader; combined with ``worker_id``.
    """
    worker_seed = derive_seed(base_seed, f"dataloader:{worker_id}")
    random.seed(worker_seed)
    np.random.seed(worker_seed % _SEED_MODULUS)
    torch.manual_seed(worker_seed)


def rng_state_to_json(rng: random.Random) -> dict[str, Any]:
    """Serialise a :class:`random.Random` state to JSON-safe plain data.

    Resuming a search must continue the *same* random stream, not restart it. Re-seeding
    from the master seed would replay proposals that were already evaluated. The full
    Mersenne Twister state is therefore checkpointed.

    :meth:`random.Random.getstate` returns ``(version, keys, gauss_next)`` where ``keys``
    is a 625-element tuple of integers. JSON has no tuples, so the structure is flattened
    into lists and reassembled on load.

    Args:
        rng: Generator whose state should be captured.

    Returns:
        A JSON-serialisable dictionary.
    """
    version, keys, gauss_next = rng.getstate()
    return {"version": version, "keys": list(keys), "gauss_next": gauss_next}


def rng_state_from_json(payload: dict[str, Any]) -> random.Random:
    """Rebuild a :class:`random.Random` from :func:`rng_state_to_json` output.

    Args:
        payload: Previously serialised state.

    Returns:
        A generator positioned exactly where the original left off.

    Raises:
        ValueError: If the payload is missing keys or has an unsupported state version.
    """
    required = {"version", "keys", "gauss_next"}
    missing = required - payload.keys()
    if missing:
        msg = f"RNG state payload is missing keys {sorted(missing)}"
        raise ValueError(msg)
    rng = random.Random()  # noqa: S311 - reproducible search sampling, not cryptography
    try:
        rng.setstate(
            (int(payload["version"]), tuple(int(k) for k in payload["keys"]), payload["gauss_next"])
        )
    except (TypeError, ValueError) as exc:
        msg = f"RNG state payload could not be restored: {exc}"
        raise ValueError(msg) from exc
    return rng


def rng_state_snapshot() -> dict[str, Any]:
    """Capture a coarse snapshot of global RNG state for diagnostics.

    The snapshot is intentionally *not* used to restore state — restoring RNG state
    across processes and PyTorch versions is fragile. Resumed searches instead
    re-derive seeds from the persisted master seed, which is version independent.

    Returns:
        A dictionary describing the current global RNG state.
    """
    return {
        "python_random": random.getstate()[1][:4],
        "torch_initial_seed": torch.initial_seed(),
        "cuda_available": torch.cuda.is_available(),
    }


__all__ = [
    "SeedBundle",
    "dataloader_worker_init",
    "derive_seed",
    "rng_state_from_json",
    "rng_state_snapshot",
    "rng_state_to_json",
    "seed_everything",
    "torch_generator",
]
