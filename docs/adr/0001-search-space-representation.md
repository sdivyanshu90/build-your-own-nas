# ADR 0001 — Architectures are typed, immutable, canonically-hashed Pydantic models

**Status:** Accepted
**Date:** 2026-07-27
**Supersedes:** none

## Context

Everything else depends on how an architecture is represented. The search strategy mutates
it, the model builder consumes it, the database stores it, the deduplicator compares it,
and the report explains it. A representation that is convenient for one of those is usually
wrong for another.

Five properties are non-negotiable, and they pull against each other:

1. **Identity.** Two architectures that describe the same network must compare equal and
   produce the same key, or the search re-evaluates work it has already done. Deduplication
   is not an optimisation here — an evolutionary strategy proposes near-duplicates
   constantly, and re-evaluating them wastes most of the budget.
2. **Serialisability.** A candidate crosses a process boundary to a worker, and lands in a
   database that must still be readable in a year.
3. **Validity.** A search space produces architectures mechanically. Most of what it can
   produce is nonsense — a stride that reduces a feature map below 1×1, a residual
   connection between mismatched widths. Invalid architectures must be rejectable
   *cheaply*, before anything is built.
4. **Mutability under control.** Evolution needs to produce a modified copy. It must not be
   able to modify the original, which may still be in a population, a checkpoint, or a
   database row.
5. **Legibility.** A human reading a stored architecture should be able to tell what it is.

The obvious candidates each fail at least one.

## Decision

An architecture is a tree of **frozen Pydantic models** — `ArchitectureSpec` containing a
`StemSpec`, a tuple of `StageSpec`, each holding a tuple of `BlockSpec`, and a `HeadSpec`.
Every categorical field is an enum from a closed vocabulary (`OperationType`,
`ActivationType`, `NormalizationType`, `PoolingType`). Every numeric field carries its
bounds in the type.

Four things follow from that core decision:

**Genotype and phenotype are separate.** The spec is a *description*; a `torch.nn.Module`
is built from it on demand by `ModelBuilder`. The spec never holds tensors, never holds a
device, and is cheap to copy, hash, store, and send to a worker. A parameter count can be
computed from it analytically, without building anything.

**Identity is a content hash of canonical JSON.** `to_canonical_dict` produces a mapping
with sorted keys and compact separators, and — critically — **canonicalises inactive
fields**. When a block's operation is `pooling`, its `expansion_ratio` is meaningless; the
canonical form resets it to the default so that two blocks which build the identical layer
hash identically. `architecture_hash` is BLAKE2b truncated to 128 bits, rendered as 32 hex
characters.

**Modification returns a new instance.** `evolve(**changes)` re-runs the constructor with
the changed fields, so the result is validated rather than merely constructed. Frozen models
make the alternative — accidental in-place mutation of a population member — impossible
rather than merely discouraged.

**Imported JSON is untrusted.** `from_canonical_dict` rejects unknown fields
(`extra="forbid"`), rejects enum values outside the vocabulary, and enforces every numeric
bound. It never executes anything from the payload.

## Alternatives considered

### A plain `dict`

The default choice, and the one most research code uses.

*Rejected.* Nothing prevents a typo (`"kernel"` for `"kernel_size"`) from being silently
accepted and then failing thousands of lines away, in the model builder, with a
`KeyError`. There is no bound checking, so a stride of 47 is representable. Hashing needs a
hand-written canonicalisation that has to be kept in sync by discipline. And a `dict` is
mutable, so a mutation operator can corrupt a population member it was only supposed to
read.

The failure mode that decided it: with dicts, an invalid architecture is discovered when
PyTorch raises, which is *after* the training loop has been set up and the data loaded.

### A dataclass with hand-written validation

Better. Frozen dataclasses give immutability and structural equality for free.

*Rejected*, narrowly. The validation has to be written by hand in `__post_init__`, and JSON
round-tripping has to be written by hand too — including the nested-tuple reconstruction,
which is where the bugs live. Pydantic v2 does both from the type annotations, and its
error messages name the field, the constraint, and the received value. That is exactly what
turns a rejected import into an actionable message.

The cost is a runtime dependency and Pydantic's own quirks — see *Consequences*.

### A string DSL, e.g. `"conv3x3-bn-relu|pool2|conv3x3"`

Compact, human-readable, and trivially hashable.

*Rejected.* Parsing it is a second implementation of the schema, and the two drift. Partial
modification means string surgery. Conditional structure — an expansion ratio that only
applies to separable convolutions — has no natural encoding. And the "human-readable"
claim degrades quickly once the grammar covers everything a real space needs.

`ArchitectureSummary.compact()` provides the readable one-line form for logs and tables,
which is what the DSL was actually wanted for.

### A graph (adjacency matrix plus operation list)

The representation used by NAS-Bench-101 and by cell-based search.

*Rejected for this project*, and this is the most substantive trade-off made here. A graph
is strictly more expressive: multi-branch cells, arbitrary skip patterns, and DAG topologies
are natural in it and inexpressible in a linear stack.

Against that: graph identity requires **graph isomorphism** testing, because the same
network has many adjacency-matrix encodings. NAS-Bench-101 needs a dedicated canonicalisation
pass for exactly this. Shape inference becomes a topological traversal with merge-point
compatibility checking rather than a fold. Mutation must preserve connectivity, so most
random edits are invalid and need repair. Every one of those is a source of subtle bugs.

The chosen scope — sequential CNNs with optional residual connections — is expressible
without a graph, and covers the architectures a reader of this project is likely to search
over. **The linear representation is a scope decision, not a claim that graphs are worse.**
The boundary is documented in
[architecture encoding](../concepts/architecture-encoding.md), and the repository is
explicit that a cell-based space is the natural next extension.

### Pickle for serialisation

*Rejected outright.* Unpickling executes code in the payload. An architecture read from a
database, a shared directory, or a colleague is untrusted input by definition. Pickle is
also version-fragile: a class rename breaks every stored artefact.

### A hash of `repr()` or Python's `hash()`

*Rejected.* `hash()` is salted per process by `PYTHONHASHSEED`, so it is not stable across
runs — which is the one property required. `repr()` is not specified as stable across
Python or Pydantic versions.

## Consequences

### Good

- An invalid architecture is rejected at construction, with a message naming the field.
- Deduplication is an equality check on a 32-character string, so it works across
  processes, across runs, and inside a SQL `UNIQUE` constraint.
- Determinism follows: the same seed produces the same specs and therefore the same hashes.
- Analytic cost is exact for parameter counts —
  `test_analytic_cost_equals_the_measured_parameter_count` asserts the analytic count equals
  PyTorch's for every sampled architecture, which is what makes it safe to use as an
  objective and as a pruning criterion.
- Workers receive plain data. No tensors, no CUDA context, no import-order surprises.

### Bad

- **The hash is a compatibility surface.** Changing the canonical form invalidates every
  stored hash and makes historical results incomparable. `ARCHITECTURE_SCHEMA_VERSION` and
  a golden fixture in `tests/regression/test_golden_fixtures.py` make such a change loud, but
  they cannot make it free.
- **Pydantic v2 has a sharp edge here.** An `@model_validator(mode="after")` that *returns a
  new instance* is silently ignored on a frozen model. Normalisation must mutate in place
  via `object.__setattr__`, which the `_force()` helper wraps. This was a real bug during
  development; it is now covered by a test.
- **Multi-branch topologies are out of scope.** Stated above; stated again because it is the
  limit a reader is most likely to hit.
- A runtime dependency on Pydantic, and its validation cost on every construction — roughly
  100 µs per architecture, which is negligible against a training run and *not* negligible
  in a tight sampling loop. `ArchitectureSampler` constructs once per candidate, not per
  attempt.

## Verification

| Property                                                   | Test                                                                   | Kind            |
| ---------------------------------------------------------- | ---------------------------------------------------------------------- | --------------- |
| Equal architectures hash identically                       | `test_equal_architectures_hash_identically`                            | property        |
| Changing an active field changes the hash                  | `test_changing_an_active_field_changes_the_hash`                       | property        |
| Distinct architectures effectively never collide           | `test_different_architectures_almost_never_collide`                    | property        |
| Inactive fields hold sentinels after canonicalisation      | `test_inactive_fields_hold_their_sentinel_values`                      | property        |
| Two specs differing only in dead fields are equal          | `test_two_specs_differing_only_in_dead_fields_are_equal`               | unit            |
| Canonicalisation is idempotent                             | `test_canonicalisation_is_idempotent`                                  | property + unit |
| JSON round-trips losslessly, and is a fixed point          | `test_json_round_trip_is_lossless`, `test_round_trip_is_a_fixed_point` | property        |
| Unknown fields and operations are rejected                 | `test_rejects_unknown_fields`, `test_rejects_unknown_operations`       | unit            |
| Errors name the offending field                            | `test_reports_the_offending_field`                                     | unit            |
| Specs are immutable                                        | `test_blocks_are_immutable`                                            | unit            |
| `evolve` re-canonicalises rather than bypassing validation | `test_evolve_reapplies_canonicalisation`                               | unit            |
| Analytic parameters equal PyTorch's, exactly               | `test_analytic_cost_equals_the_measured_parameter_count`               | property        |
| Static shapes equal PyTorch's                              | `test_static_shapes_match_what_pytorch_produces`                       | property        |
| The hash has not drifted                                   | `test_hashes_are_unchanged`                                            | regression      |
| The canonical bytes have not drifted                       | `test_canonical_form_round_trips_to_the_same_bytes`                    | regression      |

## See also

- [Architecture encoding](../concepts/architecture-encoding.md) — the concept in full.
- [Search spaces](../concepts/search-spaces.md)
- [Defining search spaces](../guides/defining-search-spaces.md)
- [Security](../architecture/security.md) — the untrusted-input boundary.
