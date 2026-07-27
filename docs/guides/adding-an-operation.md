# Adding an operation

Extending the block vocabulary — and the one consequence you must understand before you
start.

## Read this first

**Adding an operation to the enumeration changes the canonical form, which changes every
architecture hash.**

The hash is computed from canonical JSON. Adding an enum member does not by itself change
any *existing* architecture's JSON — so in the common case, hashes are stable. But if your
change also alters canonicalisation (a new conditional field, a different sentinel value,
a new default), every stored hash becomes wrong, and:

- resuming an existing search re-proposes everything as "novel";
- historical results become incomparable;
- the golden fixtures fail, which is the intended alarm.

If canonicalisation changes, bump `ARCHITECTURE_SCHEMA_VERSION`, regenerate the fixtures,
and document the migration. If it does not, existing hashes stay valid and the fixture test
passes unchanged — which is exactly how you find out which case you are in.

---

## The five files

Worked example: adding a **dilated convolution**.

### 1. The enumeration

`src/nas_engine/architectures/types.py`

```python
class OperationType(str, Enum):
    CONV = "conv"
    DW_SEP_CONV = "dw_sep_conv"
    IDENTITY = "identity"
    MAX_POOL = "max_pool"
    AVG_POOL = "avg_pool"
    DILATED_CONV = "dilated_conv"        # new

    @property
    def is_parametric(self) -> bool:
        return self in {
            OperationType.CONV,
            OperationType.DW_SEP_CONV,
            OperationType.DILATED_CONV,   # owns weights
        }
```

Check every property on the enum: `is_parametric`, `can_change_channels`,
`uses_kernel_size`, `uses_expansion_ratio`. Each drives canonicalisation, so getting one
wrong silently corrupts identity.

Document the operation in the class docstring alongside the others. That docstring is the
reference for what the vocabulary means.

### 2. The block specification, if a new field is needed

`src/nas_engine/architectures/spec.py`

A dilation rate is a new conditional field:

```python
class BlockSpec(_FrozenModel):
    ...
    dilation: Annotated[int, Field(ge=1, le=4)] = 1

    @model_validator(mode="after")
    def _canonicalise(self) -> BlockSpec:
        ...
        # Dilation is meaningful only for the dilated convolution.
        if self.operation is not OperationType.DILATED_CONV and self.dilation != 1:
            _force(self, "dilation", 1)
        return self
```

**This is the change that invalidates hashes**, because the canonical JSON of every
existing block gains a `dilation` key.

Add a `uses_dilation` property to `OperationType` rather than hard-coding the check, so the
knowledge stays in one place.

### 3. Shape inference and the cost model

`src/nas_engine/architectures/shapes.py`

Dilation changes the effective kernel size: $k_{\text{eff}} = d(k-1) + 1$, so same padding
becomes $p = d(k-1)/2$.

```python
def conv_output_size(input_size: int, kernel_size: int, stride: int, dilation: int = 1) -> int:
    effective = dilation * (kernel_size - 1) + 1
    padding = effective // 2
    return (input_size + 2 * padding - effective) // stride + 1
```

`src/nas_engine/architectures/cost.py`

Dilation does not change the parameter count — only the receptive field — so
`conv_parameters` is unchanged. MACs are unchanged too, since the output size is unchanged.

**The analytic cost model must remain exact.** An exactness test asserts that the analytic
parameter count *equals* what PyTorch reports, for every sampled architecture. If your
operation adds parameters, add them here or the test fails immediately — which is the point.

### 4. The model builder

`src/nas_engine/models/blocks.py`

```python
def build_operation(block: BlockSpec, in_channels: int) -> nn.Module:
    ...
    if operation is OperationType.DILATED_CONV:
        effective = block.dilation * (block.kernel_size - 1) + 1
        bias = block.normalization is NormalizationType.NONE
        return nn.Sequential(
            nn.Conv2d(
                in_channels, block.out_channels,
                kernel_size=block.kernel_size, stride=block.stride,
                padding=effective // 2, dilation=block.dilation, bias=bias,
            ),
            build_normalization(block.normalization, block.out_channels),
            build_activation(block.activation),
        )
```

mypy will now flag the exhaustiveness guard at the end of the function as reachable again,
which is how you know you have not missed a branch.

Keep the **bias convention**: no bias when a normalisation follows, bias when
`NormalizationType.NONE`. The cost model depends on it.

### 5. The search space

`src/nas_engine/search_space/space.py`

```python
class BlockChoices(_SpaceModel):
    ...
    dilations: tuple[int, ...] = (1,)

    @model_validator(mode="after")
    def _validate(self) -> BlockChoices:
        ...
        object.__setattr__(self, "dilations", _unique_ordered(self.dilations, "dilations"))
        if any(value < 1 for value in self.dilations):
            raise ValueError(f"dilations must be >= 1, received {list(self.dilations)}")
        return self
```

Update `per_block_choice_count` so the cardinality estimate stays honest:

```python
dilations = len(self.block.dilations) if operation.uses_dilation else 1
total += kernels * expansions * dilations * norms * activations * residual
```

`src/nas_engine/search_space/sampler.py` — draw the new field. Draw it
**unconditionally**, even when inactive, so the number of random draws per block stays
fixed. That keeps a change of operation from shifting every later draw in the stream.

```python
dilation = self._rng.choice(choices.dilations)
```

`src/nas_engine/search_space/validation.py` — check membership:

```python
if block.operation.uses_dilation and block.dilation not in space.block.dilations:
    collector.add("membership", f"{location}.dilation",
                  "dilation is not offered by the search space",
                  received=block.dilation, expected=list(space.block.dilations))
```

`src/nas_engine/search_space/mutation.py` — add an operator and register it:

```python
def mutate_dilation(spec, space, rng):
    positions = [(s, b) for s, b in _all_block_positions(spec)
                 if spec.stages[s].blocks[b].operation.uses_dilation]
    rng.shuffle(positions)
    for stage_index, block_index in positions:
        block = spec.stages[stage_index].blocks[block_index]
        dilation = _choose_other(rng, space.block.dilations, block.dilation)
        if dilation is None:
            continue
        child = spec.with_block(stage_index, block_index, block.evolve(dilation=dilation))
        return child, f"dilation s{stage_index}b{block_index}: {block.dilation} -> {dilation}"
    return None

DEFAULT_OPERATORS = (..., ("dilation", mutate_dilation))
```

---

## Tests to add

| Test | Where | Asserts |
| --- | --- | --- |
| Enum properties are right | `tests/unit/test_architectures.py` | `is_parametric`, `uses_dilation`, etc. |
| Canonicalisation erases the inactive field | same | `dilation == 1` for non-dilated operations |
| The block builds and produces the right shape | `tests/unit/test_models.py` | Output shape matches inference |
| The cost model stays exact | `tests/property/test_architecture_properties.py` | Already parametrised — it will cover the new operation automatically |
| The sampler can produce it | `tests/unit/test_search_space.py` | Appears in a sample from a space that offers it |
| Membership validation catches a foreign value | same | A dilation outside the space is rejected |
| Mutation stays inside the space | `tests/property/test_search_properties.py` | Already parametrised |

Most of the property tests are parametrised over the enum or over sampled architectures, so
they cover a new operation **without modification** — and will fail if you got something
wrong. That is the design working.

---

## Then run the gates

```bash
make lint typecheck
make test
```

mypy is the useful one here: removing an enum member from an exhaustiveness check turns the
defensive branch reachable again, and adding one without handling it makes the function's
return type unsatisfiable. Both are type errors, caught before any test runs.

The golden-fixture test tells you whether hashes changed:

```text
FAILED tests/regression/test_golden_fixtures.py::test_hashes_are_unchanged
  The architecture hash changed. Every stored hash in every existing database is now wrong.
  If this change is intentional, bump fixture_version in tests/fixtures/architectures.json
  and document the migration.
```

If it passes, existing hashes are safe. If it fails and the change was intentional:

1. Bump `ARCHITECTURE_SCHEMA_VERSION` in `spec.py`.
2. Bump `fixture_version` and `architecture_schema_version` in
   `tests/fixtures/architectures.json`.
3. Regenerate the golden values.
4. Note the break in the changelog, so users know their existing databases hold hashes from
   the old scheme.

---

## Adding a normalisation or an activation instead

Much smaller — those are not conditional on the operation:

| Change              | Files                                                                          |
| ------------------- | ------------------------------------------------------------------------------ |
| A new activation    | `types.py` (enum), `models/operations.py` (`build_activation`)                 |
| A new normalisation | `types.py`, `models/operations.py`, `architectures/cost.py` (parameter counts) |
| A new pooling type  | `types.py`, `models/operations.py` (`build_global_pool`)                       |

The cost-model update is the one people forget. `normalization_parameters` must return the
right `(trainable, non_trainable)` pair, and the exactness test will catch it if it does
not.

---

## A checklist

- [ ] Enum member added, with a docstring entry
- [ ] Every enum property reviewed (`is_parametric`, `can_change_channels`,
      `uses_kernel_size`, `uses_expansion_ratio`, and any new one)
- [ ] Canonicalisation handles the new conditional field
- [ ] Shape inference is correct
- [ ] The cost model is **exact** — the exactness test passes
- [ ] The builder produces the right module
- [ ] The bias convention is preserved
- [ ] The search space offers the choice, and validates membership
- [ ] The sampler draws it, unconditionally, to keep the draw count fixed
- [ ] `per_block_choice_count` updated
- [ ] A mutation operator exists and is registered
- [ ] Tests added
- [ ] `make lint typecheck test` passes
- [ ] The golden-fixture outcome is understood and acted on

## See also

- [Architecture encoding](../concepts/architecture-encoding.md) — why hashes matter.
- [Component design](../architecture/component-design.md) — where each file sits.
- [Defining search spaces](defining-search-spaces.md) — using the new operation.
