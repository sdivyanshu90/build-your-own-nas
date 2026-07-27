"""Property-based tests for the architecture genotype.

Each property here states an invariant that must hold for *every* architecture, not just
the ones a developer thought to write down. Hypothesis searches for counterexamples and
shrinks any it finds to a minimal case.

The invariants tested are the ones the rest of the system silently assumes:

* canonicalisation is idempotent and erases inactive conditional fields;
* equal canonical forms produce equal hashes, and unequal forms almost never collide;
* serialisation round-trips exactly;
* the analytic cost model equals the measured parameter count;
* static shape inference agrees with what PyTorch actually does;
* a built model's output shape matches the configured class count.
"""

from __future__ import annotations

import pytest
import torch
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from nas_engine.architectures.canonical import (
    from_canonical_dict,
    from_canonical_json,
    to_canonical_dict,
    to_canonical_json,
)
from nas_engine.architectures.cost import compute_cost
from nas_engine.architectures.hashing import architecture_hash
from nas_engine.architectures.shapes import conv_output_size, infer_shapes, make_divisible
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
from nas_engine.exceptions import ShapeInferenceError
from nas_engine.models.builder import ModelBuilder, count_parameters
from tests.profiles import scaled

pytestmark = [pytest.mark.property]

#: Hypothesis profile for tests that build PyTorch models. Model construction dominates
#: the runtime, so the example count is deliberately modest; the nightly workflow raises it.
MODEL_SETTINGS = settings(
    max_examples=scaled(25),
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)

DATA_SETTINGS = settings(max_examples=scaled(100), deadline=None)


# ---------------------------------------------------------------------- strategies --
kernel_sizes = st.sampled_from([1, 3, 5, 7])
channels = st.sampled_from([4, 8, 16, 24, 32])
strides = st.sampled_from([1, 2])
operations = st.sampled_from(list(OperationType))
normalizations = st.sampled_from(list(NormalizationType))
activations = st.sampled_from(list(ActivationType))
expansions = st.sampled_from([0.5, 1.0, 2.0, 3.0, 4.0])


@st.composite
def blocks(
    draw: st.DrawFn,
    *,
    out_channels: int | None = None,
    operation: st.SearchStrategy[OperationType] | None = None,
) -> BlockSpec:
    """Draw an arbitrary, canonical block specification.

    Args:
        draw: Hypothesis draw function.
        out_channels: Fix the output width instead of drawing one.
        operation: Restrict the operation. Pass a narrowed strategy rather than filtering
            the result with ``assume``: there are five operations, so a test that wants
            only one would discard four fifths of everything Hypothesis generates, which
            is both slow and enough to trip the ``filter_too_much`` health check on an
            unlucky seed.
    """
    return BlockSpec(
        operation=draw(operation if operation is not None else operations),
        kernel_size=draw(kernel_sizes),
        expansion_ratio=draw(expansions),
        out_channels=out_channels if out_channels is not None else draw(channels),
        stride=draw(strides),
        use_residual=draw(st.booleans()),
        normalization=draw(normalizations),
        activation=draw(activations),
    )


@st.composite
def buildable_architectures(draw: st.DrawFn) -> ArchitectureSpec:
    """Draw an architecture that is guaranteed to pass shape inference.

    Rather than drawing freely and discarding invalid results — which wastes most of
    Hypothesis's budget — the generator constructs valid architectures by construction:
    the first block of each stage owns the width change and the stride, later blocks keep
    the stage width at stride 1, and residuals are only offered where shapes match.
    """
    input_size = draw(st.sampled_from([16, 32]))
    num_classes = draw(st.integers(min_value=2, max_value=10))
    stem_channels = draw(st.sampled_from([8, 16]))

    stem = StemSpec(
        out_channels=stem_channels,
        kernel_size=draw(st.sampled_from([3, 5])),
        stride=1,
        normalization=draw(normalizations),
        activation=draw(activations),
    )

    stage_count = draw(st.integers(min_value=1, max_value=3))
    current_channels = stem_channels
    current_size = input_size
    stages: list[StageSpec] = []

    for _ in range(stage_count):
        width = draw(channels)
        depth = draw(st.integers(min_value=1, max_value=3))
        stage_blocks: list[BlockSpec] = []
        for position in range(depth):
            stride = draw(strides) if position == 0 else 1
            if current_size // stride < 2:
                stride = 1
            preserves = current_channels == width
            candidate_operations = [
                operation
                for operation in OperationType
                if (preserves or operation.can_change_channels)
                and not (stride > 1 and operation is OperationType.IDENTITY)
            ]
            operation = draw(st.sampled_from(candidate_operations))
            out_channels = width if operation.can_change_channels else current_channels
            shape_preserved = stride == 1 and out_channels == current_channels
            stage_blocks.append(
                BlockSpec(
                    operation=operation,
                    kernel_size=draw(kernel_sizes),
                    expansion_ratio=draw(expansions),
                    out_channels=out_channels,
                    stride=stride,
                    use_residual=draw(st.booleans()) and shape_preserved,
                    normalization=draw(normalizations),
                    activation=draw(activations),
                )
            )
            current_channels = out_channels
            current_size = conv_output_size(current_size, stage_blocks[-1].kernel_size, stride)
        stages.append(StageSpec(blocks=tuple(stage_blocks)))

    head = HeadSpec(
        pooling=draw(st.sampled_from(list(PoolingType))),
        hidden_units=draw(st.sampled_from([0, 8, 16])),
        dropout=draw(st.sampled_from([0.0, 0.1, 0.25])),
        activation=draw(activations),
    )
    return ArchitectureSpec(
        input_channels=3,
        input_size=input_size,
        num_classes=num_classes,
        stem=stem,
        stages=tuple(stages),
        head=head,
    )


# ------------------------------------------------------------------------ properties --
class TestBlockCanonicalisation:
    @given(block=blocks())
    @DATA_SETTINGS
    def test_canonicalisation_is_idempotent(self, block: BlockSpec) -> None:
        assert block.evolve() == block

    @given(block=blocks())
    @DATA_SETTINGS
    def test_inactive_fields_hold_their_sentinel_values(self, block: BlockSpec) -> None:
        if not block.operation.uses_kernel_size:
            assert block.kernel_size == 1
        if not block.operation.uses_expansion_ratio:
            assert block.expansion_ratio == 1.0
        if not block.operation.is_parametric:
            assert block.normalization is NormalizationType.NONE
            assert block.activation is ActivationType.IDENTITY
        if block.operation is OperationType.IDENTITY:
            assert block.stride == 1
            assert block.use_residual is False

    @given(
        block=blocks(operation=st.just(OperationType.IDENTITY)),
        kernel=kernel_sizes,
    )
    @DATA_SETTINGS
    def test_kernel_size_is_irrelevant_for_identity(self, block: BlockSpec, kernel: int) -> None:
        assert block.operation is OperationType.IDENTITY
        assert block.evolve(kernel_size=kernel) == block

    @given(
        block=blocks(
            operation=st.sampled_from([op for op in OperationType if not op.uses_expansion_ratio])
        ),
        expansion=expansions,
    )
    @DATA_SETTINGS
    def test_expansion_is_irrelevant_for_non_separable_operations(
        self, block: BlockSpec, expansion: float
    ) -> None:
        assert not block.operation.uses_expansion_ratio
        assert block.evolve(expansion_ratio=expansion) == block


class TestHashing:
    @given(spec=buildable_architectures())
    @DATA_SETTINGS
    def test_equal_architectures_hash_identically(self, spec: ArchitectureSpec) -> None:
        clone = from_canonical_dict(to_canonical_dict(spec))
        assert architecture_hash(clone) == architecture_hash(spec)

    @given(spec=buildable_architectures())
    @DATA_SETTINGS
    def test_hash_has_a_fixed_shape(self, spec: ArchitectureSpec) -> None:
        digest = architecture_hash(spec)
        assert len(digest) == 32
        assert all(character in "0123456789abcdef" for character in digest)

    @given(first=buildable_architectures(), second=buildable_architectures())
    @DATA_SETTINGS
    def test_different_architectures_almost_never_collide(
        self, first: ArchitectureSpec, second: ArchitectureSpec
    ) -> None:
        if to_canonical_json(first) == to_canonical_json(second):
            assert architecture_hash(first) == architecture_hash(second)
        else:
            assert architecture_hash(first) != architecture_hash(second)

    @given(spec=buildable_architectures(), data=st.data())
    @DATA_SETTINGS
    def test_changing_an_active_field_changes_the_hash(
        self, spec: ArchitectureSpec, data: st.DataObject
    ) -> None:
        block = spec.stages[0].blocks[0]
        assume(block.operation.uses_kernel_size)
        # Draw a *different* kernel rather than drawing freely and filtering: the
        # alternative discards a third of every example that survived the assume above.
        kernel = data.draw(
            st.sampled_from([size for size in (3, 5, 7) if size != block.kernel_size])
        )
        mutated = spec.with_block(0, 0, block.evolve(kernel_size=kernel))
        assert architecture_hash(mutated) != architecture_hash(spec)


class TestSerialisation:
    @given(spec=buildable_architectures())
    @DATA_SETTINGS
    def test_json_round_trip_is_lossless(self, spec: ArchitectureSpec) -> None:
        assert from_canonical_json(to_canonical_json(spec)) == spec

    @given(spec=buildable_architectures())
    @DATA_SETTINGS
    def test_canonical_json_is_ascii(self, spec: ArchitectureSpec) -> None:
        assert to_canonical_json(spec).isascii()

    @given(spec=buildable_architectures())
    @DATA_SETTINGS
    def test_round_trip_is_a_fixed_point(self, spec: ArchitectureSpec) -> None:
        once = to_canonical_json(spec)
        twice = to_canonical_json(from_canonical_json(once))
        assert once == twice


class TestShapeArithmetic:
    @given(
        size=st.integers(min_value=1, max_value=256),
        kernel=kernel_sizes,
        stride=st.integers(min_value=1, max_value=4),
    )
    @DATA_SETTINGS
    def test_same_padding_equals_ceiling_division(
        self, size: int, kernel: int, stride: int
    ) -> None:
        expected = -(-size // stride)  # ceiling division
        assert conv_output_size(size, kernel, stride) == expected

    @given(size=st.integers(min_value=1, max_value=256), kernel=kernel_sizes)
    @DATA_SETTINGS
    def test_stride_one_preserves_the_extent(self, size: int, kernel: int) -> None:
        assert conv_output_size(size, kernel, 1) == size

    @given(value=st.floats(min_value=1.0, max_value=4096.0, allow_nan=False))
    @DATA_SETTINGS
    def test_rounding_stays_on_the_grid(self, value: float) -> None:
        rounded = make_divisible(value)
        assert rounded % 8 == 0
        assert rounded >= 8

    @given(value=st.floats(min_value=8.0, max_value=4096.0, allow_nan=False))
    @DATA_SETTINGS
    def test_rounding_never_loses_more_than_ten_percent(self, value: float) -> None:
        assert make_divisible(value) >= 0.9 * value


class TestGeneratedArchitectures:
    @given(spec=buildable_architectures())
    @DATA_SETTINGS
    def test_generated_architectures_pass_shape_inference(self, spec: ArchitectureSpec) -> None:
        trace = infer_shapes(spec)
        assert trace.features_shape.height >= 1
        assert trace.output_features == spec.num_classes

    @given(spec=buildable_architectures())
    @DATA_SETTINGS
    def test_costs_are_positive_and_consistent(self, spec: ArchitectureSpec) -> None:
        cost = compute_cost(spec)
        assert cost.trainable_parameters > 0
        assert cost.multiply_accumulates > 0
        assert cost.total_parameters >= cost.trainable_parameters

    @given(spec=buildable_architectures())
    @DATA_SETTINGS
    def test_residuals_only_appear_where_shapes_match(self, spec: ArchitectureSpec) -> None:
        trace = infer_shapes(spec)
        block_layers = trace.layers[1 : 1 + spec.total_blocks]
        for (_, _, block), layer in zip(spec.iter_blocks(), block_layers, strict=True):
            if block.use_residual:
                assert layer.input_shape.as_tuple() == layer.output_shape.as_tuple()


class TestModelAgreement:
    @given(spec=buildable_architectures())
    @MODEL_SETTINGS
    def test_analytic_cost_equals_the_measured_parameter_count(
        self, spec: ArchitectureSpec
    ) -> None:
        model = ModelBuilder(initialize=False).build(spec)
        trainable, _ = count_parameters(model)
        assert trainable == compute_cost(spec).trainable_parameters

    @given(spec=buildable_architectures())
    @MODEL_SETTINGS
    def test_output_shape_matches_the_class_count(self, spec: ArchitectureSpec) -> None:
        model = ModelBuilder(initialize=False).build(spec)
        output = model(torch.zeros(2, spec.input_channels, spec.input_size, spec.input_size))
        assert output.shape == (2, spec.num_classes)

    @given(spec=buildable_architectures())
    @MODEL_SETTINGS
    def test_static_shapes_match_what_pytorch_produces(self, spec: ArchitectureSpec) -> None:
        model = ModelBuilder(initialize=False).build(spec)
        runtime = dict(
            model.feature_shapes(
                torch.zeros(1, spec.input_channels, spec.input_size, spec.input_size)
            )
        )
        static = infer_shapes(spec)
        assert runtime["stem"][1:] == static.layers[0].output_shape.as_tuple()
        assert runtime[f"stages.{spec.num_stages - 1}"][1:] == (static.features_shape.as_tuple())

    @given(spec=buildable_architectures())
    @MODEL_SETTINGS
    def test_forward_pass_produces_finite_logits(self, spec: ArchitectureSpec) -> None:
        model = ModelBuilder().build(spec)
        model.eval()
        with torch.no_grad():
            output = model(torch.randn(2, spec.input_channels, spec.input_size, spec.input_size))
        assert torch.isfinite(output).all()


class TestInvalidArchitecturesAreRejected:
    @given(
        out_channels=st.sampled_from([8, 16, 32]),
        wrong_channels=st.sampled_from([64, 128]),
    )
    @DATA_SETTINGS
    def test_channel_preserving_operations_cannot_change_width(
        self, out_channels: int, wrong_channels: int
    ) -> None:
        spec = ArchitectureSpec(
            input_size=16,
            num_classes=4,
            stem=StemSpec(out_channels=out_channels),
            stages=(
                StageSpec(
                    blocks=(
                        BlockSpec(operation=OperationType.MAX_POOL, out_channels=wrong_channels),
                    )
                ),
            ),
        )
        with pytest.raises(ShapeInferenceError):
            infer_shapes(spec)
