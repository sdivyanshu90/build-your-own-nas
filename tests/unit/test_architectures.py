"""Unit tests for the architecture genotype.

Covers: field validation, conditional-field canonicalisation, immutability, canonical
serialisation, hashing, equality semantics, shape inference, the analytic cost model,
summaries, and lineage reconstruction.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nas_engine.architectures.canonical import (
    architectures_equal,
    from_canonical_dict,
    from_canonical_json,
    to_canonical_dict,
    to_canonical_json,
)
from nas_engine.architectures.cost import (
    compute_cost,
    conv_macs,
    conv_parameters,
    normalization_parameters,
    separable_hidden_channels,
)
from nas_engine.architectures.hashing import (
    ARCHITECTURE_HASH_LENGTH,
    architecture_hash,
    short_hash,
)
from nas_engine.architectures.lineage import LineageGraph, LineageNode
from nas_engine.architectures.shapes import (
    TensorShape,
    conv_output_size,
    infer_shapes,
    make_divisible,
)
from nas_engine.architectures.spec import (
    ARCHITECTURE_SCHEMA_VERSION,
    ArchitectureSpec,
    BlockSpec,
    HeadSpec,
    StageSpec,
    StemSpec,
    quantise,
)
from nas_engine.architectures.summary import summarise
from nas_engine.architectures.types import (
    ActivationType,
    NormalizationType,
    OperationType,
)
from nas_engine.exceptions import ArchitectureValidationError, ShapeInferenceError

pytestmark = pytest.mark.unit


def _conv_block(**overrides: object) -> BlockSpec:
    """Build a convolutional block with test-friendly defaults."""
    defaults: dict[str, object] = {
        "operation": OperationType.CONV,
        "kernel_size": 3,
        "out_channels": 16,
        "stride": 1,
    }
    defaults.update(overrides)
    return BlockSpec(**defaults)  # type: ignore[arg-type]


class TestOperationTypeProperties:
    def test_only_convolutions_are_parametric(self) -> None:
        parametric = {op for op in OperationType if op.is_parametric}
        assert parametric == {OperationType.CONV, OperationType.DW_SEP_CONV}

    def test_only_parametric_operations_change_channels(self) -> None:
        for operation in OperationType:
            assert operation.can_change_channels == operation.is_parametric

    def test_identity_has_no_meaningful_kernel(self) -> None:
        assert not OperationType.IDENTITY.uses_kernel_size
        assert OperationType.MAX_POOL.uses_kernel_size

    def test_only_separable_convolution_uses_expansion(self) -> None:
        using = {op for op in OperationType if op.uses_expansion_ratio}
        assert using == {OperationType.DW_SEP_CONV}


class TestBlockCanonicalisation:
    def test_identity_erases_every_inactive_field(self) -> None:
        block = BlockSpec(
            operation=OperationType.IDENTITY,
            kernel_size=5,
            expansion_ratio=4.0,
            stride=2,
            use_residual=True,
            normalization=NormalizationType.BATCH,
            activation=ActivationType.SILU,
        )
        assert block.kernel_size == 1
        assert block.expansion_ratio == 1.0
        assert block.stride == 1
        assert block.use_residual is False
        assert block.normalization is NormalizationType.NONE
        assert block.activation is ActivationType.IDENTITY

    def test_convolution_ignores_expansion_ratio(self) -> None:
        assert _conv_block(expansion_ratio=4.0).expansion_ratio == 1.0

    def test_pooling_has_no_normalisation_or_activation(self) -> None:
        block = BlockSpec(
            operation=OperationType.AVG_POOL,
            kernel_size=3,
            normalization=NormalizationType.GROUP,
            activation=ActivationType.GELU,
        )
        assert block.normalization is NormalizationType.NONE
        assert block.activation is ActivationType.IDENTITY

    def test_separable_convolution_keeps_its_expansion(self) -> None:
        block = BlockSpec(operation=OperationType.DW_SEP_CONV, expansion_ratio=3.0)
        assert block.expansion_ratio == 3.0

    def test_canonicalisation_is_idempotent(self) -> None:
        block = BlockSpec(operation=OperationType.IDENTITY, kernel_size=5, stride=2)
        assert block.evolve() == block

    def test_two_specs_differing_only_in_dead_fields_are_equal(self) -> None:
        first = BlockSpec(operation=OperationType.MAX_POOL, kernel_size=3, expansion_ratio=2.0)
        second = BlockSpec(operation=OperationType.MAX_POOL, kernel_size=3, expansion_ratio=8.0)
        assert first == second

    def test_even_kernel_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="kernel_size"):
            _conv_block(kernel_size=2)

    def test_unsupported_kernel_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _conv_block(kernel_size=9)

    def test_out_of_range_fields_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _conv_block(out_channels=0)
        with pytest.raises(ValidationError):
            _conv_block(stride=0)

    def test_unknown_fields_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Extra inputs"):
            BlockSpec(operation=OperationType.CONV, nonsense=1)  # type: ignore[call-arg]

    def test_blocks_are_immutable(self) -> None:
        block = _conv_block()
        with pytest.raises(ValidationError):
            block.kernel_size = 5  # type: ignore[misc]

    def test_evolve_reapplies_canonicalisation(self) -> None:
        block = BlockSpec(operation=OperationType.DW_SEP_CONV, expansion_ratio=4.0)
        converted = block.evolve(operation=OperationType.MAX_POOL)
        assert converted.expansion_ratio == 1.0
        assert converted.normalization is NormalizationType.NONE

    def test_describe_summarises_the_block(self) -> None:
        text = _conv_block(use_residual=True).describe()
        assert "conv" in text
        assert "res" in text
        assert BlockSpec(operation=OperationType.IDENTITY).describe() == "identity"


class TestHeadCanonicalisation:
    def test_dropout_is_quantised(self) -> None:
        assert HeadSpec(dropout=0.1 + 0.2).dropout == 0.3

    def test_activation_is_neutralised_without_a_hidden_layer(self) -> None:
        assert HeadSpec(hidden_units=0, activation=ActivationType.GELU).activation is (
            ActivationType.IDENTITY
        )

    def test_activation_survives_with_a_hidden_layer(self) -> None:
        assert HeadSpec(hidden_units=32, activation=ActivationType.GELU).activation is (
            ActivationType.GELU
        )

    def test_quantise_normalises_negative_zero(self) -> None:
        assert quantise(-0.0) == 0.0


class TestArchitectureSpec:
    def test_derived_properties(self, manual_spec: ArchitectureSpec) -> None:
        assert manual_spec.num_stages == 2
        assert manual_spec.total_blocks == 3
        assert manual_spec.total_stride == 2
        assert manual_spec.final_channels == 16

    def test_iter_blocks_is_in_execution_order(self, manual_spec: ArchitectureSpec) -> None:
        positions = [(stage, block) for stage, block, _ in manual_spec.iter_blocks()]
        assert positions == [(0, 0), (0, 1), (1, 0)]

    def test_with_block_leaves_the_original_untouched(self, manual_spec: ArchitectureSpec) -> None:
        before = architecture_hash(manual_spec)
        replacement = _conv_block(kernel_size=5, out_channels=16)
        child = manual_spec.with_block(0, 0, replacement)
        assert architecture_hash(manual_spec) == before
        assert architecture_hash(child) != before

    def test_with_block_validates_indices(self, manual_spec: ArchitectureSpec) -> None:
        with pytest.raises(IndexError, match="stage_index"):
            manual_spec.with_block(9, 0, _conv_block())
        with pytest.raises(IndexError, match="block_index"):
            manual_spec.with_block(0, 9, _conv_block())

    def test_rejects_a_future_schema_version(self) -> None:
        with pytest.raises(ValidationError, match="newer than the supported version"):
            ArchitectureSpec(
                schema_version=ARCHITECTURE_SCHEMA_VERSION + 1,
                stages=(StageSpec(blocks=(_conv_block(),)),),
            )

    def test_rejects_a_zero_schema_version(self) -> None:
        with pytest.raises(ValidationError, match="must be >= 1"):
            ArchitectureSpec(schema_version=0, stages=(StageSpec(blocks=(_conv_block(),)),))

    def test_requires_at_least_one_stage(self) -> None:
        with pytest.raises(ValidationError):
            ArchitectureSpec(stages=())

    def test_requires_at_least_one_block_per_stage(self) -> None:
        with pytest.raises(ValidationError):
            StageSpec(blocks=())

    def test_stem_rejects_even_kernels(self) -> None:
        with pytest.raises(ValidationError, match="stem kernel_size"):
            StemSpec(kernel_size=4)


class TestCanonicalSerialisation:
    def test_round_trip_preserves_equality(self, manual_spec: ArchitectureSpec) -> None:
        assert from_canonical_json(to_canonical_json(manual_spec)) == manual_spec

    def test_round_trip_preserves_the_hash(self, manual_spec: ArchitectureSpec) -> None:
        restored = from_canonical_dict(to_canonical_dict(manual_spec))
        assert architecture_hash(restored) == architecture_hash(manual_spec)

    def test_canonical_json_sorts_keys(self, manual_spec: ArchitectureSpec) -> None:
        text = to_canonical_json(manual_spec)
        assert text.startswith('{"head":')

    def test_canonical_json_is_ascii_and_compact(self, manual_spec: ArchitectureSpec) -> None:
        text = to_canonical_json(manual_spec)
        assert text.isascii()
        assert ", " not in text

    def test_equality_helper_matches_hash_equality(self, manual_spec: ArchitectureSpec) -> None:
        other = manual_spec.with_block(0, 0, _conv_block(kernel_size=5))
        assert architectures_equal(manual_spec, manual_spec)
        assert not architectures_equal(manual_spec, other)

    def test_rejects_non_object_payloads(self) -> None:
        with pytest.raises(ArchitectureValidationError, match="must be a JSON object"):
            from_canonical_dict([1, 2, 3])

    def test_rejects_unknown_operations(self, manual_spec: ArchitectureSpec) -> None:
        payload = to_canonical_dict(manual_spec)
        payload["stages"][0]["blocks"][0]["operation"] = "__import__"
        with pytest.raises(ArchitectureValidationError, match="failed validation"):
            from_canonical_dict(payload)

    def test_rejects_unknown_fields(self, manual_spec: ArchitectureSpec) -> None:
        payload = to_canonical_dict(manual_spec)
        payload["evil"] = True
        with pytest.raises(ArchitectureValidationError, match="failed validation"):
            from_canonical_dict(payload)

    def test_reports_the_offending_field(self, manual_spec: ArchitectureSpec) -> None:
        payload = to_canonical_dict(manual_spec)
        payload["num_classes"] = 1
        with pytest.raises(ArchitectureValidationError) as excinfo:
            from_canonical_dict(payload)
        assert "num_classes" in str(excinfo.value)

    def test_rejects_malformed_documents(self) -> None:
        with pytest.raises(ArchitectureValidationError, match="could not be parsed"):
            from_canonical_json("{not json")


class TestHashing:
    def test_hash_has_the_documented_length(self, manual_spec: ArchitectureSpec) -> None:
        assert len(architecture_hash(manual_spec)) == ARCHITECTURE_HASH_LENGTH

    def test_hash_is_lowercase_hexadecimal(self, manual_spec: ArchitectureSpec) -> None:
        digest = architecture_hash(manual_spec)
        assert digest == digest.lower()
        int(digest, 16)

    def test_equal_specifications_hash_identically(self, manual_spec: ArchitectureSpec) -> None:
        clone = from_canonical_dict(to_canonical_dict(manual_spec))
        assert architecture_hash(clone) == architecture_hash(manual_spec)

    def test_short_hash_truncates(self, manual_spec: ArchitectureSpec) -> None:
        digest = architecture_hash(manual_spec)
        assert short_hash(digest, length=8) == digest[:8]

    def test_short_hash_rejects_bad_lengths(self, manual_spec: ArchitectureSpec) -> None:
        digest = architecture_hash(manual_spec)
        with pytest.raises(ValueError, match="short hash length"):
            short_hash(digest, length=2)
        with pytest.raises(ValueError, match="short hash length"):
            short_hash(digest, length=99)


class TestShapeArithmetic:
    @pytest.mark.parametrize(
        ("size", "kernel", "stride", "expected"),
        [(32, 3, 1, 32), (32, 3, 2, 16), (32, 5, 2, 16), (31, 3, 2, 16), (1, 3, 2, 1)],
    )
    def test_same_padding_matches_ceiling_division(
        self, size: int, kernel: int, stride: int, expected: int
    ) -> None:
        assert conv_output_size(size, kernel, stride) == expected

    def test_rejects_non_positive_arguments(self) -> None:
        with pytest.raises(ValueError, match="positive arguments"):
            conv_output_size(0, 3, 1)

    @pytest.mark.parametrize(
        ("value", "expected"), [(8, 8), (12, 16), (3, 8), (1, 8), (33, 32), (36, 40)]
    )
    def test_make_divisible_rounds_to_multiples_of_eight(self, value: float, expected: int) -> None:
        assert make_divisible(value) == expected

    def test_make_divisible_never_rounds_down_more_than_ten_percent(self) -> None:
        assert make_divisible(100) >= 90

    def test_make_divisible_rejects_a_bad_divisor(self) -> None:
        with pytest.raises(ValueError, match="divisor must be positive"):
            make_divisible(16, divisor=0)

    def test_tensor_shape_helpers(self) -> None:
        shape = TensorShape(3, 4, 5)
        assert shape.elements == 60
        assert shape.as_tuple() == (3, 4, 5)
        assert str(shape) == "3x4x5"


class TestShapeInference:
    def test_traces_every_layer(self, manual_spec: ArchitectureSpec) -> None:
        trace = infer_shapes(manual_spec)
        names = [layer.name for layer in trace.layers]
        assert names[0] == "stem"
        assert "stages.0.blocks.0" in names
        assert names[-1] == "head.classifier"

    def test_reports_the_final_feature_map(self, manual_spec: ArchitectureSpec) -> None:
        trace = infer_shapes(manual_spec)
        assert trace.features_shape.as_tuple() == (16, 8, 8)
        assert trace.pooled_features == 16
        assert trace.output_features == 4

    def test_rejects_channel_changing_pooling(self) -> None:
        spec = ArchitectureSpec(
            input_size=16,
            num_classes=4,
            stem=StemSpec(out_channels=8),
            stages=(
                StageSpec(blocks=(BlockSpec(operation=OperationType.MAX_POOL, out_channels=32),)),
            ),
        )
        with pytest.raises(ShapeInferenceError, match="cannot change the channel count"):
            infer_shapes(spec)

    def test_rejects_residual_across_a_channel_change(self) -> None:
        spec = ArchitectureSpec(
            input_size=16,
            num_classes=4,
            stem=StemSpec(out_channels=8),
            stages=(StageSpec(blocks=(_conv_block(out_channels=32, use_residual=True),)),),
        )
        with pytest.raises(ShapeInferenceError, match="identical shapes"):
            infer_shapes(spec)

    def test_rejects_residual_across_a_stride(self) -> None:
        spec = ArchitectureSpec(
            input_size=16,
            num_classes=4,
            stem=StemSpec(out_channels=16),
            stages=(
                StageSpec(blocks=(_conv_block(out_channels=16, stride=2, use_residual=True),)),
            ),
        )
        with pytest.raises(ShapeInferenceError, match="spatial sizes differ"):
            infer_shapes(spec)

    def test_error_names_the_offending_block(self) -> None:
        spec = ArchitectureSpec(
            input_size=16,
            num_classes=4,
            stem=StemSpec(out_channels=8),
            stages=(
                StageSpec(
                    blocks=(
                        _conv_block(out_channels=8),
                        BlockSpec(operation=OperationType.AVG_POOL, out_channels=64),
                    )
                ),
            ),
        )
        with pytest.raises(ShapeInferenceError) as excinfo:
            infer_shapes(spec)
        assert excinfo.value.details["location"] == "stages.0.blocks.1"

    def test_trace_rows_render_for_display(self, manual_spec: ArchitectureSpec) -> None:
        rows = infer_shapes(manual_spec).to_rows()
        assert all(len(row) == 4 for row in rows)


class TestCostModel:
    def test_convolution_parameter_formula(self) -> None:
        assert conv_parameters(3, 16, 3) == 3 * 16 * 9
        assert conv_parameters(3, 16, 3, bias=True) == 3 * 16 * 9 + 16
        assert conv_parameters(16, 16, 3, groups=16) == 16 * 9

    def test_convolution_mac_formula(self) -> None:
        assert conv_macs(3, 16, 3, 8, 8) == 3 * 9 * 16 * 64

    def test_batch_norm_owns_buffers(self) -> None:
        assert normalization_parameters(NormalizationType.BATCH, 16) == (32, 33)

    def test_group_norm_has_no_buffers(self) -> None:
        assert normalization_parameters(NormalizationType.GROUP, 16) == (32, 0)

    def test_absent_normalisation_costs_nothing(self) -> None:
        assert normalization_parameters(NormalizationType.NONE, 16) == (0, 0)

    def test_unit_expansion_skips_the_widening_convolution(self) -> None:
        block = BlockSpec(operation=OperationType.DW_SEP_CONV, expansion_ratio=1.0)
        assert separable_hidden_channels(block, 24) == 24

    def test_expansion_rounds_to_a_multiple_of_eight(self) -> None:
        block = BlockSpec(operation=OperationType.DW_SEP_CONV, expansion_ratio=2.0)
        assert separable_hidden_channels(block, 12) == 24

    def test_cost_is_positive_and_self_consistent(self, manual_spec: ArchitectureSpec) -> None:
        cost = compute_cost(manual_spec)
        assert cost.trainable_parameters > 0
        assert cost.total_parameters == (cost.trainable_parameters + cost.non_trainable_parameters)
        assert cost.multiply_accumulates > 0
        assert cost.depth == manual_spec.total_blocks

    def test_wider_stages_cost_more(self) -> None:
        narrow = ArchitectureSpec(
            input_size=16,
            num_classes=4,
            stem=StemSpec(out_channels=8),
            stages=(StageSpec(blocks=(_conv_block(out_channels=8),)),),
        )
        wide = narrow.with_block(0, 0, _conv_block(out_channels=64))
        assert compute_cost(wide).trainable_parameters > compute_cost(narrow).trainable_parameters

    def test_cost_serialises_to_plain_data(self, manual_spec: ArchitectureSpec) -> None:
        payload = compute_cost(manual_spec).to_dict()
        assert set(payload) >= {"trainable_parameters", "multiply_accumulates", "depth"}


class TestSummary:
    def test_compact_line_mentions_the_key_facts(self, manual_spec: ArchitectureSpec) -> None:
        text = summarise(manual_spec).compact()
        assert "stages" in text
        assert "params" in text

    def test_text_summary_includes_a_layer_table(self, manual_spec: ArchitectureSpec) -> None:
        text = summarise(manual_spec).to_text()
        assert "stem" in text
        assert "head.classifier" in text
        assert "block detail" in text

    def test_markdown_summary_is_a_table(self, manual_spec: ArchitectureSpec) -> None:
        markdown = summarise(manual_spec).to_markdown()
        assert markdown.startswith("**Architecture")
        assert "| Layer | Kind | Input | Output |" in markdown


class TestLineage:
    @staticmethod
    def _graph() -> LineageGraph:
        return LineageGraph.from_nodes(
            [
                LineageNode("root", "h0"),
                LineageNode("child", "h1", parent_id="root", mutation="kernel 3->5"),
                LineageNode("grandchild", "h2", parent_id="child", objective_value=0.9),
                LineageNode("orphan", "h3", parent_id="missing"),
            ]
        )

    def test_ancestry_runs_root_to_leaf(self) -> None:
        chain = self._graph().ancestry("grandchild")
        assert [node.candidate_id for node in chain.nodes] == ["root", "child", "grandchild"]
        assert chain.depth == 3
        assert not chain.truncated

    def test_ancestry_of_unknown_candidate_is_empty(self) -> None:
        chain = self._graph().ancestry("nope")
        assert chain.nodes == ()
        assert chain.truncated

    def test_missing_parent_truncates_rather_than_raising(self) -> None:
        chain = self._graph().ancestry("orphan")
        assert chain.truncated
        assert chain.depth == 1

    def test_cycles_terminate(self) -> None:
        graph = LineageGraph.from_nodes(
            [LineageNode("a", "h", parent_id="b"), LineageNode("b", "h", parent_id="a")]
        )
        chain = graph.ancestry("a")
        assert chain.truncated
        assert chain.depth <= 2

    def test_descendants_are_breadth_first(self) -> None:
        assert self._graph().descendants("root") == ["child", "grandchild"]

    def test_roots_include_nodes_with_missing_parents(self) -> None:
        assert self._graph().roots() == ["orphan", "root"]

    def test_statistics_describe_the_forest(self) -> None:
        statistics = self._graph().statistics()
        assert statistics["nodes"] == 4
        assert statistics["max_depth"] == 3
        assert statistics["mutated_nodes"] == 3

    def test_chain_renders_as_a_tree(self) -> None:
        text = self._graph().ancestry("grandchild").to_text()
        assert "kernel 3->5" in text
        assert "objective=0.9000" in text
