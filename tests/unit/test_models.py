"""Unit tests for model construction.

Covers: layer factories, group-norm arithmetic, block assembly, residual legality, the
classifier head, weight initialisation policy, parameter counting, and the builder's
error handling.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

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
from nas_engine.exceptions import ModelBuildError, ShapeInferenceError
from nas_engine.models.blocks import NasBlock, SeparableConvBlock, build_operation
from nas_engine.models.builder import (
    ModelBuilder,
    build_model,
    count_parameters,
    state_dict_bytes,
    summarize_model,
)
from nas_engine.models.initialization import (
    initialize_weights,
    zero_initialize_residual_branches,
)
from nas_engine.models.operations import (
    GlobalPoolFlatten,
    build_activation,
    build_conv_bn_act,
    build_normalization,
    group_count,
)

pytestmark = pytest.mark.unit


class TestLayerFactories:
    @pytest.mark.parametrize("activation", list(ActivationType))
    def test_every_activation_builds_and_runs(self, activation: ActivationType) -> None:
        module = build_activation(activation)
        output = module(torch.randn(2, 3))
        assert output.shape == (2, 3)

    def test_activations_are_not_in_place(self) -> None:
        module = build_activation(ActivationType.RELU)
        inputs = torch.tensor([-1.0, 1.0])
        module(inputs)
        assert inputs.tolist() == [-1.0, 1.0]

    @pytest.mark.parametrize("normalization", list(NormalizationType))
    def test_every_normalisation_builds_and_runs(self, normalization: NormalizationType) -> None:
        module = build_normalization(normalization, 8)
        output = module(torch.randn(2, 8, 4, 4))
        assert output.shape == (2, 8, 4, 4)

    def test_absent_normalisation_is_an_identity(self) -> None:
        assert isinstance(build_normalization(NormalizationType.NONE, 8), nn.Identity)

    @pytest.mark.parametrize(
        ("channels", "expected"), [(8, 8), (16, 8), (12, 6), (6, 6), (7, 7), (1, 1), (5, 5)]
    )
    def test_group_count_divides_the_channel_count(self, channels: int, expected: int) -> None:
        groups = group_count(channels)
        assert groups == expected
        assert channels % groups == 0

    def test_group_count_rejects_non_positive_channels(self) -> None:
        with pytest.raises(ValueError, match="channels must be positive"):
            group_count(0)

    def test_convolution_omits_bias_when_normalised(self) -> None:
        block = build_conv_bn_act(3, 8, 3, 1, NormalizationType.BATCH, ActivationType.RELU)
        assert block[0].bias is None

    def test_convolution_keeps_bias_without_normalisation(self) -> None:
        block = build_conv_bn_act(3, 8, 3, 1, NormalizationType.NONE, ActivationType.RELU)
        assert block[0].bias is not None

    def test_global_pool_flattens_to_channels(self) -> None:
        module = GlobalPoolFlatten(PoolingType.MAX)
        assert module(torch.randn(2, 8, 5, 5)).shape == (2, 8)


class TestBlocks:
    def test_convolution_block_changes_shape_as_declared(self) -> None:
        block = NasBlock(BlockSpec(operation=OperationType.CONV, out_channels=16, stride=2), 8)
        assert block(torch.randn(2, 8, 16, 16)).shape == (2, 16, 8, 8)

    def test_identity_block_is_a_pass_through(self) -> None:
        block = NasBlock(BlockSpec(operation=OperationType.IDENTITY, out_channels=8), 8)
        inputs = torch.randn(2, 8, 4, 4)
        assert torch.equal(block(inputs), inputs)

    def test_pooling_preserves_channels(self) -> None:
        block = NasBlock(BlockSpec(operation=OperationType.MAX_POOL, out_channels=8, stride=2), 8)
        assert block(torch.randn(2, 8, 16, 16)).shape == (2, 8, 8, 8)

    def test_residual_adds_the_input(self) -> None:
        spec = BlockSpec(operation=OperationType.CONV, out_channels=8, stride=1, use_residual=True)
        block = NasBlock(spec, 8)
        # Zero every convolution weight so the main path contributes nothing; the output
        # must then equal the input exactly.
        for module in block.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.zeros_(module.weight)
        block.eval()
        inputs = torch.randn(2, 8, 4, 4)
        assert torch.allclose(block(inputs), inputs, atol=1e-5)

    def test_illegal_residual_is_rejected(self) -> None:
        spec = BlockSpec(operation=OperationType.CONV, out_channels=16, stride=1, use_residual=True)
        with pytest.raises(ModelBuildError, match="residual connection requires"):
            NasBlock(spec, 8)

    def test_separable_block_requires_the_right_operation(self) -> None:
        with pytest.raises(ModelBuildError, match="SeparableConvBlock requires"):
            SeparableConvBlock(BlockSpec(operation=OperationType.CONV), 8)

    def test_separable_block_skips_expansion_at_ratio_one(self) -> None:
        block = SeparableConvBlock(
            BlockSpec(operation=OperationType.DW_SEP_CONV, expansion_ratio=1.0, out_channels=8),
            8,
        )
        assert block.expand is None

    def test_separable_block_expands_above_ratio_one(self) -> None:
        block = SeparableConvBlock(
            BlockSpec(operation=OperationType.DW_SEP_CONV, expansion_ratio=4.0, out_channels=8),
            8,
        )
        assert block.expand is not None

    def test_separable_projection_has_no_activation(self) -> None:
        block = SeparableConvBlock(
            BlockSpec(operation=OperationType.DW_SEP_CONV, out_channels=8), 8
        )
        assert not any(isinstance(module, (nn.ReLU, nn.SiLU, nn.GELU)) for module in block.project)

    def test_average_pooling_excludes_padding_from_the_mean(self) -> None:
        module = build_operation(
            BlockSpec(operation=OperationType.AVG_POOL, kernel_size=3, out_channels=1), 1
        )
        assert isinstance(module, nn.AvgPool2d)
        assert module.count_include_pad is False

    def test_repr_shows_the_genotype(self) -> None:
        block = NasBlock(BlockSpec(operation=OperationType.CONV, out_channels=8), 8)
        assert "conv" in repr(block)


class TestInitialization:
    def test_convolutions_are_not_left_at_zero(self) -> None:
        model = build_model(_simple_spec())
        weights = [module.weight for module in model.modules() if isinstance(module, nn.Conv2d)]
        assert weights
        assert all(float(weight.abs().sum()) > 0 for weight in weights)

    def test_biases_start_at_zero(self) -> None:
        model = build_model(_simple_spec())
        for module in model.modules():
            if isinstance(module, nn.Linear) and module.bias is not None:
                assert float(module.bias.abs().sum()) == 0.0

    def test_residual_branches_start_as_identities(self) -> None:
        spec = ArchitectureSpec(
            input_size=16,
            num_classes=4,
            stem=StemSpec(out_channels=8),
            stages=(
                StageSpec(
                    blocks=(
                        BlockSpec(operation=OperationType.CONV, out_channels=8, use_residual=True),
                    )
                ),
            ),
        )
        model = ModelBuilder(zero_init_residual=True).build(spec)
        block = next(module for module in model.modules() if isinstance(module, NasBlock))
        norm = next(
            module for module in block.operation.modules() if isinstance(module, nn.BatchNorm2d)
        )
        assert float(norm.weight.abs().sum()) == 0.0

    def test_zero_init_can_be_disabled(self) -> None:
        spec = ArchitectureSpec(
            input_size=16,
            num_classes=4,
            stem=StemSpec(out_channels=8),
            stages=(
                StageSpec(
                    blocks=(
                        BlockSpec(operation=OperationType.CONV, out_channels=8, use_residual=True),
                    )
                ),
            ),
        )
        model = ModelBuilder(zero_init_residual=False).build(spec)
        block = next(module for module in model.modules() if isinstance(module, NasBlock))
        norm = next(
            module for module in block.operation.modules() if isinstance(module, nn.BatchNorm2d)
        )
        assert float(norm.weight.abs().sum()) > 0.0

    def test_zero_init_reports_how_many_branches_it_touched(self) -> None:
        model = build_model(_residual_spec())
        assert zero_initialize_residual_branches(model) == 1

    def test_initialisation_is_reproducible(self) -> None:
        from nas_engine.utilities.seeding import seed_everything

        spec = _simple_spec()
        seed_everything(5)
        first = build_model(spec).state_dict()
        seed_everything(5)
        second = build_model(spec).state_dict()
        assert all(torch.equal(first[key], second[key]) for key in first)

    def test_initialize_weights_accepts_a_bare_module(self) -> None:
        module = nn.Sequential(nn.Conv2d(3, 8, 3), nn.BatchNorm2d(8), nn.Linear(8, 4))
        initialize_weights(module, zero_init_residual=False)
        assert float(module[1].weight.mean()) == 1.0


class TestBuilder:
    def test_output_shape_matches_the_class_count(self, manual_spec: ArchitectureSpec) -> None:
        model = build_model(manual_spec)
        output = model(torch.randn(3, 3, 16, 16))
        assert output.shape == (3, manual_spec.num_classes)

    def test_model_carries_its_specification_and_trace(self, manual_spec: ArchitectureSpec) -> None:
        model = build_model(manual_spec)
        assert model.spec == manual_spec
        assert model.trace.output_features == manual_spec.num_classes

    def test_runtime_shapes_match_the_static_trace(self, manual_spec: ArchitectureSpec) -> None:
        model = build_model(manual_spec)
        shapes = dict(model.feature_shapes(torch.randn(2, 3, 16, 16)))
        assert shapes["stem"] == (2, manual_spec.stem.out_channels, 16, 16)
        assert shapes["stages.1"] == (2, 16, 8, 8)
        assert shapes["head"] == (2, manual_spec.num_classes)

    def test_analytic_cost_matches_the_measured_count(self, manual_spec: ArchitectureSpec) -> None:
        _, summary = ModelBuilder().build_and_summarize(manual_spec)
        assert summary.matches_analytic_estimate

    def test_summary_serialises_to_plain_data(self, manual_spec: ArchitectureSpec) -> None:
        payload = summarize_model(build_model(manual_spec)).to_dict()
        assert payload["matches_analytic_estimate"] is True
        assert int(payload["state_dict_bytes"]) > 0  # type: ignore[call-overload]

    def test_structurally_invalid_specs_fail_before_allocation(self) -> None:
        spec = ArchitectureSpec(
            input_size=16,
            num_classes=4,
            stem=StemSpec(out_channels=8),
            stages=(
                StageSpec(blocks=(BlockSpec(operation=OperationType.MAX_POOL, out_channels=64),)),
            ),
        )
        with pytest.raises(ShapeInferenceError):
            build_model(spec)

    def test_builder_can_skip_initialisation(self, manual_spec: ArchitectureSpec) -> None:
        model = ModelBuilder(initialize=False).build(manual_spec)
        assert isinstance(model, nn.Module)

    def test_device_placement_is_honoured(self, manual_spec: ArchitectureSpec) -> None:
        model = build_model(manual_spec, device="cpu")
        assert next(model.parameters()).device.type == "cpu"

    def test_dtype_conversion_is_honoured(self, manual_spec: ArchitectureSpec) -> None:
        model = ModelBuilder().build(manual_spec, dtype=torch.float64)
        assert next(model.parameters()).dtype == torch.float64

    def test_parameter_counting_separates_buffers(self, manual_spec: ArchitectureSpec) -> None:
        trainable, non_trainable = count_parameters(build_model(manual_spec))
        assert trainable > 0
        assert non_trainable > 0

    def test_state_dict_bytes_is_positive(self, manual_spec: ArchitectureSpec) -> None:
        assert state_dict_bytes(build_model(manual_spec)) > 0

    def test_two_builds_produce_identical_module_ordering(
        self, manual_spec: ArchitectureSpec
    ) -> None:
        first = list(build_model(manual_spec).state_dict())
        second = list(build_model(manual_spec).state_dict())
        assert first == second

    def test_gradients_flow_to_every_trainable_parameter(
        self, manual_spec: ArchitectureSpec
    ) -> None:
        model = build_model(manual_spec)
        output = model(torch.randn(4, 3, 16, 16))
        output.sum().backward()
        missing = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        assert not missing


def _simple_spec() -> ArchitectureSpec:
    """A minimal buildable architecture."""
    return ArchitectureSpec(
        input_size=16,
        num_classes=4,
        stem=StemSpec(out_channels=8),
        stages=(StageSpec(blocks=(BlockSpec(operation=OperationType.CONV, out_channels=8),)),),
        head=HeadSpec(hidden_units=8),
    )


def _residual_spec() -> ArchitectureSpec:
    """An architecture containing exactly one residual block."""
    return ArchitectureSpec(
        input_size=16,
        num_classes=4,
        stem=StemSpec(out_channels=8),
        stages=(
            StageSpec(
                blocks=(BlockSpec(operation=OperationType.CONV, out_channels=8, use_residual=True),)
            ),
        ),
    )
