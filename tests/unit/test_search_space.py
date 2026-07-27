"""Unit tests for the search space, sampler, repair, mutation, and validation.

Covers: choice-set validation and de-duplication, cardinality estimation, seeded sampling,
duplicate avoidance, exhaustion behaviour, sampler checkpointing, structural repair,
mutation purity and closure, and the four validation layers.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from nas_engine.architectures.hashing import architecture_hash
from nas_engine.architectures.spec import ArchitectureSpec, BlockSpec, StageSpec, StemSpec
from nas_engine.architectures.types import (
    ActivationType,
    NormalizationType,
    OperationType,
    PoolingType,
)
from nas_engine.exceptions import MutationError, SearchSpaceError
from nas_engine.search_space.mutation import (
    DEFAULT_OPERATORS,
    MutationOperator,
    mutate_expansion_ratio,
    mutate_head,
    mutate_kernel_size,
    mutate_num_stages,
    mutate_operation,
    mutate_residual,
    mutate_stage_depth,
    mutate_stage_width,
    mutate_stem,
    mutate_stride,
)
from nas_engine.search_space.presets import PRESETS, get_preset
from nas_engine.search_space.repair import repair_architecture, stage_widths
from nas_engine.search_space.sampler import ArchitectureSampler
from nas_engine.search_space.space import (
    BlockChoices,
    HeadChoices,
    SearchSpace,
    SpaceConstraints,
    StemChoices,
)
from nas_engine.search_space.validation import (
    check_architecture,
    check_membership,
    validate_architecture,
)

pytestmark = pytest.mark.unit


class TestChoiceValidation:
    def test_duplicates_are_removed_to_avoid_biasing_sampling(self) -> None:
        choices = BlockChoices(kernel_sizes=(3, 3, 5))
        assert choices.kernel_sizes == (3, 5)

    def test_empty_choice_sets_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="at least one choice"):
            BlockChoices(kernel_sizes=())

    def test_even_kernels_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must be odd"):
            BlockChoices(kernel_sizes=(2, 4))

    def test_unsupported_kernels_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="outside the supported set"):
            BlockChoices(kernel_sizes=(9,))

    def test_non_positive_expansions_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must be positive"):
            BlockChoices(expansion_ratios=(0.0,))

    def test_head_dropouts_must_be_probabilities(self) -> None:
        with pytest.raises(ValidationError, match="must lie in"):
            HeadChoices(dropouts=(1.5,))

    def test_stem_strides_are_bounded(self) -> None:
        with pytest.raises(ValidationError, match="strides must be in"):
            StemChoices(strides=(9,))

    def test_parametric_operations_are_required(self) -> None:
        with pytest.raises(ValidationError, match="no parametric operation"):
            SearchSpace(
                block=BlockChoices(operations=(OperationType.MAX_POOL, OperationType.IDENTITY))
            )

    def test_stage_widths_are_sorted_and_deduplicated(self) -> None:
        assert SearchSpace(stage_channels=(64, 16, 64, 32)).stage_channels == (16, 32, 64)

    def test_infeasible_parameter_interval_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="exceeds"):
            SpaceConstraints(min_parameters=100, max_parameters=10)

    def test_stage_counts_are_bounded(self) -> None:
        with pytest.raises(ValidationError, match="num_stages"):
            SearchSpace(num_stages=(0,))
        with pytest.raises(ValidationError, match="num_stages"):
            SearchSpace(num_stages=(20,))

    def test_block_counts_are_bounded(self) -> None:
        with pytest.raises(ValidationError, match="blocks_per_stage"):
            SearchSpace(blocks_per_stage=(0,))


class TestSpaceIntrospection:
    def test_cardinality_grows_with_the_choice_sets(self) -> None:
        small = SearchSpace(num_stages=(1,), blocks_per_stage=(1,))
        large = SearchSpace(num_stages=(1, 2, 3), blocks_per_stage=(1, 2, 3))
        assert large.cardinality_upper_bound() > small.cardinality_upper_bound()

    def test_log_cardinality_matches_the_bound(self, default_space: SearchSpace) -> None:
        assert default_space.log10_cardinality() > 10

    def test_per_block_count_excludes_inactive_choices(self) -> None:
        space = SearchSpace(
            block=BlockChoices(
                operations=(OperationType.CONV, OperationType.IDENTITY),
                kernel_sizes=(3, 5),
                expansion_ratios=(1.0, 2.0),
                normalizations=(NormalizationType.BATCH, NormalizationType.GROUP),
                activations=(ActivationType.RELU,),
                allow_residual=True,
            ),
            stage_channels=(16,),
        )
        # conv: 2 kernels x 1 expansion (inactive) x 2 norms x 1 activation x 2 residual = 8
        # identity: every conditional field is inactive, so it contributes exactly 1
        assert space.per_block_choice_count() == 9

    def test_describe_mentions_the_key_settings(self, default_space: SearchSpace) -> None:
        text = default_space.describe()
        assert "default_cnn" in text
        assert "monotonic" in text

    def test_impossible_parameter_ceiling_is_reported_early(self) -> None:
        space = SearchSpace(constraints=SpaceConstraints(max_parameters=2))
        with pytest.raises(SearchSpaceError, match="no candidate can ever be feasible"):
            space.require_non_empty()

    def test_feasible_ceiling_passes(self, default_space: SearchSpace) -> None:
        default_space.require_non_empty()


class TestPresets:
    @pytest.mark.parametrize("name", sorted(PRESETS))
    def test_every_preset_is_valid_and_samplable(self, name: str) -> None:
        space = get_preset(name)
        space.require_non_empty()
        spec = ArchitectureSampler(space, seed=0).sample()
        validate_architecture(spec, space)

    def test_overrides_are_applied(self) -> None:
        space = get_preset("tiny_cnn", input_size=32, num_classes=7)
        assert space.input_size == 32
        assert space.num_classes == 7

    def test_unknown_preset_is_rejected(self) -> None:
        with pytest.raises(SearchSpaceError, match="unknown search-space preset"):
            get_preset("does_not_exist")


class TestSampler:
    def test_sampled_architectures_are_valid(self, tiny_space: SearchSpace) -> None:
        sampler = ArchitectureSampler(tiny_space, seed=1)
        for _ in range(20):
            validate_architecture(sampler.sample(), tiny_space)

    def test_sampling_is_reproducible(self, tiny_space: SearchSpace) -> None:
        first = [architecture_hash(ArchitectureSampler(tiny_space, seed=9).sample())]
        second = [architecture_hash(ArchitectureSampler(tiny_space, seed=9).sample())]
        assert first == second

    def test_different_seeds_explore_differently(self, tiny_space: SearchSpace) -> None:
        first = {
            architecture_hash(ArchitectureSampler(tiny_space, seed=1).sample()) for _ in range(1)
        }
        second = {
            architecture_hash(ArchitectureSampler(tiny_space, seed=2).sample()) for _ in range(1)
        }
        # Not a strict guarantee for one draw, so sample a handful instead.
        sampler_a = ArchitectureSampler(tiny_space, seed=1)
        sampler_b = ArchitectureSampler(tiny_space, seed=2)
        first = {architecture_hash(sampler_a.sample()) for _ in range(15)}
        second = {architecture_hash(sampler_b.sample()) for _ in range(15)}
        assert first != second

    def test_unique_sampling_avoids_known_hashes(self, tiny_space: SearchSpace) -> None:
        sampler = ArchitectureSampler(tiny_space, seed=3)
        seen: set[str] = set()
        for _ in range(25):
            spec = sampler.sample_unique(seen)
            assert spec is not None
            digest = architecture_hash(spec)
            assert digest not in seen
            seen.add(digest)

    def test_exhausted_space_returns_none(self, micro_space: SearchSpace) -> None:
        sampler = ArchitectureSampler(micro_space, seed=4)
        seen: set[str] = set()
        while True:
            spec = sampler.sample_unique(seen, max_attempts=60)
            if spec is None:
                break
            seen.add(architecture_hash(spec))
        assert 0 < len(seen) <= 4
        assert sampler.statistics.duplicates > 0

    def test_statistics_track_rejections(self) -> None:
        space = SearchSpace(constraints=SpaceConstraints(max_multiply_accumulates=1000))
        sampler = ArchitectureSampler(space, seed=5, max_attempts=25)
        with pytest.raises(SearchSpaceError, match="failed to sample a valid architecture"):
            sampler.sample()
        assert sampler.statistics.rejected > 0
        assert sampler.statistics.acceptance_rate == 0.0

    def test_state_round_trip_continues_the_stream(self, tiny_space: SearchSpace) -> None:
        original = ArchitectureSampler(tiny_space, seed=6)
        original.sample()
        state = original.state_dict()
        expected = architecture_hash(original.sample())

        restored = ArchitectureSampler(tiny_space, seed=999)
        restored.load_state_dict(state)
        assert architecture_hash(restored.sample()) == expected

    def test_state_restores_statistics(self, tiny_space: SearchSpace) -> None:
        original = ArchitectureSampler(tiny_space, seed=6)
        for _ in range(3):
            original.sample()
        restored = ArchitectureSampler(tiny_space, seed=0)
        restored.load_state_dict(original.state_dict())
        assert restored.statistics.accepted == original.statistics.accepted

    def test_rejects_unsupported_state_versions(self, tiny_space: SearchSpace) -> None:
        sampler = ArchitectureSampler(tiny_space, seed=6)
        with pytest.raises(SearchSpaceError, match="state version"):
            sampler.load_state_dict({"version": 99})

    def test_rejects_malformed_state(self, tiny_space: SearchSpace) -> None:
        sampler = ArchitectureSampler(tiny_space, seed=6)
        with pytest.raises(SearchSpaceError, match="could not be restored"):
            sampler.load_state_dict({"version": 1, "rng": {"nope": 1}})

    def test_constructor_validates_arguments(self, tiny_space: SearchSpace) -> None:
        with pytest.raises(ValueError, match="max_attempts"):
            ArchitectureSampler(tiny_space, seed=0, max_attempts=0)
        with pytest.raises(ValueError, match="residual_probability"):
            ArchitectureSampler(tiny_space, seed=0, residual_probability=2.0)

    def test_monotonic_widths_are_respected(self, default_space: SearchSpace) -> None:
        sampler = ArchitectureSampler(default_space, seed=7)
        for _ in range(20):
            widths = stage_widths(sampler.sample())
            assert list(widths) == sorted(widths)

    def test_only_the_first_block_of_a_stage_downsamples(self, default_space: SearchSpace) -> None:
        sampler = ArchitectureSampler(default_space, seed=8)
        for _ in range(20):
            spec = sampler.sample()
            for stage in spec.stages:
                assert all(block.stride == 1 for block in stage.blocks[1:])


class TestRepair:
    def test_valid_architecture_is_returned_unchanged(self, manual_spec: ArchitectureSpec) -> None:
        repaired, report = repair_architecture(manual_spec)
        assert repaired is manual_spec
        assert not report.changed
        assert report.describe() == "no repairs required"

    def test_channel_preserving_blocks_are_corrected(self) -> None:
        broken = ArchitectureSpec(
            input_size=16,
            num_classes=4,
            stem=StemSpec(out_channels=8),
            stages=(
                StageSpec(
                    blocks=(
                        BlockSpec(operation=OperationType.CONV, out_channels=16),
                        BlockSpec(operation=OperationType.MAX_POOL, out_channels=99),
                    )
                ),
            ),
        )
        repaired, report = repair_architecture(broken)
        assert repaired.stages[0].blocks[1].out_channels == 16
        assert report.changed
        assert "channels" in report.describe()

    def test_illegal_residuals_are_removed(self) -> None:
        broken = ArchitectureSpec(
            input_size=16,
            num_classes=4,
            stem=StemSpec(out_channels=8),
            stages=(
                StageSpec(
                    blocks=(
                        BlockSpec(operation=OperationType.CONV, out_channels=16, use_residual=True),
                    )
                ),
            ),
        )
        repaired, report = repair_architecture(broken)
        assert repaired.stages[0].blocks[0].use_residual is False
        assert report.residual_fixes

    def test_impossible_downsampling_is_relaxed(self) -> None:
        blocks = tuple(
            BlockSpec(operation=OperationType.CONV, out_channels=8, stride=2) for _ in range(6)
        )
        broken = ArchitectureSpec(
            input_size=8,
            num_classes=4,
            stem=StemSpec(out_channels=8),
            stages=(StageSpec(blocks=blocks),),
        )
        repaired, report = repair_architecture(broken)
        from nas_engine.architectures.shapes import infer_shapes

        assert infer_shapes(repaired).features_shape.height >= 1
        assert report.stride_fixes or not report.changed

    def test_repair_is_idempotent(self) -> None:
        broken = ArchitectureSpec(
            input_size=16,
            num_classes=4,
            stem=StemSpec(out_channels=8),
            stages=(
                StageSpec(
                    blocks=(
                        BlockSpec(operation=OperationType.CONV, out_channels=16),
                        BlockSpec(operation=OperationType.AVG_POOL, out_channels=99),
                    )
                ),
            ),
        )
        once, _ = repair_architecture(broken)
        twice, report = repair_architecture(once)
        assert architecture_hash(once) == architecture_hash(twice)
        assert not report.changed

    def test_target_widths_are_applied(self, manual_spec: ArchitectureSpec) -> None:
        repaired, _ = repair_architecture(manual_spec, target_widths=(8, 8))
        assert stage_widths(repaired) == (8, 8)

    def test_wrong_width_count_is_rejected(self, manual_spec: ArchitectureSpec) -> None:
        with pytest.raises(ValueError, match="target_widths has"):
            repair_architecture(manual_spec, target_widths=(8,))


class TestMutationOperators:
    @pytest.mark.parametrize(
        "operator",
        [
            mutate_operation,
            mutate_kernel_size,
            mutate_expansion_ratio,
            mutate_residual,
            mutate_stride,
            mutate_stage_width,
            mutate_stage_depth,
            mutate_num_stages,
            mutate_stem,
            mutate_head,
        ],
    )
    def test_operators_either_change_or_decline(
        self, operator: Any, default_space: SearchSpace
    ) -> None:
        import random

        sampler = ArchitectureSampler(default_space, seed=21)
        rng = random.Random(3)
        for _ in range(10):
            parent = sampler.sample()
            before = architecture_hash(parent)
            outcome = operator(parent, default_space, rng)
            assert architecture_hash(parent) == before, "operator mutated its parent"
            if outcome is None:
                continue
            child, description = outcome
            assert isinstance(description, str) and description
            assert isinstance(child, ArchitectureSpec)

    def test_operators_are_all_registered(self) -> None:
        names = {name for name, _ in DEFAULT_OPERATORS}
        assert {"operation", "kernel_size", "stage_width", "num_stages", "head"} <= names

    def test_residual_operator_respects_the_space_setting(self) -> None:
        import random

        space = SearchSpace(block=BlockChoices(allow_residual=False))
        spec = ArchitectureSampler(space, seed=1).sample()
        assert mutate_residual(spec, space, random.Random(0)) is None


class TestMutationOperator:
    def test_children_are_valid_members_of_the_space(self, default_space: SearchSpace) -> None:
        sampler = ArchitectureSampler(default_space, seed=11)
        mutator = MutationOperator(default_space, seed=12)
        parent = sampler.sample()
        for _ in range(20):
            result = mutator.mutate(parent)
            validate_architecture(result.child, default_space)
            parent = result.child

    def test_children_differ_from_their_parent(self, default_space: SearchSpace) -> None:
        mutator = MutationOperator(default_space, seed=13)
        parent = ArchitectureSampler(default_space, seed=14).sample()
        for _ in range(15):
            result = mutator.mutate(parent)
            assert architecture_hash(result.child) != result.parent_hash

    def test_parent_is_never_modified(self, default_space: SearchSpace) -> None:
        mutator = MutationOperator(default_space, seed=15)
        parent = ArchitectureSampler(default_space, seed=16).sample()
        before = architecture_hash(parent)
        for _ in range(10):
            mutator.mutate(parent)
        assert architecture_hash(parent) == before

    def test_mutation_is_reproducible(self, default_space: SearchSpace) -> None:
        parent = ArchitectureSampler(default_space, seed=17).sample()
        first = MutationOperator(default_space, seed=18).mutate(parent)
        second = MutationOperator(default_space, seed=18).mutate(parent)
        assert architecture_hash(first.child) == architecture_hash(second.child)
        assert first.operator == second.operator

    def test_degenerate_space_reports_failure(self) -> None:
        space = SearchSpace(
            num_stages=(1,),
            blocks_per_stage=(1,),
            stage_channels=(8,),
            stage_strides=(1,),
            block=BlockChoices(
                operations=(OperationType.CONV,),
                kernel_sizes=(3,),
                expansion_ratios=(1.0,),
                normalizations=(NormalizationType.BATCH,),
                activations=(ActivationType.RELU,),
                allow_residual=False,
            ),
            stem=StemChoices(
                out_channels=(8,),
                kernel_sizes=(3,),
                strides=(1,),
                normalizations=(NormalizationType.BATCH,),
                activations=(ActivationType.RELU,),
            ),
            head=HeadChoices(
                poolings=(PoolingType.AVG,),
                hidden_units=(0,),
                dropouts=(0.0,),
                activations=(ActivationType.RELU,),
            ),
        )
        parent = ArchitectureSampler(space, seed=19).sample()
        mutator = MutationOperator(space, seed=20, max_attempts=10)
        with pytest.raises(MutationError, match="no valid mutation found"):
            mutator.mutate(parent)
        assert mutator.statistics.failures == 1

    def test_state_round_trip_continues_the_stream(self, default_space: SearchSpace) -> None:
        parent = ArchitectureSampler(default_space, seed=22).sample()
        original = MutationOperator(default_space, seed=23)
        original.mutate(parent)
        state = original.state_dict()
        expected = architecture_hash(original.mutate(parent).child)

        restored = MutationOperator(default_space, seed=999)
        restored.load_state_dict(state)
        assert architecture_hash(restored.mutate(parent).child) == expected

    def test_rejects_unsupported_state_versions(self, default_space: SearchSpace) -> None:
        mutator = MutationOperator(default_space, seed=24)
        with pytest.raises(MutationError, match="state version"):
            mutator.load_state_dict({"version": 99})

    def test_constructor_validates_arguments(self, default_space: SearchSpace) -> None:
        with pytest.raises(ValueError, match="max_attempts"):
            MutationOperator(default_space, seed=0, max_attempts=0)
        with pytest.raises(ValueError, match="at least one mutation operator"):
            MutationOperator(default_space, seed=0, operators=())

    def test_statistics_record_operator_usage(self, default_space: SearchSpace) -> None:
        mutator = MutationOperator(default_space, seed=25)
        parent = ArchitectureSampler(default_space, seed=26).sample()
        for _ in range(10):
            mutator.mutate(parent)
        assert sum(mutator.statistics.by_operator.values()) == 10


class TestValidationLayers:
    def test_valid_architecture_passes_every_layer(
        self, tiny_space: SearchSpace, sample_spec: ArchitectureSpec
    ) -> None:
        report = check_architecture(sample_spec, tiny_space)
        assert report.is_valid
        assert report.summary() == "architecture is valid"
        assert report.trace is not None
        assert report.cost is not None

    def test_membership_detects_a_foreign_operation(self, tiny_space: SearchSpace) -> None:
        spec = ArchitectureSpec(
            input_channels=tiny_space.input_channels,
            input_size=tiny_space.input_size,
            num_classes=tiny_space.num_classes,
            stem=StemSpec(out_channels=8),
            stages=(
                StageSpec(blocks=(BlockSpec(operation=OperationType.AVG_POOL, out_channels=8),)),
            ),
        )
        issues = check_membership(spec, tiny_space)
        assert any("operation" in issue.location for issue in issues)

    def test_membership_detects_a_shape_mismatch(self, tiny_space: SearchSpace) -> None:
        spec = ArchitectureSpec(
            input_channels=tiny_space.input_channels,
            input_size=tiny_space.input_size + 8,
            num_classes=tiny_space.num_classes,
            stem=StemSpec(out_channels=8),
            stages=(StageSpec(blocks=(BlockSpec(operation=OperationType.CONV, out_channels=8),)),),
        )
        issues = check_membership(spec, tiny_space)
        assert any(issue.location == "input_size" for issue in issues)

    def test_membership_allows_narrowing_from_the_stem(self, default_space: SearchSpace) -> None:
        spec = ArchitectureSpec(
            input_channels=3,
            input_size=32,
            num_classes=10,
            stem=StemSpec(out_channels=32),
            stages=(
                StageSpec(blocks=(BlockSpec(operation=OperationType.CONV, out_channels=16),)),
                StageSpec(blocks=(BlockSpec(operation=OperationType.CONV, out_channels=32),)),
            ),
        )
        issues = [
            issue
            for issue in check_membership(spec, default_space)
            if "non-decreasing" in issue.message
        ]
        assert not issues

    def test_membership_rejects_decreasing_widths(self, default_space: SearchSpace) -> None:
        spec = ArchitectureSpec(
            input_channels=3,
            input_size=32,
            num_classes=10,
            stem=StemSpec(out_channels=16),
            stages=(
                StageSpec(blocks=(BlockSpec(operation=OperationType.CONV, out_channels=64),)),
                StageSpec(blocks=(BlockSpec(operation=OperationType.CONV, out_channels=16),)),
            ),
        )
        issues = check_membership(spec, default_space)
        assert any("non-decreasing" in issue.message for issue in issues)

    def test_constraint_violation_is_distinguished_from_invalidity(self) -> None:
        space = SearchSpace(
            input_size=16,
            num_classes=4,
            stage_channels=(16,),
            num_stages=(1,),
            blocks_per_stage=(1,),
            stem=StemChoices(out_channels=(16,)),
            constraints=SpaceConstraints(max_parameters=100),
        )
        spec = ArchitectureSpec(
            input_channels=3,
            input_size=16,
            num_classes=4,
            stem=StemSpec(out_channels=16),
            stages=(StageSpec(blocks=(BlockSpec(operation=OperationType.CONV, out_channels=16),)),),
            head=nas_head(),
        )
        report = check_architecture(spec, space)
        assert not report.is_valid
        assert report.only_constraint_violations

    def test_raise_if_invalid_reports_constraint_violations_distinctly(self) -> None:
        from nas_engine.exceptions import ConstraintViolationError

        space = SearchSpace(
            input_size=16,
            num_classes=4,
            stage_channels=(16,),
            num_stages=(1,),
            blocks_per_stage=(1,),
            stem=StemChoices(out_channels=(16,)),
            constraints=SpaceConstraints(max_parameters=100),
        )
        spec = ArchitectureSpec(
            input_channels=3,
            input_size=16,
            num_classes=4,
            stem=StemSpec(out_channels=16),
            stages=(StageSpec(blocks=(BlockSpec(operation=OperationType.CONV, out_channels=16),)),),
            head=nas_head(),
        )
        with pytest.raises(ConstraintViolationError):
            validate_architecture(spec, space)

    def test_semantic_failure_short_circuits_costing(self, tiny_space: SearchSpace) -> None:
        spec = ArchitectureSpec(
            input_channels=tiny_space.input_channels,
            input_size=tiny_space.input_size,
            num_classes=tiny_space.num_classes,
            stem=StemSpec(out_channels=8),
            stages=(
                StageSpec(blocks=(BlockSpec(operation=OperationType.MAX_POOL, out_channels=64),)),
            ),
        )
        report = check_architecture(spec, tiny_space)
        assert report.trace is None
        assert report.cost is None
        assert report.issues_of("semantic")

    def test_membership_can_be_disabled(self, tiny_space: SearchSpace) -> None:
        spec = ArchitectureSpec(
            input_channels=tiny_space.input_channels,
            input_size=tiny_space.input_size,
            num_classes=tiny_space.num_classes,
            stem=StemSpec(out_channels=8),
            stages=(
                StageSpec(blocks=(BlockSpec(operation=OperationType.AVG_POOL, out_channels=8),)),
            ),
        )
        assert check_architecture(spec, tiny_space, check_space_membership=False).is_valid

    def test_issue_serialises_to_plain_data(self, tiny_space: SearchSpace) -> None:
        spec = ArchitectureSpec(
            input_channels=tiny_space.input_channels,
            input_size=tiny_space.input_size + 8,
            num_classes=tiny_space.num_classes,
            stem=StemSpec(out_channels=8),
            stages=(StageSpec(blocks=(BlockSpec(operation=OperationType.CONV, out_channels=8),)),),
        )
        payload = check_architecture(spec, tiny_space).issues[0].to_dict()
        assert set(payload) == {"category", "location", "message", "received", "expected"}


def nas_head() -> Any:
    """Return a minimal head used by constraint tests."""
    from nas_engine.architectures.spec import HeadSpec

    return HeadSpec(hidden_units=0, dropout=0.0)
