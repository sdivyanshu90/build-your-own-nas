"""Model construction: genotype in, :class:`torch.nn.Module` out.

Guarantees the builder provides
-------------------------------
* **Fail before allocating.** Shapes are validated statically first, so a structurally
  impossible architecture raises a precise error naming the offending block instead of a
  ``RuntimeError`` from deep inside cuDNN.
* **No hidden global state.** The builder takes everything it needs as arguments and
  returns a fresh module. Two calls with the same specification and seed produce
  identical networks; nothing is cached or memoised behind the caller's back.
* **Deterministic module order.** Submodules are registered in genotype order, so
  ``state_dict()`` keys are stable and a checkpoint saved by one process loads in
  another.
* **Introspectable.** The returned module carries its specification, its shape trace, and
  its parameter counts, so debugging never requires re-deriving them.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from nas_engine.architectures.cost import ArchitectureCost, compute_cost
from nas_engine.architectures.hashing import architecture_hash
from nas_engine.architectures.shapes import ShapeTrace, infer_shapes
from nas_engine.architectures.spec import ArchitectureSpec
from nas_engine.exceptions import ModelBuildError, ShapeInferenceError
from nas_engine.models.blocks import ClassifierHead, NasBlock
from nas_engine.models.initialization import initialize_weights
from nas_engine.models.operations import GlobalPoolFlatten, build_conv_bn_act


@dataclass(frozen=True)
class ModelSummary:
    """Measured properties of a built model.

    These are *measured*, not predicted: they come from walking the real module tree.
    Comparing them against :class:`~nas_engine.architectures.cost.ArchitectureCost` is how
    the analytic model is kept honest.

    Attributes:
        architecture_hash: Hash of the specification that produced the model.
        trainable_parameters: Parameters with ``requires_grad=True``.
        non_trainable_parameters: Frozen parameters plus persistent buffers.
        module_count: Number of leaf modules.
        state_dict_bytes: Serialised size of the state dict in bytes.
        analytic_cost: The analytic prediction, for comparison.
        trace: Static shape trace.
    """

    architecture_hash: str
    trainable_parameters: int
    non_trainable_parameters: int
    module_count: int
    state_dict_bytes: int
    analytic_cost: ArchitectureCost
    trace: ShapeTrace

    @property
    def total_parameters(self) -> int:
        """Sum of trainable and non-trainable parameters."""
        return self.trainable_parameters + self.non_trainable_parameters

    @property
    def matches_analytic_estimate(self) -> bool:
        """Whether the measured trainable count equals the analytic prediction."""
        return self.trainable_parameters == self.analytic_cost.trainable_parameters

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable representation."""
        return {
            "architecture_hash": self.architecture_hash,
            "trainable_parameters": self.trainable_parameters,
            "non_trainable_parameters": self.non_trainable_parameters,
            "total_parameters": self.total_parameters,
            "module_count": self.module_count,
            "state_dict_bytes": self.state_dict_bytes,
            "analytic_cost": self.analytic_cost.to_dict(),
            "matches_analytic_estimate": self.matches_analytic_estimate,
        }


class NasNetwork(nn.Module):
    """The phenotype: a runnable network built from an :class:`ArchitectureSpec`.

    Attributes:
        spec: The genotype this network realises.
        trace: Static shape trace for the genotype.
        stem: Entry convolution.
        stages: One :class:`torch.nn.Sequential` per stage.
        head: Classifier head.
    """

    def __init__(self, spec: ArchitectureSpec, trace: ShapeTrace) -> None:
        """Build the network.

        Args:
            spec: Validated architecture specification.
            trace: Shape trace for ``spec``.

        Raises:
            ModelBuildError: If a submodule cannot be constructed.
        """
        super().__init__()
        self.spec = spec
        self.trace = trace

        self.stem = build_conv_bn_act(
            spec.input_channels,
            spec.stem.out_channels,
            spec.stem.kernel_size,
            spec.stem.stride,
            spec.stem.normalization,
            spec.stem.activation,
        )

        stages: list[nn.Module] = []
        current_channels = spec.stem.out_channels
        for stage in spec.stages:
            blocks: list[nn.Module] = []
            for block in stage.blocks:
                blocks.append(NasBlock(block, current_channels))
                current_channels = block.out_channels
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)

        self.head = ClassifierHead(
            current_channels,
            spec.num_classes,
            pooling_module=GlobalPoolFlatten(spec.head.pooling),
            hidden_units=spec.head.hidden_units,
            dropout=spec.head.dropout,
            activation=spec.head.activation,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            inputs: Batch of shape ``(batch, input_channels, input_size, input_size)``.

        Returns:
            Logits of shape ``(batch, num_classes)``.
        """
        features: torch.Tensor = self.stem(inputs)
        features = self.stages(features)
        logits: torch.Tensor = self.head(features)
        return logits

    def feature_shapes(self, inputs: torch.Tensor) -> list[tuple[str, tuple[int, ...]]]:
        """Return the runtime shape after each stage, for debugging.

        Unlike :attr:`trace`, which is computed statically, this executes the network and
        reports what actually happened. Disagreement between the two is a bug in the
        shape model and is checked by an integration test.

        Args:
            inputs: A batch to run.

        Returns:
            ``(name, shape)`` pairs including the batch dimension.
        """
        shapes: list[tuple[str, tuple[int, ...]]] = []
        with torch.no_grad():
            features = self.stem(inputs)
            shapes.append(("stem", tuple(features.shape)))
            for index, stage in enumerate(self.stages):
                features = stage(features)
                shapes.append((f"stages.{index}", tuple(features.shape)))
            logits = self.head(features)
            shapes.append(("head", tuple(logits.shape)))
        return shapes


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Count trainable and non-trainable parameters, including buffers.

    Buffers (BatchNorm running statistics) are counted as non-trainable because they are
    persisted in the state dict and therefore contribute to on-disk model size, even
    though the optimiser never touches them.

    Args:
        model: Module to inspect.

    Returns:
        A ``(trainable, non_trainable)`` tuple.
    """
    trainable = 0
    non_trainable = 0
    for parameter in model.parameters():
        if parameter.requires_grad:
            trainable += parameter.numel()
        else:
            non_trainable += parameter.numel()
    for buffer in model.buffers():
        non_trainable += buffer.numel()
    return trainable, non_trainable


def state_dict_bytes(model: nn.Module) -> int:
    """Return the total byte size of a model's state dict tensors.

    This is the *tensor payload* size, which is what dominates a checkpoint file. The
    real file is slightly larger because of the ZIP container and metadata; the measured
    on-disk size is recorded separately during evaluation.

    Args:
        model: Module to measure.

    Returns:
        Size in bytes.
    """
    total = 0
    for tensor in model.state_dict().values():
        if isinstance(tensor, torch.Tensor):
            total += tensor.numel() * tensor.element_size()
    return total


def summarize_model(model: NasNetwork) -> ModelSummary:
    """Measure a built model and pair the result with its analytic prediction.

    Args:
        model: Built network.

    Returns:
        A :class:`ModelSummary`.
    """
    trainable, non_trainable = count_parameters(model)
    leaf_modules = sum(1 for module in model.modules() if not list(module.children()))
    return ModelSummary(
        architecture_hash=architecture_hash(model.spec),
        trainable_parameters=trainable,
        non_trainable_parameters=non_trainable,
        module_count=leaf_modules,
        state_dict_bytes=state_dict_bytes(model),
        analytic_cost=compute_cost(model.spec, model.trace),
        trace=model.trace,
    )


class ModelBuilder:
    """Builds networks from specifications.

    A class rather than a bare function so that build-time policy (weight initialisation,
    device placement, dtype) is configured once and injected wherever models are needed —
    the evaluator receives a builder, not a hard-coded call. That is what makes the
    evaluator testable with a stub builder.

    Args:
        zero_init_residual: Whether to zero-initialise residual branches.
        initialize: Whether to apply the initialisation policy at all. Disable when the
            weights will immediately be overwritten from a checkpoint.
    """

    def __init__(self, *, zero_init_residual: bool = True, initialize: bool = True) -> None:
        self._zero_init_residual = zero_init_residual
        self._initialize = initialize

    def build(
        self,
        spec: ArchitectureSpec,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> NasNetwork:
        """Build a network from a specification.

        Args:
            spec: Architecture to build.
            device: Device to move the network to; ``None`` leaves it on CPU.
            dtype: Parameter dtype; ``None`` keeps the PyTorch default.

        Returns:
            The constructed network.

        Raises:
            ShapeInferenceError: If the architecture is structurally invalid.
            ModelBuildError: If a submodule cannot be constructed despite valid shapes.
        """
        try:
            trace = infer_shapes(spec)
        except ShapeInferenceError:
            # Re-raised unchanged: it already names the offending block and explains the
            # fix, and callers distinguish it from ModelBuildError.
            raise

        try:
            model = NasNetwork(spec, trace)
        except ModelBuildError:
            raise
        except Exception as exc:
            msg = (
                f"failed to construct model for architecture {architecture_hash(spec)}: "
                f"{type(exc).__name__}: {exc}"
            )
            raise ModelBuildError(
                msg,
                details={"architecture_hash": architecture_hash(spec), "error": str(exc)},
            ) from exc

        if self._initialize:
            initialize_weights(model, zero_init_residual=self._zero_init_residual)
        if dtype is not None:
            model = model.to(dtype=dtype)
        if device is not None:
            model = model.to(device=torch.device(device))
        return model

    def build_and_summarize(
        self,
        spec: ArchitectureSpec,
        *,
        device: torch.device | str | None = None,
    ) -> tuple[NasNetwork, ModelSummary]:
        """Build a network and measure it.

        Args:
            spec: Architecture to build.
            device: Device to move the network to.

        Returns:
            The network and its summary.
        """
        model = self.build(spec, device=device)
        return model, summarize_model(model)


def build_model(spec: ArchitectureSpec, *, device: torch.device | str | None = None) -> NasNetwork:
    """Build a network with default policy.

    Convenience wrapper around :class:`ModelBuilder` for scripts and the public API.

    Args:
        spec: Architecture to build.
        device: Device to move the network to.

    Returns:
        The constructed network.
    """
    return ModelBuilder().build(spec, device=device)


__all__ = [
    "ModelBuilder",
    "ModelSummary",
    "NasNetwork",
    "build_model",
    "count_parameters",
    "state_dict_bytes",
    "summarize_model",
]
