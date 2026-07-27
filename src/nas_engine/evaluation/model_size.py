"""Serialised model size measurement.

Parameter count and on-disk size are related but not the same, and the difference matters
when the objective is deployment footprint:

* Buffers (BatchNorm running statistics) are saved but are not parameters.
* Integer buffers such as ``num_batches_tracked`` are 8 bytes, not 4.
* PyTorch's ``.pt`` format is a ZIP container, so there is a small constant overhead plus
  per-tensor metadata.

The measurement therefore serialises the real state dict rather than multiplying the
parameter count by four. Serialisation happens into an in-memory buffer by default, so no
temporary file is created and no cleanup can be missed.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass(frozen=True)
class ModelSizeMeasurement:
    """Serialised size of a model.

    Attributes:
        state_dict_bytes: Sum of tensor payload sizes.
        serialized_bytes: Size of the serialised container, including format overhead.
        parameter_bytes: Payload size of trainable parameters only.
        buffer_bytes: Payload size of buffers only.
    """

    state_dict_bytes: int
    serialized_bytes: int
    parameter_bytes: int
    buffer_bytes: int

    @property
    def overhead_bytes(self) -> int:
        """Bytes the container format adds on top of the tensor payload."""
        return max(0, self.serialized_bytes - self.state_dict_bytes)

    def to_metrics(self) -> dict[str, float]:
        """Return the subset used as optimisation metrics."""
        return {
            "model_size_bytes": float(self.serialized_bytes),
            "parameter_bytes": float(self.parameter_bytes),
        }

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-serialisable representation."""
        return {
            "state_dict_bytes": self.state_dict_bytes,
            "serialized_bytes": self.serialized_bytes,
            "parameter_bytes": self.parameter_bytes,
            "buffer_bytes": self.buffer_bytes,
            "overhead_bytes": self.overhead_bytes,
        }


def measure_model_size(model: nn.Module) -> ModelSizeMeasurement:
    """Measure a model's serialised size.

    Args:
        model: Model to measure.

    Returns:
        A :class:`ModelSizeMeasurement`.
    """
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )
    buffer_bytes = sum(buffer.numel() * buffer.element_size() for buffer in model.buffers())

    state_dict = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    payload_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in state_dict.values()
        if isinstance(tensor, torch.Tensor)
    )

    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    serialized = buffer.getbuffer().nbytes

    return ModelSizeMeasurement(
        state_dict_bytes=payload_bytes,
        serialized_bytes=serialized,
        parameter_bytes=parameter_bytes,
        buffer_bytes=buffer_bytes,
    )


def save_model_weights(model: nn.Module, path: Path) -> int:
    """Save a model's weights to ``path`` atomically and return the file size.

    Only the state dict is saved, never the module object. Pickling a module records its
    class path, which makes the file unloadable after any refactor and unsafe to load from
    an untrusted source.

    Args:
        model: Model whose weights are saved.
        path: Destination file.

    Returns:
        Size of the written file in bytes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(
        {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        temporary,
    )
    temporary.replace(path)
    return path.stat().st_size


__all__ = ["ModelSizeMeasurement", "measure_model_size", "save_model_weights"]
