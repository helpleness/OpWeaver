"""PyTorch construction, code emission and capture adapters share one registry."""

from dataclasses import replace
from typing import Literal

from pydantic import Field, model_validator
from torch import nn

from .ir import Spec, TensorSpec
from .registry import EmptyParams, OperatorPlugin, add_shape, identity_shape


def constructor(name: str, params: Spec) -> str:
    return (
        f"nn.{name}("
        + ", ".join(f"{key}={value!r}" for key, value in params.model_dump().items())
        + ")"
    )


class Conv2dParams(Spec):
    in_channels: int = Field(default=3, gt=0)
    out_channels: int = Field(default=8, gt=0)
    kernel_size: int | list[int] = 3
    stride: int | list[int] = 1
    padding: int | list[int] = 0
    dilation: int | list[int] = 1
    groups: int = Field(default=1, gt=0)
    bias: bool = True
    padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros"

    @model_validator(mode="after")
    def dimensions(self):
        for name in ("kernel_size", "stride", "padding", "dilation"):
            values = pair(getattr(self, name))
            if any(v < (0 if name == "padding" else 1) for v in values):
                raise ValueError(f"Invalid Conv2d {name}")
        if self.in_channels % self.groups or self.out_channels % self.groups:
            raise ValueError("Conv2d channels must be divisible by groups")
        return self


def pair(value):
    if isinstance(value, int):
        return [value, value]
    if len(value) != 2 or any(not isinstance(v, int) for v in value):
        raise ValueError("Expected an integer or a pair of integers")
    return value


class BatchNormParams(Spec):
    num_features: int = Field(default=16, gt=0)
    eps: float = Field(default=1e-5, gt=0)
    momentum: float | None = 0.1
    affine: bool = True
    track_running_stats: bool = True


class LayerNormParams(Spec):
    normalized_shape: list[int] = Field(default_factory=lambda: [16], min_length=1)
    eps: float = Field(default=1e-5, gt=0)
    elementwise_affine: bool = True
    bias: bool = True

    @model_validator(mode="after")
    def positive_shape(self):
        if any(value < 1 for value in self.normalized_shape):
            raise ValueError("LayerNorm dimensions must be positive")
        return self


class ConcatParams(Spec):
    dim: int = 1


class Concat(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, a, b):
        import torch

        return torch.cat((a, b), dim=self.dim)


class Multiply(nn.Module):
    def forward(self, a, b):
        return a * b


def conv_shape(inputs: list[TensorSpec], params: Conv2dParams):
    value = inputs[0]
    if len(value.shape) != 4 or value.shape[1] != params.in_channels:
        raise ValueError("Conv2d expects [B, in_channels, height, width]")
    spatial = []
    for size, kernel, stride, pad, dilation in zip(
        value.shape[2:],
        pair(params.kernel_size),
        pair(params.stride),
        pair(params.padding),
        pair(params.dilation),
    ):
        if not isinstance(size, int):
            raise ValueError("Conv2d spatial dimensions must be concrete in Design Graph v1")
        out = (size + 2 * pad - dilation * (kernel - 1) - 1) // stride + 1
        if out < 1:
            raise ValueError("Conv2d produces a non-positive output dimension")
        spatial.append(out)
    return [value.model_copy(update={"shape": [value.shape[0], params.out_channels, *spatial]})]


def norm_shape(inputs, params, rank=None):
    value = inputs[0]
    if (
        len(value.shape) < 2
        or (rank and len(value.shape) not in rank)
        or value.shape[1] != params.num_features
    ):
        raise ValueError("BatchNorm rank/channels do not match num_features")
    return identity_shape(inputs, params)


def layer_norm_shape(inputs, params):
    if inputs[0].shape[-len(params.normalized_shape) :] != params.normalized_shape:
        raise ValueError("LayerNorm normalized_shape must match the trailing dimensions")
    return identity_shape(inputs, params)


def concat_shape(inputs, params):
    a, b = inputs
    rank = len(a.shape)
    dim = params.dim + rank if params.dim < 0 else params.dim
    if len(b.shape) != rank or not 0 <= dim < rank or a.dtype != b.dtype:
        raise ValueError("Concat rank, dtype or axis mismatch")
    if any(x != y for i, (x, y) in enumerate(zip(a.shape, b.shape)) if i != dim):
        raise ValueError("Concat shapes must match outside the concatenation axis")
    if not isinstance(a.shape[dim], int) or not isinstance(b.shape[dim], int):
        raise ValueError("Concat axis must have concrete sizes")
    if a.device != b.device and "any" not in (a.device, b.device):
        raise ValueError("Concat input devices differ")
    shape = list(a.shape)
    shape[dim] += b.shape[dim]
    return [a.model_copy(update={"shape": shape})]


def extra_operators():
    entries = [
        OperatorPlugin(
            "core.conv2d",
            "Conv2d",
            "Layers",
            ("in",),
            ("out",),
            Conv2dParams,
            lambda p: nn.Conv2d(**p.model_dump()),
            conv_shape,
            "Two-dimensional convolution.",
        ),
        OperatorPlugin(
            "core.layernorm",
            "LayerNorm",
            "Normalization",
            ("in",),
            ("out",),
            LayerNormParams,
            lambda p: nn.LayerNorm(**p.model_dump()),
            layer_norm_shape,
            "Normalize trailing dimensions.",
        ),
        OperatorPlugin(
            "core.concat",
            "Concat",
            "Tensor",
            ("a", "b"),
            ("out",),
            ConcatParams,
            lambda p: Concat(p.dim),
            concat_shape,
            "Concatenate two tensors along an axis.",
        ),
        OperatorPlugin(
            "core.multiply",
            "Multiply",
            "Tensor",
            ("a", "b"),
            ("out",),
            EmptyParams,
            lambda p: Multiply(),
            add_shape,
            "Elementwise multiplication with identical shapes.",
        ),
    ]
    for name, module, ranks in [
        ("1d", nn.BatchNorm1d, (2, 3)),
        ("2d", nn.BatchNorm2d, (4,)),
        ("3d", nn.BatchNorm3d, (5,)),
    ]:
        entries.append(
            OperatorPlugin(
                f"core.batchnorm{name}",
                f"BatchNorm{name}",
                "Normalization",
                ("in",),
                ("out",),
                BatchNormParams,
                lambda p, module=module: module(**p.model_dump()),
                lambda i, p, ranks=ranks: norm_shape(i, p, ranks),
                "Batch normalization with running state.",
            )
        )
    return entries


def module_params(module: nn.Module, fields: tuple[str, ...]):
    result = {}
    for key in fields:
        value = getattr(module, key)
        result[key] = list(value) if isinstance(value, tuple) else value
    return result


def add_codegen(op: OperatorPlugin):
    descriptors = {
        "core.linear": (nn.Linear, ("in_features", "out_features")),
        "core.relu": (nn.ReLU, ()),
        "core.gelu": (nn.GELU, ()),
        "core.dropout": (nn.Dropout, ("p",)),
        "core.flatten": (nn.Flatten, ("start_dim", "end_dim")),
        "core.conv2d": (nn.Conv2d, tuple(Conv2dParams.model_fields)),
        "core.layernorm": (nn.LayerNorm, tuple(LayerNormParams.model_fields)),
        "core.batchnorm1d": (nn.BatchNorm1d, tuple(BatchNormParams.model_fields)),
        "core.batchnorm2d": (nn.BatchNorm2d, tuple(BatchNormParams.model_fields)),
        "core.batchnorm3d": (nn.BatchNorm3d, tuple(BatchNormParams.model_fields)),
    }
    if op.id in descriptors:
        module, fields = descriptors[op.id]

        def extract(value):
            params = module_params(value, fields)
            if op.id == "core.linear":
                params["bias"] = value.bias is not None
            elif op.id == "core.conv2d":
                params["bias"] = value.bias is not None
            elif op.id == "core.layernorm":
                params["bias"] = value.bias is not None
            return params

        return replace(
            op,
            python_module=lambda p: constructor(module.__name__, p),
            module_type=module,
            module_parameters=extract,
        )
    expressions = {
        "core.add": lambda args, p: f"({args[0]} + {args[1]})",
        "core.multiply": lambda args, p: f"({args[0]} * {args[1]})",
        "core.concat": lambda args, p: f"torch.cat(({args[0]}, {args[1]}), dim={p.dim})",
    }
    if op.id in expressions:
        return replace(op, python_expression=expressions[op.id])
    return op
