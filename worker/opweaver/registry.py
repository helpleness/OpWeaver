"""Versioned operator plugins with explicit capabilities and shape functions."""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from pydantic import Field
from torch import nn

from .ir import Dimension, Spec, TensorSpec


class EmptyParams(Spec):
    pass


class InputParams(Spec):
    shape: list[Dimension] = Field(default_factory=lambda: ["B", 8], min_length=1)
    dtype: str = "float32"
    device: str = "any"


class LinearParams(Spec):
    in_features: int = Field(default=8, gt=0, le=65536)
    out_features: int = Field(default=16, gt=0, le=65536)
    bias: bool = True


class DropoutParams(Spec):
    p: float = Field(default=0.1, ge=0, lt=1)


class FlattenParams(Spec):
    start_dim: int = Field(default=1, ge=0)
    end_dim: int = -1


class Add(nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b


def identity_shape(inputs: list[TensorSpec], _: Spec) -> list[TensorSpec]:
    return [inputs[0].model_copy(deep=True)]


def input_shape(_: list[TensorSpec], params: InputParams) -> list[TensorSpec]:
    return [TensorSpec(shape=params.shape, dtype=params.dtype, device=params.device)]


def linear_shape(inputs: list[TensorSpec], params: LinearParams) -> list[TensorSpec]:
    value = inputs[0]
    if not value.shape or value.shape[-1] != params.in_features:
        raise ValueError(f"Linear expects last dimension {params.in_features}, got {value.shape}")
    return [value.model_copy(update={"shape": [*value.shape[:-1], params.out_features]})]


def add_shape(inputs: list[TensorSpec], _: Spec) -> list[TensorSpec]:
    a, b = inputs
    if a.shape != b.shape or a.dtype != b.dtype:
        raise ValueError(
            f"Add requires identical shape/dtype; got {a.shape}/{a.dtype} and {b.shape}/{b.dtype}"
        )
    if a.device != b.device and "any" not in (a.device, b.device):
        raise ValueError("Add inputs must be on the same device")
    return [a.model_copy(update={"device": b.device if a.device == "any" else a.device})]


def flatten_shape(inputs: list[TensorSpec], params: FlattenParams) -> list[TensorSpec]:
    value = inputs[0]
    rank = len(value.shape)
    end = params.end_dim + rank if params.end_dim < 0 else params.end_dim
    if not 0 <= params.start_dim <= end < rank:
        raise ValueError("Flatten dimensions are outside the tensor rank")
    collapsed = value.shape[params.start_dim : end + 1]
    size = 1
    for dim in collapsed:
        if not isinstance(dim, int):
            raise ValueError(
                "Flatten cannot collapse a symbolic dimension in IR v1; preserve the batch axis"
            )
        size *= dim
    return [
        value.model_copy(
            update={"shape": value.shape[: params.start_dim] + [size] + value.shape[end + 1 :]}
        )
    ]


@dataclass(frozen=True)
class OperatorPlugin:
    id: str
    label: str
    category: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    parameters: type[Spec]
    factory: Callable[[Spec], nn.Module]
    infer: Callable[[list[TensorSpec], Spec], list[TensorSpec]]
    description: str
    version: str = "1"
    dtypes: tuple[str, ...] = ("float32",)
    devices: tuple[str, ...] = ("cpu", "cuda")
    autograd: bool = True
    torch_export: bool = False
    onnx: bool = False
    custom_kernel: bool = False
    python_module: Callable[[Spec], str] | None = None
    python_expression: Callable[[list[str], Spec], str] | None = None
    module_type: type[nn.Module] | None = None
    module_parameters: Callable[[nn.Module], dict] | None = None
    python_imports: tuple[str, ...] = ()

    def manifest(self) -> dict:
        return {
            "id": self.id,
            "version": self.version,
            "label": self.label,
            "category": self.category,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "defaults": self.parameters().model_dump(),
            "parameters": self.parameters.model_json_schema(),
            "description": self.description,
            "capabilities": {
                "dtypes": self.dtypes,
                "devices": self.devices,
                "autograd": self.autograd,
                "torch_export": self.torch_export,
                "onnx": self.onnx,
                "custom_kernel": self.custom_kernel,
                "backends": ["pytorch.eager"],
                "codegen": self.python_module is not None or self.python_expression is not None,
            },
        }


class OperatorRegistry:
    def __init__(self) -> None:
        self._operators: dict[tuple[str, str], OperatorPlugin] = {}

    def register(self, operator: OperatorPlugin) -> None:
        key = (operator.id, operator.version)
        if key in self._operators:
            raise ValueError(f"Operator already registered: {key}")
        self._operators[key] = operator

    def get(self, id: str, version: str = "1") -> OperatorPlugin:
        try:
            return self._operators[id, version]
        except KeyError as exc:
            raise ValueError(
                f"Missing operator {id}@{version}. Install a trusted compatible plugin."
            ) from exc

    def manifests(self) -> list[dict]:
        return [op.manifest() for op in self._operators.values()]

    def operators(self) -> tuple[OperatorPlugin, ...]:
        return tuple(self._operators.values())


def builtin_registry() -> OperatorRegistry:
    registry = OperatorRegistry()
    entries = [
        OperatorPlugin(
            "core.input",
            "Input",
            "Structure",
            (),
            ("out",),
            InputParams,
            lambda p: nn.Identity(),
            input_shape,
            "External tensor. B denotes the dynamic batch dimension.",
        ),
        OperatorPlugin(
            "core.output",
            "Output",
            "Structure",
            ("in",),
            (),
            EmptyParams,
            lambda p: nn.Identity(),
            lambda i, p: [],
            "Named model output, ordered by its position in the IR.",
        ),
        OperatorPlugin(
            "core.linear",
            "Linear",
            "Layers",
            ("in",),
            ("out",),
            LinearParams,
            lambda p: nn.Linear(p.in_features, p.out_features, bias=p.bias),
            linear_shape,
            "Learnable affine transform on the final axis.",
        ),
        OperatorPlugin(
            "core.relu",
            "ReLU",
            "Activation",
            ("in",),
            ("out",),
            EmptyParams,
            lambda p: nn.ReLU(),
            identity_shape,
            "Rectified linear activation.",
        ),
        OperatorPlugin(
            "core.gelu",
            "GELU",
            "Activation",
            ("in",),
            ("out",),
            EmptyParams,
            lambda p: nn.GELU(),
            identity_shape,
            "Gaussian error linear activation.",
        ),
        OperatorPlugin(
            "core.add",
            "Add",
            "Tensor",
            ("a", "b"),
            ("out",),
            EmptyParams,
            lambda p: Add(),
            add_shape,
            "Elementwise residual sum. IR v1 requires identical shapes.",
        ),
        OperatorPlugin(
            "core.flatten",
            "Flatten",
            "Tensor",
            ("in",),
            ("out",),
            FlattenParams,
            lambda p: nn.Flatten(p.start_dim, p.end_dim),
            flatten_shape,
            "Flatten contiguous dimensions; preserve symbolic batch axis.",
        ),
        OperatorPlugin(
            "core.dropout",
            "Dropout",
            "Layers",
            ("in",),
            ("out",),
            DropoutParams,
            lambda p: nn.Dropout(p.p),
            identity_shape,
            "Randomly zero activations during training only.",
        ),
    ]
    from .operators import add_codegen, extra_operators

    for entry in [*entries, *extra_operators()]:
        entry = add_codegen(entry)
        registry.register(entry)
    return registry
