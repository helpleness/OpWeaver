"""Trusted registration example; never auto-loads third-party plugin paths."""

from torch import nn

from .registry import EmptyParams, OperatorPlugin, identity_shape


class Square(nn.Module):
    def forward(self, value):
        return value.square()


def register_example(registry):
    registry.register(
        OperatorPlugin(
            id="example.square",
            version="1",
            label="Square",
            category="Custom",
            inputs=("in",),
            outputs=("out",),
            parameters=EmptyParams,
            factory=lambda p: Square(),
            infer=identity_shape,
            description="Trusted differentiable square example.",
            python_module=lambda p: "Square()",
            module_type=Square,
            module_parameters=lambda module: {},
            python_imports=("from opweaver.custom_example import Square",),
        )
    )
