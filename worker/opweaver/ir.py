"""Versioned semantic IR. Canvas positions never enter this representation."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Spec(BaseModel):
    model_config = ConfigDict(extra="forbid")


Dimension = Annotated[int, Field(gt=0)] | Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*$")]


class TensorSpec(Spec):
    shape: list[Dimension]
    dtype: Literal["float32", "float64", "float16", "bfloat16", "int64", "bool"] = "float32"
    device: Literal["cpu", "cuda", "any"] = "any"


class Port(Spec):
    name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
    tensor: TensorSpec | None = None


class PluginReference(Spec):
    id: str
    version: str
    kind: Literal["operator", "model", "training", "optimization", "kernel", "backend"] = "operator"


class OperatorSpec(Spec):
    id: str
    version: str = "1"


class Node(Spec):
    id: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_-]*$")
    label: str = ""
    operator: OperatorSpec
    params: dict[str, Any] = Field(default_factory=dict)
    inputs: list[Port] = Field(default_factory=list)
    outputs: list[Port] = Field(default_factory=list)
    share_with: str | None = None
    module_ref: str | None = None


class Edge(Spec):
    id: str
    source: str
    source_port: str = "out"
    target: str
    target_port: str = "in"


class ModuleSpec(Spec):
    id: str
    version: str
    graph_ref: str
    inputs: list[Port]
    outputs: list[Port]


class GraphMetadata(Spec):
    name: str = "Untitled model"
    model_version: str = "0.1.0"
    description: str = ""


class Graph(Spec):
    ir_version: Literal["1.0"] = "1.0"
    id: str
    metadata: GraphMetadata = Field(default_factory=GraphMetadata)
    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)
    modules: list[ModuleSpec] = Field(default_factory=list)
    plugins: list[PluginReference] = Field(default_factory=list)
