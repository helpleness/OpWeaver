"""Static inspection and precise source edits. Importing a file is never implicit."""

import ast
import difflib
import hashlib
import json
from typing import Literal

from pydantic import Field

from .ir import Edge, Spec


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_source(path) -> str:
    with open(path, encoding="utf-8", newline="") as file:
        return file.read()


def parse_ast(source: str, filename="<source>"):
    return ast.parse(source.removeprefix("\ufeff"), filename=filename)


class SourceMapping(Spec):
    file_path: str
    start_line: int | None = None
    end_line: int | None = None
    module_path: str = ""
    graph_node_id: str
    status: Literal["exact", "inferred", "unmapped"] = "unmapped"
    reason: str = ""


class AnalysisNode(Spec):
    id: str
    label: str
    operator: str
    kind: str
    module_path: str = ""
    parameters: dict = Field(default_factory=dict)
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=lambda: ["out"])
    tensors: dict = Field(default_factory=dict)
    details: dict = Field(default_factory=dict)


class AnalysisGraph(Spec):
    schema_version: Literal["1.0"] = "1.0"
    origin: Literal["pytorch.fx", "onnx"] = "pytorch.fx"
    id: str
    name: str
    source_hash: str
    nodes: list[AnalysisNode]
    edges: list[Edge]
    read_only: bool = True
    diagnostics: list[dict] = Field(default_factory=list)


def inspect_source(source: str, file_path="model.py") -> dict:
    try:
        tree = parse_ast(source, file_path)
    except SyntaxError as exc:
        return {
            "hash": digest(source),
            "entries": [],
            "diagnostics": [
                {"severity": "error", "message": exc.msg, "line": exc.lineno, "column": exc.offset}
            ],
        }
    entries = []
    for item in tree.body:
        if isinstance(item, ast.ClassDef) and any(
            isinstance(child, ast.FunctionDef) and child.name == "forward" for child in item.body
        ):
            init = next(
                (
                    child
                    for child in item.body
                    if isinstance(child, ast.FunctionDef) and child.name == "__init__"
                ),
                None,
            )
            entries.append(
                {
                    "name": item.name,
                    "kind": "class",
                    "line": item.lineno,
                    "signature": ast.unparse(init.args) if init else "",
                }
            )
        elif isinstance(item, ast.FunctionDef):
            entries.append(
                {
                    "name": item.name,
                    "kind": "factory",
                    "line": item.lineno,
                    "signature": ast.unparse(item.args),
                }
            )
    return {"hash": digest(source), "entries": entries, "diagnostics": []}


def attribute_path(value) -> str | None:
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Attribute):
        parent = attribute_path(value.value)
        return f"{parent}.{value.attr}" if parent else None
    return None


def module_definitions(source: str, class_name: str) -> dict[str, ast.Call]:
    tree = parse_ast(source)
    cls = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name), None)
    if cls is None:
        return {}
    init = next(
        (n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None
    )
    definitions = {}
    if init:
        for assignment in init.body:
            if (
                isinstance(assignment, ast.Assign)
                and len(assignment.targets) == 1
                and isinstance(assignment.value, ast.Call)
            ):
                path = attribute_path(assignment.targets[0])
                if path and path.startswith("self."):
                    definitions[path[5:]] = assignment.value
        # Reassignments, conditional construction and aliases cannot be mapped exactly.
        counts = {}
        for statement in ast.walk(init):
            if isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = (
                    statement.targets if isinstance(statement, ast.Assign) else [statement.target]
                )
                for target in targets:
                    name = attribute_path(target)
                    if name and name.startswith("self."):
                        counts[name[5:]] = counts.get(name[5:], 0) + 1
        definitions = {name: value for name, value in definitions.items() if counts.get(name) == 1}
    return definitions


def source_mapping(source, file_path, class_name, node_id, module_path="", call_line=None):
    definition = module_definitions(source, class_name).get(module_path)
    if definition:
        return SourceMapping(
            file_path=file_path,
            start_line=definition.lineno,
            end_line=definition.end_lineno,
            module_path=module_path,
            graph_node_id=node_id,
            status="exact",
        )
    if call_line:
        return SourceMapping(
            file_path=file_path,
            start_line=call_line,
            end_line=call_line,
            module_path=module_path,
            graph_node_id=node_id,
            status="inferred",
            reason="Captured call site; constructor could not be resolved exactly",
        )
    return SourceMapping(
        file_path=file_path,
        graph_node_id=node_id,
        module_path=module_path,
        reason="No unambiguous source location was captured",
    )


def offset(source: str, line: int, byte_column: int) -> int:
    lines = source.splitlines(keepends=True)
    if line == 1 and lines[0].startswith("\ufeff"):
        return 1 + len(lines[0][1:].encode("utf-8")[:byte_column].decode("utf-8"))
    return sum(map(len, lines[: line - 1])) + len(
        lines[line - 1].encode("utf-8")[:byte_column].decode("utf-8")
    )


def patch_parameter(
    source: str, class_name: str, module_path: str, key: str, value, expected_hash: str
) -> dict:
    if digest(source) != expected_hash:
        raise ValueError(
            "Source conflict: file content changed since capture. Reparse before editing the graph."
        )
    definitions = module_definitions(source, class_name)
    call = definitions.get(module_path)
    if call is None:
        raise ValueError(
            "Read-only structure: no unique direct module constructor is available for this node"
        )
    constructor_name = attribute_path(call.func)
    # Only torch.nn constructors with a statically verified import alias are patchable.
    tree = parse_ast(source)
    allowed = set()
    for statement in tree.body:
        if isinstance(statement, ast.ImportFrom) and statement.module in ("torch", "torch.nn"):
            for alias in statement.names:
                if statement.module == "torch" and alias.name == "nn":
                    allowed.add(alias.asname or alias.name)
                elif statement.module == "torch.nn":
                    allowed.add(alias.asname or alias.name)
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                if alias.name == "torch":
                    allowed.add((alias.asname or "torch") + ".nn")
                elif alias.name == "torch.nn":
                    allowed.add(alias.asname or "torch.nn")
    leaf = (constructor_name or "").split(".")[-1]
    prefix = (
        (constructor_name or "").rsplit(".", 1)[0]
        if "." in (constructor_name or "")
        else constructor_name
    )
    if prefix not in allowed:
        raise ValueError("Read-only constructor: it is not a statically resolved torch.nn import")
    # Ambiguous/reassigned imports cannot provide a safe reverse mapping.
    for n in ast.walk(tree):
        if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            if any(isinstance(t, ast.Name) and t.id in allowed for t in targets):
                raise ValueError("Read-only constructor: torch import alias is reassigned")
    argument_names = {
        "Linear": ("in_features", "out_features", "bias"),
        "Conv2d": (
            "in_channels",
            "out_channels",
            "kernel_size",
            "stride",
            "padding",
            "dilation",
            "groups",
            "bias",
            "padding_mode",
        ),
        "LayerNorm": ("normalized_shape", "eps", "elementwise_affine", "bias"),
        "BatchNorm1d": ("num_features", "eps", "momentum", "affine", "track_running_stats"),
        "BatchNorm2d": ("num_features", "eps", "momentum", "affine", "track_running_stats"),
        "BatchNorm3d": ("num_features", "eps", "momentum", "affine", "track_running_stats"),
        "Dropout": ("p",),
        "Flatten": ("start_dim", "end_dim"),
    }
    if leaf not in argument_names or key not in argument_names[leaf]:
        raise ValueError("This parameter has no safe reverse-edit adapter")
    if any(isinstance(n, ast.Starred) for n in call.args) or any(
        k.arg is None for k in call.keywords
    ):
        raise ValueError("Read-only constructor: unpacked arguments cannot be safely rewritten")
    index = argument_names[leaf].index(key)
    target = next((k.value for k in call.keywords if k.arg == key), None)
    if target is None and index < len(call.args):
        target = call.args[index]
    literal = repr(value)
    ast.literal_eval(literal)
    if target is not None:
        try:
            ast.literal_eval(target)
        except (ValueError, TypeError):
            raise ValueError(
                "Read-only parameter: value is computed from Python code, not a literal"
            ) from None
        start, end = (
            offset(source, target.lineno, target.col_offset),
            offset(source, target.end_lineno, target.end_col_offset),
        )
        updated = source[:start] + literal + source[end:]
    else:
        end = offset(source, call.end_lineno, call.end_col_offset) - 1
        before = source[:end].rstrip()
        separator = " " if before.endswith(",") else ", " if call.args or call.keywords else ""
        updated = before + separator + f"{key}={literal}" + source[end:]
    parse_ast(updated)
    return preview(source, updated)


def preview(before: str, after: str) -> dict:
    return {
        "before_hash": digest(before),
        "after_hash": digest(after),
        "text": after,
        "diff": "".join(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile="current",
                tofile="proposed",
            )
        ),
    }


def graph_hash(graph) -> str:
    return digest(json.dumps(graph.model_dump(), sort_keys=True))
