"""Registered model importers. PyTorch delegates to the existing P0-A capture."""

import importlib.util
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .model_artifact import GraphImportResult, ModelArtifact
from .source import inspect_source, read_source


class ModelImporter(Protocol):
    id: str
    format: str
    extensions: tuple[str, ...]
    requires_execution: bool

    def import_model(self, request: dict, folder: Path) -> dict: ...


@dataclass
class OnnxImporter:
    id = "onnx"
    format = "onnx"
    extensions = (".onnx",)
    requires_execution = False

    def import_model(self, request, folder):
        from .onnx_import import import_onnx

        return import_onnx(request["file_path"], request["project"], folder)


@dataclass
class PyTorchImporter:
    id = "pytorch-source"
    format = "pytorch"
    extensions = (".py",)
    requires_execution = True

    def import_model(self, request, folder):
        from .capture import capture
        from .model_files import copy_file

        if request.get("execute_confirmed") is not True:
            raise ValueError("PyTorch import executes local Python. Trigger it explicitly.")
        project = Path(request["project"]).resolve()
        source = Path(request["file_path"])
        if source.stat().st_size > 2 * 1024 * 1024:
            raise ValueError("Python source exceeds the 2 MiB editor limit")
        target = project / "src" / f"imported_{folder.name}.py"
        if not target.resolve().is_relative_to(project):
            raise ValueError("Python source path escapes the project")
        identity = copy_file(source, target)
        result = capture(
            str(target), request["entry"], request.get("constructor", {}), request["inputs"]
        )
        result["capture"]["constructor"] = request.get("constructor", {})
        result["analysis"]["source_model_id"] = folder.name
        operator_nodes = [
            n for n in result["analysis"]["nodes"] if n["kind"] not in ("placeholder", "output")
        ]
        return GraphImportResult(
            artifact=ModelArtifact(
                id=folder.name,
                format="pytorch",
                name=request["entry"],
                file_path=target.relative_to(project).as_posix(),
                sha256=identity["sha256"],
                file_bytes=identity["file_bytes"],
                total_file_bytes=identity["file_bytes"],
                imported_at=time.time(),
                source_name=source.name,
                source_path=str(source),
                importer_id=self.id,
            ),
            analysis=result["analysis"],
            model={
                "ir_version": "FX",
                "producer": "PyTorch",
                "producer_version": result["capture"]["pytorch"],
                "parser_version": result["capture"]["pytorch"],
                "scope": result["capture"]["specialized"]
                + "; source artifact size excludes model weights",
                "entry": request["entry"],
            },
            inputs=[
                {"name": n["label"], **n["tensors"]}
                for n in result["analysis"]["nodes"]
                if n["kind"] == "placeholder"
            ],
            outputs=[
                {"name": f"output_{i}", **v} for i, v in enumerate(result["forward"]["outputs"])
            ],
            initializers=[],
            statistics={
                "nodes": len(operator_nodes),
                "graph_nodes": len(result["analysis"]["nodes"]),
                "edges": len(result["analysis"]["edges"]),
                "operators": dict(Counter(n["operator"] for n in operator_nodes)),
                "initializer_count": 0,
                "initializer_elements": None,
                "learnable_parameters": result["forward"]["parameters"],
                "parameters_note": "PyTorch parameter elements from the constructed module; weights are not serialized by this source import",
                "logical_tensor_bytes": None,
                "unknown_operators": {},
                "flops": None,
                "weight_dtypes": {},
            },
            validation={
                "checker": "not_applicable",
                "shape_inference": "FX ShapeProp",
                "runtime": "CPU eval forward",
                "reference_passed": result["validation"]["passed"],
            },
            diagnostics=[
                {
                    "severity": d["severity"],
                    "message": d["message"],
                    "code": "PYTORCH_ANALYSIS",
                    "node": d.get("node"),
                    "source": "pytorch.fx",
                    "suggested_action": d.get("hint", ""),
                }
                for d in result["diagnostics"]
            ],
            pytorch=result,
        ).model_dump()


class ImporterRegistry:
    def __init__(self):
        self.entries: dict[str, ModelImporter] = {}

    def register(self, importer: ModelImporter):
        if importer.id in self.entries:
            raise ValueError(f"Importer already registered: {importer.id}")
        self.entries[importer.id] = importer

    def manifests(self):
        return [
            {
                "id": v.id,
                "format": v.format,
                "version": "1",
                "extensions": v.extensions,
                "requires_execution": v.requires_execution,
                "available": importlib.util.find_spec("onnx" if v.format == "onnx" else "torch")
                is not None,
            }
            for v in self.entries.values()
        ]

    def resolve(self, path):
        suffix = Path(path).suffix.lower()
        for importer in self.entries.values():
            if suffix in importer.extensions:
                return importer
        if suffix in (".pt", ".pth"):
            raise ValueError(
                ".pt/.pth may be weights without architecture or pickled model objects. Automatic deserialization is disabled; import the matching Python model definition instead."
            )
        raise ValueError("No registered importer recognizes this model format")

    def probe(self, path):
        importer = self.resolve(path)
        result = next(v for v in self.manifests() if v["id"] == importer.id)
        if importer.requires_execution:
            source = Path(path)
            if source.stat().st_size > 2 * 1024 * 1024:
                raise ValueError("Python source exceeds 2 MiB")
            result = {**result, **inspect_source(read_source(source), str(source))}
        return result


def builtin_importers():
    registry = ImporterRegistry()
    registry.register(OnnxImporter())
    registry.register(PyTorchImporter())
    return registry
