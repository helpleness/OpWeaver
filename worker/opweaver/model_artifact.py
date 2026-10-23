"""Versioned, format-specific ONNX import report. Weights remain separate files."""

from typing import Literal

from pydantic import Field

from .ir import Spec
from .source import AnalysisGraph


class ModelDiagnostic(Spec):
    severity: Literal["error", "warning", "info"]
    code: str = "MODEL_ANALYSIS"
    message: str
    node: str | None = None
    source: str = "model.import"
    suggested_action: str = ""


class ModelMetadata(Spec):
    ir_version: int | str
    opsets: dict[str, int] = Field(default_factory=dict)
    producer: str = ""
    producer_version: str = ""
    model_version: int = 0
    domain: str = ""
    doc_string: str = ""
    metadata: list[dict] = Field(default_factory=list)
    parser_version: str
    local_functions: list[str] = Field(default_factory=list)
    overridable_initializers: list[str] = Field(default_factory=list)
    scope: str
    entry: str = ""


class ModelArtifact(Spec):
    schema_version: Literal["1.0"] = "1.0"
    id: str
    format: Literal["onnx", "pytorch"] = "onnx"
    name: str
    file_path: str
    sha256: str
    file_bytes: int
    total_file_bytes: int
    external_files: list[dict] = Field(default_factory=list)
    imported_at: float
    source_name: str
    source_path: str
    importer_id: str
    importer_version: str = "1"
    parent_model_id: str | None = None
    experiment_id: str | None = None
    target_environment: dict | None = None


class GraphImportResult(Spec):
    schema_version: Literal["1.0"] = "1.0"
    success: Literal[True] = True
    artifact: ModelArtifact
    analysis: AnalysisGraph
    model: ModelMetadata
    inputs: list[dict]
    outputs: list[dict]
    initializers: list[dict]
    statistics: dict
    validation: dict
    diagnostics: list[ModelDiagnostic]
    subgraphs: dict[str, AnalysisGraph] = Field(default_factory=dict)
    pytorch: dict | None = None
    quantization: dict | None = None


OnnxReport = GraphImportResult
