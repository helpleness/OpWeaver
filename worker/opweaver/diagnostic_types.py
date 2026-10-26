"""Versioned diagnostic contracts. Only statistics and metadata cross the RPC boundary."""

from typing import Literal

from pydantic import Field

from .lab_types import Versioned

MatchStatus = Literal["Matched", "Partially Matched", "Unmatched", "Unsupported"]
ID = r"^[0-9a-f]{32}$"


class DiagnosticConfig(Versioned):
    fp32_model_id: str = Field(pattern=ID)
    quantized_model_id: str = Field(pattern=ID)
    dataset_id: str = Field(pattern=ID)
    max_samples: int = Field(default=3, ge=1, le=16)
    max_tensors: int = Field(default=32, ge=2, le=256)
    selected_nodes: list[str] = Field(default_factory=list, max_length=64)
    operator_types: list[str] = Field(default_factory=list, max_length=64)
    memory_budget_mb: int = Field(default=128, ge=1, le=512)
    disk_budget_mb: int = Field(default=512, ge=1, le=1024)
    timeout: int = Field(default=180, ge=3, le=900)
    provider: Literal["CPUExecutionProvider"] = "CPUExecutionProvider"
    include_weights: bool = True
    temporary_policy: Literal["delete_on_terminal"] = "delete_on_terminal"


class TensorMatchResult(Versioned):
    id: str
    fp32_tensor: str
    quantized_tensor: str | None = None
    fp32_node_id: str
    quantized_node_id: str | None = None
    quantized_tensor_producer_id: str | None = None
    node_name: str = ""
    node_name_unique: bool = False
    operator_type: str
    fp32_shape: list[int | str | None] | None = None
    quantized_shape: list[int | str | None] | None = None
    match_method: str = "none"
    match_status: MatchStatus = "Unmatched"
    message: str = ""


class ErrorMetrics(Versioned):
    valid: bool = False
    reason: str | None = None
    elements: int = 0
    relative_defined_elements: int = 0
    zero_reference_elements: int = 0
    mae: float | None = None
    mse: float | None = None
    rmse: float | None = None
    max_abs_error: float | None = None
    mean_relative_error: float | None = None
    max_relative_error: float | None = None
    cosine_similarity: float | None = None


class LayerErrorRecord(TensorMatchResult):
    kind: Literal["activation", "weight"] = "activation"
    samples: list[dict] = Field(default_factory=list)
    aggregate: dict | None = None
    quantization: dict | None = None


class TensorCaptureResult(Versioned):
    sample_index: int
    feed_fingerprint: str
    example_count: int
    fp32_tensors: int
    quantized_tensors: int
    captured_bytes: int
    error: str | None = None


class DiagnosticResult(Versioned):
    diagnostic_version: str = "1.0"
    id: str = Field(pattern=ID)
    config: DiagnosticConfig
    status: Literal["Pending", "Running", "Succeeded", "Failed", "Cancelled"] = "Pending"
    success: bool = False
    created_at: float
    finished_at: float | None = None
    elapsed_seconds: float | None = None
    sample_unit: str = (
        "One sample is one saved dataset batch/inference feed; examples are counted separately"
    )
    models: dict = Field(default_factory=dict)
    dataset: dict | None = None
    environment: dict = Field(default_factory=dict)
    lineage: dict = Field(default_factory=dict)
    quantization: dict = Field(default_factory=dict)
    resources: dict = Field(default_factory=dict)
    captures: list[TensorCaptureResult] = Field(default_factory=list)
    records: list[LayerErrorRecord] = Field(default_factory=list)
    matching_summary: dict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(
        default_factory=lambda: [
            "High error may accumulate upstream; it does not establish a root cause.",
            "Top-level float32/float16 QDQ activations only; no QOperator, custom operators or control-flow subgraph capture.",
            "Tensor names plus QDQ structure and recursive producer contracts are checked; renamed/fused/replaced tensor paths may remain unmatched.",
            "Inferred shapes are bound to actual input symbols. Unknown/data-dependent shapes are not captured.",
            "Capture is flattened by ORT; inferred/concrete shape and actual capture shape are recorded separately.",
            "Tensor working budget bounds exposed outputs and statistics workspace, not total process RAM or ORT kernel allocations.",
            "Diagnostic elapsed time includes instrumentation/session creation and is not a performance benchmark.",
            "Raw activations are reduced after each batch and are never stored in SQLite or the frontend.",
        ]
    )
    logs: list[str] = Field(default_factory=list)
    diagnostics: list[dict] = Field(default_factory=list)
    error: str | None = None
