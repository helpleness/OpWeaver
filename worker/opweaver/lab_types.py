"""Versioned Optimization Lab contracts. Tensor payloads stay on disk."""

from typing import Literal

from pydantic import Field

from .ir import Spec
from .model_artifact import ModelArtifact

Level = Literal["disable", "basic", "extended", "all"]
Status = Literal["Pending", "Validating", "Running", "Succeeded", "Failed", "Cancelled"]
PHASES = ("environment", "baseline", "optimize", "validation", "optimized_benchmark")


class Versioned(Spec):
    schema_version: Literal["1.0"] = "1.0"


class BackendCapability(Versioned):
    id: str
    name: str
    available: bool
    implemented: bool = True
    version: str | None = None
    formats: list[str] = Field(default_factory=list)
    devices: list[str] = Field(default_factory=list)
    precisions: list[str] = Field(default_factory=list)
    levels: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    reason: str = ""
    note: str = ""
    environment: dict | None = None
    passes: list[dict] = Field(default_factory=list)


class OptimizationEnvironment(Versioned):
    backend_id: str = "microsoft.olive"
    python_path: str = ""


class WorkflowPass(Versioned):
    id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
    type: str
    enabled: bool = True
    parameters: dict = Field(default_factory=dict)


class WorkflowOutput(Versioned):
    model_name: str = Field(
        default="optimized.onnx", pattern=r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,80}\.onnx$"
    )
    save_external_data: bool = True
    external_data_threshold: int = Field(default=1024, ge=0, le=1024**3)


class OptimizationWorkflow(Versioned):
    id: str = Field(default="local_workflow", pattern=r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
    passes: list[WorkflowPass] = Field(default_factory=list, max_length=16)
    output: WorkflowOutput = Field(default_factory=WorkflowOutput)
    log_level: int = Field(default=1, ge=0, le=4)


class InputConfig(Versioned):
    mode: Literal["random", "npz"] = "random"
    shapes: dict[str, list[int]] = Field(default_factory=dict)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    files: list[str] = Field(default_factory=list, max_length=16)


class BenchmarkConfig(Versioned):
    warmup: int = Field(default=10, ge=0, le=10000)
    runs: int = Field(default=100, ge=1, le=100000)
    batch_size: int | None = Field(default=None, ge=1, le=1000000)
    batch_input: str | None = None
    intra_op_threads: int = Field(default=1, ge=1, le=128)
    inter_op_threads: int = Field(default=1, ge=1, le=128)
    session_optimization: Level = "disable"
    provider: Literal["CPUExecutionProvider"] = "CPUExecutionProvider"
    execution_mode: Literal["sequential"] = "sequential"
    concurrency: Literal[1] = 1


class OptimizationConfig(Versioned):
    backend_id: str = "onnxruntime.cpu"
    level: Level = "extended"
    device: Literal["cpu"] = "cpu"
    precision: Literal["preserve"] = "preserve"
    environment: OptimizationEnvironment | None = None
    workflow: OptimizationWorkflow | None = None
    calibration_dataset_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")


class DatasetConfig(Versioned):
    name: str = Field(default="Dataset", min_length=1, max_length=120)
    source_model_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    role: Literal["calibration", "evaluation"] = "calibration"
    mode: Literal["random", "npz"] = "npz"
    files: list[str] = Field(default_factory=list)
    mapping: dict[str, str] = Field(default_factory=dict)
    shapes: dict[str, list[int]] = Field(default_factory=dict)
    seed: int = Field(default=42, ge=0, le=2**32 - 513)
    count: int = Field(default=8, ge=1, le=512)
    label_key: str | None = None


class QualityGates(Versioned):
    max_mean_abs_error: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    max_abs_error: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    max_changed_fraction: float | None = Field(default=None, ge=0, le=1)
    max_accuracy_drop: float | None = Field(default=None, ge=0, le=1)


class EvaluationConfig(Versioned):
    dataset_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    allow_calibration_reuse: bool = False
    task: Literal["none", "classification_top1"] = "none"
    output_name: str | None = None
    gates: QualityGates = Field(default_factory=QualityGates)


class ValidationConfig(Versioned):
    samples: int = Field(default=3, ge=2, le=16)
    atol: float = Field(default=1e-5, ge=0, allow_inf_nan=False)
    rtol: float = Field(default=1e-4, ge=0, allow_inf_nan=False)


class ExperimentConfig(Versioned):
    mode: Literal["baseline", "optimize"] = "optimize"
    source_model_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    inputs: InputConfig = Field(default_factory=InputConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    timeout: int = Field(default=300, ge=3, le=1800)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)


class BenchmarkResult(Versioned):
    model_id: str
    model_hash: str
    model_bytes: int
    config: BenchmarkConfig
    input_config: InputConfig
    session_options: dict
    environment: dict
    inputs: list[dict]
    input_digest: str
    conditions_hash: str
    session_init_ms: float
    warmup_ms: float
    latencies_ms: list[float]
    mean_ms: float
    p50_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    requests_per_second: float | None
    samples_per_second: float | None
    timing_scope: str = "Sequential session.run wall time, including runtime output allocation; input preparation and session creation excluded"


class ValidationResult(Versioned):
    passed: bool
    samples: int
    atol: float
    rtol: float
    outputs: list[dict]
    rule: str = "abs(candidate-reference) <= atol + rtol*abs(reference); integers/bools exact; NaN fails, matching signed Inf passes"
    scope: str = "Numerical consistency on the recorded samples only; not task accuracy or proof of equivalence"
    task_quality: dict | None = None
    execution_valid: bool | None = None
    numerical_summary: dict | None = None
    quality_gate: dict | None = None


class OptimizationResult(Versioned):
    artifact: ModelArtifact
    backend_id: str
    backend_version: str
    config: OptimizationConfig
    target_environment: dict
    duration_ms: float
    serialization_options: dict = Field(default_factory=dict)
    applied_passes: list[str] | None = None
    note: str = "ORT serialized the graph at the selected level. Applied passes are not inferred from the requested level or node count. Target/runtime compatibility must be rechecked elsewhere."
    workflow: dict | None = None


class Experiment(Versioned):
    id: str
    source_model_id: str
    config: ExperimentConfig
    original: ModelArtifact | None = None
    optimization: OptimizationResult | None = None
    baseline: BenchmarkResult | None = None
    optimized: BenchmarkResult | None = None
    validation: ValidationResult | None = None
    comparison: dict | None = None
    graph_comparison: dict | None = None
    environment: dict = Field(default_factory=dict)
    input_samples: list[dict] = Field(default_factory=list)
    status: Status = "Pending"
    phases: dict[str, str] = Field(default_factory=lambda: dict.fromkeys(PHASES, "Pending"))
    logs: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    created_at: float
    finished_at: float | None = None
    error: str | None = None
    success: bool = False
    diagnostics: list[dict] = Field(default_factory=list)
    workflow_execution: dict | None = None
    datasets: dict | None = None
    quantization: dict | None = None
    data_availability: dict | None = None
    preprocessed_artifacts: list[ModelArtifact] = Field(default_factory=list)


class OptimizationJob(Versioned):
    id: str
    status: Status
    phase: str = "environment"
    progress: float | None = 0
    phases: dict[str, str] = Field(default_factory=lambda: dict.fromkeys(PHASES, "Pending"))
