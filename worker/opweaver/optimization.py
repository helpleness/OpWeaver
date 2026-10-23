"""Optimization adapter registry. Sampling and validation are independent services."""

import time
from abc import ABC, abstractmethod

from .benchmark import LEVELS, create_session, runtime_environment
from .lab_types import BackendCapability


class OptimizationBackend(ABC):
    id: str

    @abstractmethod
    def probe(self) -> BackendCapability: ...

    @abstractmethod
    def validate(self, artifact, config): ...

    @abstractmethod
    def optimize(self, path, output, config, benchmark_config, progress): ...


class OrtCPUBackend(OptimizationBackend):
    id = "onnxruntime.cpu"

    def probe(self):
        try:
            import onnxruntime as ort

            available = "CPUExecutionProvider" in ort.get_available_providers()
            return BackendCapability(
                id=self.id,
                name="ONNX Runtime · CPU",
                available=available,
                version=ort.__version__,
                formats=["onnx"],
                devices=["cpu"],
                precisions=["preserve"],
                levels=[
                    name
                    for name, enum in LEVELS.items()
                    if hasattr(ort.GraphOptimizationLevel, enum)
                ],
                dependencies=["onnxruntime>=1.24,<2"],
                reason="" if available else "CPUExecutionProvider is unavailable",
                note="Official ORT graph serialization. A level is a capability, not a record of applied passes. All may include platform-specific layout transformations.",
            )
        except Exception as exc:
            return BackendCapability(
                id=self.id,
                name="ONNX Runtime · CPU",
                available=False,
                dependencies=["onnxruntime>=1.24,<2"],
                reason=f"ONNX Runtime dependency is unavailable: {type(exc).__name__}: {exc}",
            )

    def validate(self, artifact, config):
        capability = self.probe()
        if not capability.available:
            raise ValueError(capability.reason)
        if artifact.format not in capability.formats or config.level not in capability.levels:
            raise ValueError("Model format or optimization level is unsupported by this backend")

    def optimize(self, path, output, config, benchmark_config, progress):
        progress(0, f"ORT {config.level}: serializing a separate model")
        start = time.perf_counter_ns()
        session, _ = create_session(path, benchmark_config, level=config.level, output=output)
        del session
        if not output.is_file() or output.stat().st_size == 0:
            raise ValueError("ORT did not produce a readable optimized model")
        progress(1, "ORT serialization completed; output will be reloaded and validated")
        return {
            "duration_ms": (time.perf_counter_ns() - start) / 1e6,
            "environment": runtime_environment(),
        }


def backends(environment=None, model=None):
    from .olive_adapter import OliveBackend

    return {OrtCPUBackend.id: OrtCPUBackend(), OliveBackend.id: OliveBackend(environment, model)}


def capabilities(environment=None, model=None):
    return [b.probe().model_dump() for b in backends(environment, model).values()] + [
        BackendCapability(
            id="nvidia.modelopt",
            name="NVIDIA Model Optimizer",
            available=False,
            implemented=False,
            reason="Adapter is not implemented; model format, feature, CUDA and hardware support must be probed per feature",
        ).model_dump(),
    ]
