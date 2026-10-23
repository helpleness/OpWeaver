"""Real ORT CPU sessions and a backend-independent sampling/statistics engine."""

import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from .lab_types import BenchmarkConfig, BenchmarkResult

LEVELS = {
    "disable": "ORT_DISABLE_ALL",
    "basic": "ORT_ENABLE_BASIC",
    "extended": "ORT_ENABLE_EXTENDED",
    "all": "ORT_ENABLE_ALL",
}


def digest_file(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as reader:
        while block := reader.read(1024 * 1024):
            hasher.update(block)
    return hasher.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def runtime_environment():
    import onnxruntime as ort

    return {
        "onnxruntime": ort.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "provider": "CPUExecutionProvider",
        "available_providers": ort.get_available_providers(),
        "cpu": platform.processor() or platform.machine(),
        "logical_cpus": os.cpu_count(),
        "architecture": platform.machine(),
        "os": platform.platform(),
        "device": "cpu",
    }


def create_session(path, config: BenchmarkConfig, level=None, output=None):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = getattr(
        ort.GraphOptimizationLevel, LEVELS[level or config.session_optimization]
    )
    options.intra_op_num_threads = config.intra_op_threads
    options.inter_op_num_threads = config.inter_op_threads
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.enable_mem_pattern = True
    options.enable_mem_reuse = True
    options.enable_cpu_mem_arena = True
    options.log_severity_level = 2
    if output is not None:
        options.optimized_model_filepath = str(output)
        # Explicitly reserialize external initializers next to the new graph.
        # Otherwise ORT can retain relative references to the original weights.
        options.add_session_config_entry(
            "session.optimized_model_external_initializers_file_name", Path(output).name + ".data"
        )
        options.add_session_config_entry(
            "session.optimized_model_external_initializers_min_size_in_bytes", "0"
        )
    start = time.perf_counter_ns()
    session = ort.InferenceSession(str(path), sess_options=options, providers=[config.provider])
    elapsed = (time.perf_counter_ns() - start) / 1e6
    if session.get_providers() != [config.provider]:
        raise ValueError(f"Unexpected execution providers: {session.get_providers()}")
    return session, elapsed


class BenchmarkEngine:
    """Measures a supplied runner; optimizer implementations never own the timer."""

    def run(self, run, config, progress=lambda value, message: None):
        last_progress = time.monotonic()
        warmup_ms = 0.0
        for i in range(config.warmup):
            start = time.perf_counter_ns()
            run()
            warmup_ms += (time.perf_counter_ns() - start) / 1e6
            if i % max(1, config.warmup // 20) == 0 and time.monotonic() - last_progress >= 0.2:
                progress(0.2 * (i + 1) / max(1, config.warmup), f"Warmup {i + 1}/{config.warmup}")
                last_progress = time.monotonic()
        samples = []
        for i in range(config.runs):
            start = time.perf_counter_ns()
            run()
            samples.append((time.perf_counter_ns() - start) / 1e6)
            if (
                i % max(1, config.runs // 30) == 0 and time.monotonic() - last_progress >= 0.2
            ) or i == config.runs - 1:
                progress(
                    0.2 + 0.8 * (i + 1) / config.runs, f"Timed inference {i + 1}/{config.runs}"
                )
                last_progress = time.monotonic()
        return warmup_ms, samples


def benchmark(session, init_ms, artifact, feed, config, environment, progress, input_config):
    warmup_ms, samples = BenchmarkEngine().run(lambda: session.run(None, feed), config, progress)
    mean = float(np.mean(samples))
    requests = 1000 / mean if mean > 0 else None
    input_info = [
        {"name": name, "shape": list(array.shape), "dtype": str(array.dtype)}
        for name, array in feed.items()
    ]
    input_digest = digest_json(
        [
            {**info, "sha256": hashlib.sha256(feed[info["name"]].tobytes()).hexdigest()}
            for info in input_info
        ]
    )
    return BenchmarkResult(
        model_id=artifact.id,
        model_hash=artifact.sha256,
        model_bytes=artifact.total_file_bytes,
        config=config,
        input_config=input_config,
        session_options={
            "graph_optimization_level": config.session_optimization,
            "intra_op_num_threads": config.intra_op_threads,
            "inter_op_num_threads": config.inter_op_threads,
            "execution_mode": "ORT_SEQUENTIAL",
            "enable_mem_pattern": True,
            "enable_mem_reuse": True,
            "enable_cpu_mem_arena": True,
            "providers": [config.provider],
        },
        environment=environment,
        inputs=input_info,
        input_digest=input_digest,
        conditions_hash=digest_json(
            {"config": config.model_dump(), "environment": environment, "inputs": input_digest}
        ),
        session_init_ms=init_ms,
        warmup_ms=warmup_ms,
        latencies_ms=samples,
        mean_ms=mean,
        p50_ms=float(np.percentile(samples, 50)),
        p95_ms=float(np.percentile(samples, 95)),
        min_ms=min(samples),
        max_ms=max(samples),
        requests_per_second=requests,
        samples_per_second=requests * config.batch_size if requests and config.batch_size else None,
    )


def compare_benchmarks(baseline, optimized):
    """Only same-condition, finite positive measurements can claim a speedup."""
    if not baseline or not optimized:
        return {
            "comparable": False,
            "outcome": "Unavailable",
            "reason": "Both benchmark results are required",
            "speedup": None,
        }
    a = baseline.model_dump() if isinstance(baseline, BenchmarkResult) else baseline
    b = optimized.model_dump() if isinstance(optimized, BenchmarkResult) else optimized

    def ratio(x, y):
        return (
            x / y
            if isinstance(x, (int, float))
            and isinstance(y, (int, float))
            and np.isfinite(x)
            and np.isfinite(y)
            and x > 0
            and y > 0
            else None
        )

    same = bool(a.get("conditions_hash")) and a.get("conditions_hash") == b.get("conditions_hash")
    speedup = ratio(a.get("mean_ms"), b.get("mean_ms")) if same else None
    latency = ratio(b.get("mean_ms"), a.get("mean_ms")) if same else None
    throughput = ratio(b.get("requests_per_second"), a.get("requests_per_second")) if same else None
    size = ratio(b.get("model_bytes"), a.get("model_bytes"))
    return {
        "comparable": same and speedup is not None,
        "outcome": "Unavailable"
        if speedup is None
        else "Regression"
        if speedup < 1
        else "Speedup"
        if speedup > 1
        else "Unchanged",
        "reason": "Identical recorded environment, inputs and session settings; sequential runs are subject to system noise"
        if same
        else "Environment, inputs or session settings differ; no same-condition speedup",
        "speedup": speedup,
        "latency_change_percent": (latency - 1) * 100 if latency is not None else None,
        "throughput_change_percent": (throughput - 1) * 100 if throughput is not None else None,
        "size_change_percent": (size - 1) * 100 if size is not None else None,
    }
