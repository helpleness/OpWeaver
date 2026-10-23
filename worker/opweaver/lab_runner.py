"""Disposable experiment: baseline → adapter artifact → validation → benchmark."""

import json
import os
import sys
import time
import traceback
import uuid
from pathlib import Path

from .benchmark import benchmark, compare_benchmarks, create_session, runtime_environment
from .lab_inputs import check_feed, prepare_inputs
from .lab_reports import save_report, verify_artifact, write_json
from .lab_types import PHASES, Experiment, ExperimentConfig, InputConfig, OptimizationResult
from .lab_validation import validate_outputs
from .model_artifact import ModelArtifact
from .model_tasks import ModelTasks
from .onnx_import import import_onnx
from .optimization import backends
from .storage import RunStore


def register_stage(project, output, parent, experiment_id, environment):
    model_id = uuid.uuid4().hex
    folder = project / ".opweaver" / "imports" / model_id
    model = import_onnx(output, project, folder)
    artifact = ModelArtifact.model_validate(
        {
            **model["artifact"],
            "parent_model_id": parent.id,
            "experiment_id": experiment_id,
            "target_environment": {**environment, "stage": "preprocessed"},
        }
    )
    model["artifact"] = artifact.model_dump()
    write_json(folder / "result.json", model)
    RunStore(project).set_metadata(
        f"model-import:{model_id}",
        {
            "id": model_id,
            "state": "COMPLETED",
            "project": str(project),
            "file_path": str(output),
            "source_hash": artifact.sha256,
            "artifact_dir": str(folder),
            "created_at": time.time(),
            "artifact": artifact.model_dump(),
            "error": None,
        },
    )
    return artifact


def execute(request, folder):
    config = ExperimentConfig.model_validate(request["config"])
    project = Path(request["project"]).resolve(strict=True)
    report = Experiment(
        id=folder.name,
        source_model_id=config.source_model_id,
        config=config,
        created_at=request.get("created_at", time.time()),
        phases={phase: "Pending" for phase in PHASES},
        warnings=[
            "Random inputs are synthetic test data, not a task dataset. Numerical validation does not measure task accuracy.",
            "Benchmark sides run sequentially; system load, thermal state and timer noise can affect small models.",
        ],
    )
    phase = "environment"

    def progress(value, message):
        status = "Validating" if phase == "environment" else "Running"
        report.status = status
        report.phases[phase] = "Running"
        line = f"{time.strftime('%H:%M:%S')} [{phase}] {message}"
        report.logs.append(line)
        report.logs = report.logs[-300:]
        write_json(
            folder / "progress.json",
            {
                "phase": phase,
                "progress": value,
                "status": status,
                "phases": report.phases,
                "logs": "\n".join(report.logs),
                "runner_pid": os.getpid(),
            },
        )
        # Store completed phase results as well as live progress for interruption recovery.
        save_report(folder, report.model_dump())

    def begin(name, message):
        nonlocal phase
        phase = name
        progress(0, message)

    def finish():
        report.phases[phase] = "Succeeded"

    try:
        begin("environment", "Checking artifact hashes, runtime compatibility and explicit inputs")
        original_report = ModelTasks().load(project, config.source_model_id)
        artifact = ModelArtifact.model_validate(original_report["artifact"])
        report.original = artifact
        adapter = backends(
            config.optimization.environment.model_dump()
            if config.optimization.environment
            else None,
            original_report,
        ).get(config.optimization.backend_id)
        if adapter is None:
            raise ValueError("Optimization backend is not implemented")
        adapter.validate(artifact, config.optimization)
        path = verify_artifact(project, artifact)
        report.environment = runtime_environment()
        reference, init_ms = create_session(path, config.benchmark)
        from .datasets import DatasetFeeds, dataset_pair

        report.datasets = dataset_pair(project, config)
        if hasattr(adapter, "calibration"):
            adapter.calibration = report.datasets["calibration"]
        evaluation = report.datasets["evaluation"]
        benchmark_inputs = config.inputs
        if evaluation:
            samples = DatasetFeeds(evaluation)
            for feed in samples:
                check_feed(reference.get_inputs(), feed)
            # Reuse existing benchmark input snapshots with the actual evaluation feed.
            import numpy as np

            from .benchmark import digest_file

            (folder / "inputs").mkdir(exist_ok=True)
            target = folder / "inputs" / "evaluation-benchmark.npz"
            np.savez(target, **samples[0])
            benchmark_inputs = InputConfig(mode="npz", files=[str(target)])
            report.input_samples = [
                {
                    "index": 0,
                    "path": target.name,
                    "sha256": digest_file(target),
                    "dataset_id": evaluation["id"],
                }
            ]
            if config.benchmark.batch_size is not None:
                value = samples[0].get(config.benchmark.batch_input)
                if value is None or not value.ndim or value.shape[0] != config.benchmark.batch_size:
                    raise ValueError("Declared benchmark batch does not match evaluation data")
        else:
            count = config.validation.samples if config.mode == "optimize" else 1
            samples, report.input_samples = prepare_inputs(
                reference.get_inputs(), config.inputs, count, folder / "inputs", config.benchmark
            )
        for role in ("calibration", "evaluation", "final_evaluation"):
            if report.datasets[role]:
                report.warnings.extend(report.datasets[role]["warnings"])
        if report.datasets["warning"]:
            report.warnings.append(report.datasets["warning"])
        lossy = bool(
            config.optimization.workflow
            and config.optimization.backend_id == "microsoft.olive"
            and any(
                p.enabled and p.type == "OnnxStaticQuantization"
                for p in config.optimization.workflow.passes
            )
        )
        if lossy and (not evaluation or not report.datasets["calibration"]):
            raise ValueError("Static quantization requires calibration and evaluation datasets")
        if not any(
            report.datasets[key] for key in ("evaluation", "calibration", "final_evaluation")
        ):
            report.datasets = None
        finish()
        begin("baseline", "CPU baseline; session creation is excluded from timed runs")
        report.baseline = benchmark(
            reference,
            init_ms,
            artifact,
            samples[0],
            config.benchmark,
            report.environment,
            progress,
            benchmark_inputs,
        )
        finish()
        if config.mode == "baseline":
            from .task_evaluation import evaluate_single

            report.baseline_evaluation = evaluate_single(
                reference, evaluation, config.evaluation, progress
            )
            if report.datasets and report.datasets.get("final_evaluation"):
                report.final_evaluation = evaluate_single(
                    reference, report.datasets["final_evaluation"], config.evaluation, progress
                )
            finish()
            for name in PHASES[2:]:
                report.phases[name] = "Skipped"
        else:
            begin("optimize", f"Creating a separate model using {adapter.id}")
            output = folder / "optimized.onnx"
            info = adapter.optimize(path, output, config.optimization, config.benchmark, progress)
            output = Path(info.get("model_path", output))
            report.workflow_execution = info.get("workflow")
            parent_artifact = artifact
            for item in (report.workflow_execution or {}).get("executed_passes", []):
                if item["pass_type"].lower() == "onnxquantizationpreprocess" and item.get(
                    "artifact_path"
                ):
                    stage_path = Path(item["artifact_path"]).resolve(strict=True)
                    if not stage_path.is_relative_to(folder.resolve()):
                        raise ValueError("Preprocessed artifact escapes the experiment directory")
                    parent_artifact = register_stage(
                        project, stage_path, parent_artifact, report.id, info["environment"]
                    )
                    report.preprocessed_artifacts.append(parent_artifact)
            if lossy:
                from .quantization_analysis import analyze_quantization

                report.quantization = analyze_quantization(output)
                if not report.quantization["present"]:
                    raise ValueError("Olive output has no observed quantization representation")
            candidate, candidate_init = create_session(output, config.benchmark)
            for feed in samples:
                check_feed(candidate.get_inputs(), feed)
            model_id = uuid.uuid4().hex
            model_folder = project / ".opweaver" / "imports" / model_id
            optimized_report = import_onnx(output, project, model_folder)
            if report.quantization:
                optimized_report["quantization"] = report.quantization
            optimized_artifact = ModelArtifact.model_validate(
                {
                    **optimized_report["artifact"],
                    "parent_model_id": parent_artifact.id,
                    "experiment_id": report.id,
                    "target_environment": info["environment"],
                }
            )
            optimized_report["artifact"] = optimized_artifact.model_dump()
            encoded = json.dumps(optimized_report, ensure_ascii=True)
            if len(encoded) > 24 * 1024 * 1024:
                raise ValueError("Optimized graph report exceeds the 24 MiB viewer limit")
            write_json(model_folder / "result.json", optimized_report)
            RunStore(project).set_metadata(
                f"model-import:{model_id}",
                {
                    "id": model_id,
                    "state": "COMPLETED",
                    "project": str(project),
                    "file_path": str(output),
                    "source_hash": optimized_artifact.sha256,
                    "artifact_dir": str(model_folder),
                    "created_at": time.time(),
                    "artifact": optimized_artifact.model_dump(),
                    "error": None,
                },
            )
            report.optimization = OptimizationResult(
                artifact=optimized_artifact,
                backend_id=adapter.id,
                backend_version=info.get("backend_version", report.environment["onnxruntime"]),
                config=config.optimization,
                target_environment=info["environment"],
                duration_ms=info["duration_ms"],
                serialization_options=info.get(
                    "serialization_options",
                    {
                        "external_initializers_file_name": output.name + ".data",
                        "external_initializers_min_size_in_bytes": 0,
                    },
                ),
                **(
                    {"note": info["note"], "workflow": info["workflow"]}
                    if info.get("workflow")
                    else {}
                ),
            )
            report.graph_comparison = {
                "original_nodes": original_report["statistics"]["nodes"],
                "optimized_nodes": optimized_report["statistics"]["nodes"],
                "original_operators": original_report["statistics"]["operators"],
                "optimized_operators": optimized_report["statistics"]["operators"],
                "original_bytes": artifact.total_file_bytes,
                "optimized_bytes": optimized_artifact.total_file_bytes,
                "scope": "Top-level ONNX operators; no fusion or speedup is inferred from node count",
            }
            finish()
            begin("validation", "Comparing all outputs on the same recorded input samples")
            report.validation = validate_outputs(
                reference, candidate, samples, config.validation, progress
            )
            from .task_evaluation import evaluate_task, quality_gate

            report.validation.task_quality = evaluate_task(
                reference, candidate, evaluation, config.evaluation, progress
            )
            report.validation.quality_gate = quality_gate(
                report.validation.numerical_summary,
                report.validation.task_quality,
                config.evaluation.gates,
            )
            if lossy and (
                not report.validation.execution_valid
                or any(r.get("nonfinite_count") for r in report.validation.outputs)
            ):
                raise ValueError("Quantized model output schema/non-finite validation failed")
            if not lossy and not report.validation.passed:
                raise ValueError(
                    "Numerical consistency validation failed; inspect per-output errors"
                )
            if report.validation.quality_gate["status"] == "failed":
                report.warnings.append(
                    "Quality gate failed; execution and performance results are retained"
                )
            finish()
            begin(
                "optimized_benchmark",
                "Benchmarking the reloaded artifact with identical session settings and inputs",
            )
            # Reload the registered artifact, not only the transient ORT output.
            candidate, candidate_init = create_session(
                verify_artifact(project, optimized_artifact), config.benchmark
            )
            report.optimized = benchmark(
                candidate,
                candidate_init,
                optimized_artifact,
                samples[0],
                config.benchmark,
                report.environment,
                progress,
                benchmark_inputs,
            )
            report.comparison = compare_benchmarks(report.baseline, report.optimized)
            if (report.datasets or {}).get("final_evaluation"):
                report.final_evaluation = evaluate_task(
                    reference,
                    candidate,
                    report.datasets["final_evaluation"],
                    config.evaluation,
                    progress,
                )
            finish()
        verify_artifact(project, artifact)
        report.status, report.success = "Succeeded", True
        report.logs.append(f"{time.strftime('%H:%M:%S')} [complete] Experiment saved")
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        report.status, report.error = "Failed", f"{type(exc).__name__}: {exc}"
        report.phases[phase] = "Failed"
        report.logs.append(f"[failed] {report.error}")
        report.diagnostics = [{"severity": "error", "message": report.error}]
    report.finished_at = time.time()
    olive_state = folder / "olive" / "execution.json"
    if olive_state.is_file():
        report.workflow_execution = json.loads(olive_state.read_text(encoding="utf-8"))
    save_report(folder, report.model_dump())
    return report.model_dump()


def main():
    request_path, result_path = map(Path, sys.argv[1:3])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request.get("owner_pid"):
        from .process_owner import watch_owner

        watch_owner(request["owner_pid"], result_path.parent / "cancel.request")
    write_json(result_path, execute(request, result_path.parent))


if __name__ == "__main__":
    main()
