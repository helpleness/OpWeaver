"""Independent, bounded FP32/QDQ diagnostics over existing immutable artifacts/datasets."""

import inspect
import json
import math
import time
from collections import Counter
from pathlib import Path

import onnx
import onnxruntime as ort
from onnxruntime.quantization import qdq_loss_debug as debug

from .benchmark import runtime_environment
from .datasets import DatasetFeeds, content_digest, dataset_overlap, load_dataset
from .diagnostic_capture import (
    POSTFIX,
    augment_model,
    collect,
    concrete_shape,
    estimated_bytes,
    session_options,
)
from .diagnostic_matching import GraphIndex, TensorMatcher
from .diagnostic_metrics import aggregate_metrics, tensor_metrics
from .diagnostic_reports import cleanup_temporary, save_diagnostic
from .diagnostic_types import DiagnosticConfig, DiagnosticResult
from .diagnostic_weights import weight_records
from .lab_inputs import check_feed
from .lab_reports import experiment_folder, verify_artifact, write_json
from .model_artifact import ModelArtifact
from .model_files import copy_file
from .model_tasks import ModelTasks
from .quantization_analysis import analyze_quantization
from .storage import RunStore

MIB = 1024**2
MODEL_LIMIT = 128 * MIB


def model_copy(project, artifact, directory):
    source = verify_artifact(project, artifact)
    target = directory / source.name
    copy_file(source, target)
    for value in artifact.external_files:
        copy_file(source.parent / value["path"], directory / value["path"])
    return target


def matching_summary(rows, scope_count):
    activations = [r for r in rows if r["kind"] == "activation"]
    states = Counter(r["match_status"] for r in activations)
    comparable = sum(bool((r.get("aggregate") or {}).get("valid_samples")) for r in activations)
    return {
        "scope_tensors": scope_count,
        "recorded_tensors": len(activations),
        "omitted_scope_tensors": max(0, scope_count - len(activations)),
        "statuses": dict(states),
        "comparable_tensors": comparable,
        "coverage": comparable / scope_count if scope_count else None,
        "coverage_definition": "Activation tensors with at least one valid captured feed / all output tensors in the requested original-node scope; weights are separate",
        "weights": dict(Counter(r["match_status"] for r in rows if r["kind"] == "weight")),
    }


def compare_capture(plan, reference, candidate, feed, left, right, actual_matches, index, examples):
    """Reduce one tensor inside a short-lived scope; never retain raw capture arrays."""
    sample = {"sample_index": index, "example_count": int(examples)}
    try:
        shape_a = concrete_shape(reference, plan["reference"], feed)
        shape_b = concrete_shape(candidate, plan["post"], feed)
        if plan["qdq"]:
            pair = actual_matches.get(plan["reference"], {})
            a_values, b_values = pair.get("float"), pair.get("post_qdq")
        else:
            a_values, b_values = left.get(plan["reference"]), right.get(plan["post"])
        sample.update(
            fp32_concrete_inferred_shape=shape_a,
            quantized_concrete_inferred_shape=shape_b,
            shape_source="ONNX inference bound to actual input symbols; capture is flattened by ORT",
        )
        if not a_values or not b_values:
            sample["metrics"] = {
                "valid": False,
                "reason": "Actual ORT matching/collection returned no pair",
            }
            return sample
        a, b = a_values[0], b_values[0]
        sample.update(
            fp32_capture_shape=list(a.shape),
            quantized_capture_shape=list(b.shape),
            fp32_dtype=str(a.dtype),
            quantized_dtype=str(b.dtype),
        )
        if a.size != math.prod(shape_a) or b.size != math.prod(shape_b):
            sample["metrics"] = {
                "valid": False,
                "reason": "Actual capture length differs from inferred concrete shape",
            }
        else:
            sample["metrics"] = tensor_metrics(a.reshape(shape_a), b.reshape(shape_b))
            if plan["qdq"]:
                pre = right[plan["pre"]][0]
                sample["local_qdq_metrics"] = (
                    tensor_metrics(pre.reshape(shape_b), b.reshape(shape_b))
                    if pre.size == b.size
                    else {"valid": False, "reason": "Local QDQ shape differs"}
                )
    except (MemoryError, InterruptedError, TimeoutError):
        raise
    except Exception as exc:
        sample["metrics"] = {"valid": False, "reason": f"{type(exc).__name__}: {exc}"}
    return sample


class QuantizationDiagnosticEngine:
    def execute(self, request, folder):
        config = DiagnosticConfig.model_validate(request["config"])
        folder, project = (
            Path(folder).resolve(strict=True),
            Path(request["project"]).resolve(strict=True),
        )
        start, scope_count = time.monotonic(), 0
        report = DiagnosticResult(
            id=folder.name, config=config, created_at=request.get("created_at", time.time())
        )
        value = report.model_dump()
        temporary = folder / "temporary"
        verified = []

        def progress(phase, fraction, message):
            if (folder / "cancel.request").exists():
                raise InterruptedError("Diagnostic cancelled")
            if time.monotonic() - start > config.timeout:
                raise TimeoutError("Diagnostic timeout exceeded")
            value["status"] = "Running"
            value["logs"] = [
                *value["logs"][-149:],
                f"{time.strftime('%H:%M:%S')} [{phase}] {message}",
            ]
            if temporary.exists():
                used = sum(p.stat().st_size for p in temporary.rglob("*") if p.is_file())
                value["resources"]["temporary_disk_peak_observed_bytes"] = max(
                    used, value["resources"].get("temporary_disk_peak_observed_bytes", 0)
                )
                if used > config.disk_budget_mb * MIB:
                    raise ValueError("Diagnostic temporary disk budget exceeded")
            value["matching_summary"] = matching_summary(value["records"], scope_count)
            save_diagnostic(folder, value)
            write_json(
                folder / "progress.json",
                {
                    "phase": phase,
                    "progress": fraction,
                    "status": "Running",
                    "logs": "\n".join(value["logs"]),
                    "matching_summary": value["matching_summary"],
                },
            )

        try:
            value["resources"] = {
                "memory_budget_bytes": config.memory_budget_mb * MIB,
                "disk_budget_bytes": config.disk_budget_mb * MIB,
                "model_file_limit_bytes": MODEL_LIMIT,
                "max_samples": config.max_samples,
                "max_tensors": config.max_tensors,
                "default_node_limit": 64,
                "weight_record_limit": 64,
                "raw_tensor_retention": "none",
                "temporary_policy": config.temporary_policy,
                "budget_scope": "All exposed outputs from both models for one feed plus a conservative 96 bytes/element statistics workspace for the largest tensor; model/session/kernel RAM is separate",
            }
            progress(
                "validate",
                0,
                "Checking model artifacts, diagnostic role, hashes and installed ORT APIs",
            )
            if config.fp32_model_id == config.quantized_model_id:
                raise ValueError("Select distinct FP32 and quantized model artifacts")
            api_names = (
                "modify_model_output_intermediate_tensors",
                "collect_activations",
                "create_activation_matching",
                "create_weight_matching",
            )
            value["environment"] = {
                **runtime_environment(),
                "onnx": onnx.__version__,
                "diagnostic_api": {n: str(inspect.signature(getattr(debug, n))) for n in api_names},
                "session_options": {
                    "graph_optimization": "disable",
                    "threads": [1, 1],
                    "execution": "sequential",
                    "memory_pattern": False,
                    "cpu_arena": False,
                },
            }
            if getattr(debug, "_TENSOR_SAVE_POSTFIX", None) != POSTFIX:
                raise ValueError("Installed ORT instrumentation output naming is not supported")
            model_reports = [
                ModelTasks().load(project, id)
                for id in (config.fp32_model_id, config.quantized_model_id)
            ]
            artifacts = [ModelArtifact.model_validate(r["artifact"]) for r in model_reports]
            value["models"] = {
                "fp32": artifacts[0].model_dump(),
                "quantized": artifacts[1].model_dump(),
                "source_experiment_id": artifacts[1].experiment_id,
            }
            for artifact in artifacts:
                if artifact.format != "onnx" or artifact.total_file_bytes > MODEL_LIMIT:
                    raise ValueError(
                        "Diagnostics requires ONNX artifacts of at most 128 MiB each including external weights"
                    )
                verified.append((artifact, verify_artifact(project, artifact)))
            if sum(a.total_file_bytes for a in artifacts) * 6 + MIB > config.disk_budget_mb * MIB:
                raise ValueError(
                    "Model copies/instrumentation may exceed the disk budget; raise the budget or use a smaller model"
                )
            lineage, current, seen = [], artifacts[1], set()
            while current.parent_model_id and current.id not in seen and len(seen) < 64:
                seen.add(current.id)
                lineage.append(current.parent_model_id)
                if current.parent_model_id == artifacts[0].id:
                    break
                try:
                    current = ModelArtifact.model_validate(
                        ModelTasks().load(project, current.parent_model_id)["artifact"]
                    )
                except (ValueError, OSError):
                    break
            confirmed = artifacts[0].id in lineage
            value["lineage"] = {"confirmed": confirmed, "parent_chain": lineage}
            if not confirmed:
                value["warnings"].append(
                    "Model source relationship cannot be confirmed; metrics only describe these explicitly selected artifacts"
                )
            dataset = load_dataset(project, config.dataset_id)
            value["dataset"] = dataset
            if dataset["role"] != "diagnostic":
                raise ValueError(
                    "Select a dataset with diagnostic role; final evaluation is never a diagnostic input"
                )
            for other in RunStore(project).metadata_list("dataset:"):
                if other["id"] == dataset["id"]:
                    continue
                if dataset_overlap(dataset, other):
                    if other["role"] == "final_evaluation":
                        raise ValueError(
                            "Diagnostic data overlaps a held-out final evaluation snapshot"
                        )
                    value["warnings"].append(
                        f"Diagnostic data overlaps {other['role']} snapshot {other['id'][:8]}; it is not an independent split"
                    )
            value["warnings"].extend(dataset["warnings"])
            observed = analyze_quantization(verified[1][1])
            value["quantization"] = observed
            if artifacts[1].experiment_id:
                experiment = (
                    experiment_folder(project, artifacts[1].experiment_id, exists=False)
                    / "experiment.json"
                )
                if experiment.is_file() and experiment.stat().st_size <= 24 * MIB:
                    value["quantization"]["source_optimization_config"] = json.loads(
                        experiment.read_text(encoding="utf-8")
                    )["config"]["optimization"]
            if observed["representation"] != "QDQ":
                raise ValueError(
                    f"Unsupported quantization format {observed['representation']}; this engine requires QDQ, and does not reinterpret QOperator tensors"
                )
            temporary.mkdir(exist_ok=False)
            copies = [
                model_copy(project, a, temporary / name)
                for a, name in zip(artifacts, ("fp32", "quantized"), strict=True)
            ]
            models = [onnx.shape_inference.infer_shapes(onnx.load(p)) for p in copies]
            if any(
                v.name.endswith(POSTFIX)
                for m in models
                for v in [*m.graph.input, *m.graph.output, *m.graph.value_info]
            ):
                raise ValueError(
                    "Model tensor names collide with the ORT diagnostic capture suffix"
                )
            reference, candidate = [GraphIndex(m) for m in models]
            sessions = [
                ort.InferenceSession(
                    str(p), sess_options=session_options(), providers=[config.provider]
                )
                for p in copies
            ]
            feeds = DatasetFeeds(dataset)
            selected_count = min(config.max_samples, len(feeds))
            if len(feeds) > selected_count:
                value["warnings"].append(
                    f"Sample limit selects the first {selected_count}/{len(feeds)} saved batches in manifest order"
                )
            matcher = TensorMatcher(reference, candidate)
            rows, plans, scope_count = matcher.plan(config)
            value["records"] = rows
            # Preflight every chosen feed before exposing any intermediate output.
            for index in range(selected_count):
                feed = feeds[index]
                for session in sessions:
                    check_feed(session.get_inputs(), feed)
                for row in rows:
                    if row["id"] not in plans:
                        continue
                    plan = plans[row["id"]]
                    try:
                        a = concrete_shape(reference, plan["reference"], feed)
                        b = concrete_shape(candidate, plan["post"], feed)
                        if a != b:
                            raise ValueError("Inferred concrete shapes differ between models")
                        concrete_shape(candidate, plan["pre"], feed)
                    except ValueError as exc:
                        row.update(match_status="Unsupported", message=f"Feed {index}: {exc}")
                        plans.pop(row["id"])
            left_names = {p["reference"] for p in plans.values()}
            right_names = {n for p in plans.values() for n in (p["pre"], p["post"])}
            value["resources"].update(
                selected_batches=selected_count,
                exposed_fp32_tensors=len(left_names),
                exposed_quantized_tensors=len(right_names),
            )
            peak = 0
            for index in range(selected_count):
                feed = feeds[index]
                largest = max(
                    (math.prod(concrete_shape(reference, n, feed)) for n in left_names), default=0
                )
                estimate = (
                    estimated_bytes(reference, left_names, feed)
                    + estimated_bytes(candidate, right_names, feed)
                    + largest * 96
                    + sum(v.nbytes for v in feed.values())
                )
                peak = max(peak, estimate)
            value["resources"]["working_bytes_estimate"] = peak
            if peak > config.memory_budget_mb * MIB:
                raise ValueError(
                    "Selected capture/statistics exceeds the working memory budget before inference; select fewer/smaller tensors"
                )
            progress(
                "instrument",
                0,
                f"Preparing {len(left_names)} FP32 + {len(right_names)} QDQ capture outputs; original artifacts remain unchanged",
            )
            augmented = [temporary / "fp32-capture.onnx", temporary / "quantized-capture.onnx"]
            if plans:
                available = [
                    augment_model(p, out, names)
                    for p, out, names in zip(
                        copies, augmented, (left_names, right_names), strict=True
                    )
                ]
                for row in rows:
                    if row["id"] in plans:
                        p = plans[row["id"]]
                        if (
                            p["reference"] not in available[0]
                            or not {p["pre"], p["post"]} <= available[1]
                        ):
                            row.update(
                                match_status="Unsupported",
                                message="Installed ORT instrumenter did not expose this tensor",
                            )
                            plans.pop(row["id"])
            del sessions
            for index in range(selected_count):
                if not plans:
                    break
                feed = feeds[index]
                progress(
                    "capture",
                    index / selected_count,
                    f"Collecting the same feed {index + 1}/{selected_count} on both models",
                )
                examples = next((v.shape[0] for v in feed.values() if v.ndim), 1)
                capture = {
                    "sample_index": index,
                    "feed_fingerprint": content_digest(feed),
                    "example_count": int(examples),
                    "fp32_tensors": 0,
                    "quantized_tensors": 0,
                    "captured_bytes": 0,
                    "error": None,
                }
                try:
                    left, right = [collect(p, feed, config.provider) for p in augmented]
                    byte_count = sum(
                        v.nbytes for d in (left, right) for arrays in d.values() for v in arrays
                    )
                    if byte_count > config.memory_budget_mb * MIB:
                        raise MemoryError("Actual capture exceeds its declared tensor budget")
                    capture.update(
                        fp32_tensors=len(left),
                        quantized_tensors=len(right),
                        captured_bytes=byte_count,
                    )
                    actual_matches = debug.create_activation_matching(right, left)
                    for row in rows:
                        if row["id"] not in plans:
                            continue
                        row["samples"].append(
                            compare_capture(
                                plans[row["id"]],
                                reference,
                                candidate,
                                feed,
                                left,
                                right,
                                actual_matches,
                                index,
                                examples,
                            )
                        )
                    del left, right, actual_matches
                except (MemoryError, InterruptedError, TimeoutError):
                    raise
                except Exception as exc:
                    capture["error"] = f"{type(exc).__name__}: {exc}"
                    for row in rows:
                        if row["id"] in plans:
                            row["samples"].append(
                                {
                                    "sample_index": index,
                                    "metrics": {"valid": False, "reason": capture["error"]},
                                }
                            )
                    value["warnings"].append(
                        f"Feed {index} failed on the diagnostic models: {capture['error']}"
                    )
                value["captures"].append(capture)
                self.finish_rows(rows)
                progress(
                    "capture",
                    (index + 1) / selected_count,
                    f"Reduced feed {index + 1}/{selected_count} to statistics; raw activation arrays discarded",
                )
            if config.include_weights:
                chosen = {r["fp32_node_id"] for r in rows}
                value["records"].extend(
                    weight_records(reference, candidate, chosen, config, temporary, progress)
                )
            self.finish_rows(value["records"])
            value["matching_summary"] = matching_summary(value["records"], scope_count)
            if not value["matching_summary"]["comparable_tensors"]:
                raise ValueError(
                    "No comparable activation tensors were captured; see per-tensor matching/resource reasons"
                )
            progress(
                "complete",
                1,
                "Saved activation/weight statistics and honest matching coverage; diagnostic elapsed time is not Benchmark latency",
            )
            value.update(success=True, status="Succeeded")
        except Exception as exc:
            value.update(
                status="Cancelled" if isinstance(exc, InterruptedError) else "Failed",
                success=False,
                error=f"{type(exc).__name__}: {exc}",
            )
            value["diagnostics"] = [{"severity": "error", "message": value["error"]}]
            value["logs"].append("[failed] " + value["error"])
        finally:
            value["resources"]["temporary_cleaned"] = cleanup_temporary(folder)
            value["finished_at"], value["elapsed_seconds"] = time.time(), time.monotonic() - start
            for artifact, _ in verified:
                try:
                    verify_artifact(project, artifact)
                except (ValueError, OSError) as exc:
                    value.update(
                        success=False,
                        status="Failed",
                        error=f"Source artifact changed/missing: {exc}",
                    )
                    value["diagnostics"] = [{"severity": "error", "message": value["error"]}]
            value["matching_summary"] = matching_summary(value["records"], scope_count)
            save_diagnostic(folder, value)
        return DiagnosticResult.model_validate(value).model_dump()

    @staticmethod
    def finish_rows(rows):
        for row in rows:
            if not row["samples"]:
                continue
            row["aggregate"] = aggregate_metrics(row["samples"])
            valid = row["aggregate"]["valid_samples"]
            if not valid:
                row.update(
                    match_status="Unsupported",
                    message=row["samples"][0]["metrics"].get("reason", "No valid captured samples"),
                )
            elif valid < len(row["samples"]):
                row.update(
                    match_status="Partially Matched",
                    message="Some feeds were not comparable; aggregate excludes invalid samples",
                )
