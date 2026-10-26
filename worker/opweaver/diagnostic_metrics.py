"""Finite, shape-aware errors and explicit element-weighted multi-feed aggregation."""

import math

import numpy as np

from .diagnostic_types import ErrorMetrics


def tensor_metrics(reference, candidate):
    result = ErrorMetrics()
    if not isinstance(reference, np.ndarray) or not isinstance(candidate, np.ndarray):
        result.reason = "Unsupported non-tensor value"
    elif reference.shape != candidate.shape:
        result.reason = "Shape mismatch"
    elif reference.dtype != candidate.dtype or reference.dtype.kind != "f":
        result.reason = "DType mismatch or unsupported non-floating tensor; raw integer codes are not comparable"
    elif not reference.size:
        result.reason = "Empty tensor: error and cosine are undefined"
    elif not np.all(np.isfinite(reference)) or not np.all(np.isfinite(candidate)):
        result.reason = "NaN/Inf in captured tensor"
    else:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            a, b = (
                reference.astype(np.float64).reshape(-1),
                candidate.astype(np.float64).reshape(-1),
            )
            error = np.abs(a - b)
            mse = float(np.mean(error**2))
            nonzero = a != 0
            relative = error[nonzero] / np.abs(a[nonzero])
            norm_a, norm_b = float(np.linalg.norm(a)), float(np.linalg.norm(b))
            cosine = float(np.dot(a / norm_a, b / norm_b)) if norm_a and norm_b else None
            metrics = {
                "mae": float(np.mean(error)),
                "mse": mse,
                "rmse": math.sqrt(mse),
                "max_abs_error": float(np.max(error)),
                "mean_relative_error": float(np.mean(relative)) if relative.size else None,
                "max_relative_error": float(np.max(relative)) if relative.size else None,
                "cosine_similarity": max(-1.0, min(1.0, cosine)) if cosine is not None else None,
            }
        if any(
            v is not None and not math.isfinite(v) for v in metrics.values()
        ) or not math.isfinite(norm_a + norm_b):
            result.reason = "Floating-point overflow in error statistics"
        else:
            result = ErrorMetrics(
                valid=True,
                elements=int(a.size),
                relative_defined_elements=int(relative.size),
                zero_reference_elements=int(a.size - relative.size),
                **metrics,
            )
    return result.model_dump()


def aggregate_metrics(samples):
    valid = [s["metrics"] for s in samples if s.get("metrics", {}).get("valid")]
    elements = sum(v["elements"] for v in valid)
    relative_n = sum(v["relative_defined_elements"] for v in valid)
    cosine = [v["cosine_similarity"] for v in valid if v["cosine_similarity"] is not None]
    result = {
        "samples": len(samples),
        "valid_samples": len(valid),
        "invalid_samples": len(samples) - len(valid),
        "elements": elements,
        "relative_defined_elements": relative_n,
        "aggregation": "MAE/MSE/relative mean are weighted by valid elements; RMSE=sqrt(global MSE); max is across all valid elements; cosine is mean of defined per-feed cosines",
        "mean_sample_mae": None,
        "mae": None,
        "mse": None,
        "rmse": None,
        "max_abs_error": None,
        "mean_relative_error": None,
        "max_relative_error": None,
        "cosine_similarity": None,
        "defined_cosine_samples": len(cosine),
    }
    if elements:
        result.update(
            mae=sum(v["mae"] * (v["elements"] / elements) for v in valid),
            mse=sum(v["mse"] * (v["elements"] / elements) for v in valid),
            max_abs_error=max(v["max_abs_error"] for v in valid),
            mean_sample_mae=sum(v["mae"] / len(valid) for v in valid),
            cosine_similarity=sum(cosine) / len(cosine) if cosine else None,
        )
        result["rmse"] = math.sqrt(result["mse"])
        if relative_n:
            result["mean_relative_error"] = sum(
                (v["mean_relative_error"] or 0) * (v["relative_defined_elements"] / relative_n)
                for v in valid
            )
            result["max_relative_error"] = max(
                v["max_relative_error"] for v in valid if v["max_relative_error"] is not None
            )
    return result
