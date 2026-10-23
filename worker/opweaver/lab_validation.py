"""Output-aware numerical validation, separate from task-level quality evaluation."""

import numpy as np

from .lab_types import ValidationResult


def compare_tensor(reference, candidate, atol, rtol):
    result = {
        "passed": False,
        "execution_valid": False,
        "max_abs_error": None,
        "mean_abs_error": None,
        "max_relative_error": None,
    }
    if not isinstance(reference, np.ndarray) or not isinstance(candidate, np.ndarray):
        return {**result, "reason": "Unsupported non-tensor output"}
    result.update(shape=list(reference.shape), dtype=str(reference.dtype), elements=reference.size)
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        return {
            **result,
            "reason": f"Shape/dtype mismatch: candidate {candidate.shape}/{candidate.dtype}",
        }
    if reference.dtype.kind in "biu":
        return {
            **result,
            "passed": bool(np.array_equal(reference, candidate)),
            "execution_valid": True,
            "changed_elements": int(np.count_nonzero(reference != candidate)),
            "comparison": "exact integer/boolean",
            "reason": "Exact comparison without floating-point conversion",
        }
    if reference.dtype.kind != "f":
        return {**result, "reason": f"Unsupported output dtype {reference.dtype}"}
    if not reference.size:
        return {
            **result,
            "passed": True,
            "execution_valid": True,
            "changed_elements": 0,
            "max_abs_error": 0.0,
            "mean_abs_error": 0.0,
            "reason": "Empty tensor with matching shape/dtype",
        }
    a, b = reference.astype(np.float64), candidate.astype(np.float64)
    nan_count = int(np.count_nonzero(np.isnan(a) | np.isnan(b)))
    infinity = np.isinf(a) | np.isinf(b)
    inf_mismatch = int(np.count_nonzero(infinity & (a != b)))
    finite = np.isfinite(a) & np.isfinite(b)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        error = np.abs(a[finite] - b[finite])
        ref = np.abs(a[finite])
        passed = (
            not nan_count
            and not inf_mismatch
            and bool(np.all(np.isfinite(error)) and np.all(error <= atol + rtol * ref))
        )
        relative = error[ref != 0] / ref[ref != 0]
        maximum = float(np.max(error)) if error.size else 0.0
        mean = float(np.mean(error)) if error.size else 0.0
        max_relative = float(np.max(relative)) if relative.size else None
    return {
        **result,
        "passed": passed,
        "execution_valid": not nan_count and not inf_mismatch and bool(np.all(np.isfinite(error))),
        "changed_elements": int(np.count_nonzero(error > atol + rtol * ref)),
        "comparison": "floating tolerance against reference",
        "max_abs_error": maximum if np.isfinite(maximum) else None,
        "mean_abs_error": mean if np.isfinite(mean) else None,
        "max_relative_error": max_relative
        if max_relative is not None and np.isfinite(max_relative)
        else None,
        "zero_reference_mismatches": int(np.count_nonzero((ref == 0) & (error != 0))),
        "nan_count": nan_count,
        "nonfinite_count": int(np.count_nonzero(~finite)),
        "inf_mismatches": inf_mismatch,
        "reason": "NaN always fails; matching signed Inf passes; errors use finite pairs; relative error excludes zero reference values",
    }


def validate_outputs(reference, candidate, samples, config, progress=lambda value, message: None):
    names = [v.name for v in reference.get_outputs()]
    other = [v.name for v in candidate.get_outputs()]
    results = []
    if names != other:
        results.append(
            {
                "passed": False,
                "reason": "Output names/count/order differ",
                "reference": names,
                "candidate": other,
            }
        )
    else:
        for i, feed in enumerate(samples):
            expected = reference.run(None, feed)
            actual = candidate.run(None, feed)
            if len(expected) != len(actual) or len(expected) != len(names):
                results.append({"passed": False, "sample": i, "reason": "Output count differs"})
                break
            results.extend(
                {"sample": i, "name": name, **compare_tensor(a, b, config.atol, config.rtol)}
                for name, a, b in zip(names, expected, actual, strict=True)
            )
            progress((i + 1) / len(samples), f"Validated sample {i + 1}/{len(samples)}")
    return ValidationResult(
        passed=bool(results) and all(r["passed"] for r in results),
        samples=len(samples),
        atol=config.atol,
        rtol=config.rtol,
        outputs=results,
        execution_valid=bool(results) and all(r.get("execution_valid", False) for r in results),
        numerical_summary=numerical_summary(results),
    )


def numerical_summary(results):
    count = sum(r.get("elements", 0) for r in results)
    measured = [r for r in results if r.get("mean_abs_error") is not None]
    measured_count = sum(r.get("elements", 0) for r in measured)
    return {
        "elements": count,
        "mean_abs_error": sum(r["mean_abs_error"] * r.get("elements", 0) for r in measured)
        / measured_count
        if measured_count
        else None,
        "max_abs_error": max((r["max_abs_error"] for r in measured), default=None),
        "max_relative_error": max(
            (r["max_relative_error"] for r in results if r.get("max_relative_error") is not None),
            default=None,
        ),
        "changed_fraction": sum(r.get("changed_elements", 0) for r in results) / count
        if count
        else None,
        "changed_definition": "Fraction outside configured atol/rtol (integers: unequal)",
    }
