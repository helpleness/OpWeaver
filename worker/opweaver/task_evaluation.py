"""Task evaluators use explicit labels and output contracts; never invent accuracy."""

from abc import ABC, abstractmethod

import numpy as np

from .datasets import sample_arrays


class TaskEvaluator(ABC):
    @abstractmethod
    def evaluate(self, reference, candidate, dataset, config):
        """Return observed metrics and their sample scope."""


class ClassificationTop1(TaskEvaluator):
    def evaluate(self, reference, candidate, dataset, config):
        if not dataset or not config.output_name:
            raise ValueError("Top1 requires a labeled evaluation dataset and explicit output name")
        count, correct_a, correct_b = 0, 0, 0
        for sample in dataset["samples"]:
            feed, labels = sample_arrays(dataset["manifest_path"], sample)
            if labels is None:
                raise ValueError("Top1 evaluation requires labels for every batch")
            a = reference.run([config.output_name], feed)[0]
            b = candidate.run([config.output_name], feed)[0]
            if (
                a.ndim != 2
                or a.shape != b.shape
                or labels.shape != (a.shape[0],)
                or np.any(labels >= a.shape[1])
            ):
                raise ValueError(
                    "Top1 requires [batch, classes] scores and valid [batch] class labels"
                )
            if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
                raise ValueError("Top1 scores contain non-finite values")
            correct_a += int(np.count_nonzero(np.argmax(a, axis=1) == labels))
            correct_b += int(np.count_nonzero(np.argmax(b, axis=1) == labels))
            count += labels.size
        return {
            "status": "evaluated",
            "task": "classification_top1",
            "samples": count,
            "baseline_accuracy": correct_a / count,
            "optimized_accuracy": correct_b / count,
            "accuracy_drop": (correct_a - correct_b) / count,
            "scope": "Top1 against supplied labels only; label provenance and dataset representativeness are user responsibilities",
        }


def evaluate_task(reference, candidate, dataset, config):
    if config.task == "none":
        return {
            "status": "not_evaluated",
            "reason": "No task evaluator/ground-truth contract selected; numerical errors are not accuracy",
        }
    return ClassificationTop1().evaluate(reference, candidate, dataset, config)


def quality_gate(summary, task, gates):
    metrics = {
        "max_mean_abs_error": summary["mean_abs_error"],
        "max_abs_error": summary["max_abs_error"],
        "max_changed_fraction": summary["changed_fraction"],
        "max_accuracy_drop": task.get("accuracy_drop"),
    }
    checks = [
        {
            "metric": name,
            "limit": limit,
            "value": metrics[name],
            "passed": metrics[name] is not None and metrics[name] <= limit,
        }
        for name, limit in gates.model_dump().items()
        if name != "schema_version" and limit is not None
    ]
    return {
        "status": "not_configured"
        if not checks
        else "passed"
        if all(c["passed"] for c in checks)
        else "failed",
        "checks": checks,
        "scope": "User thresholds; failure retains the executable artifact and benchmark results",
    }
