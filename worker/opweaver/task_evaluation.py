"""Task evaluators use explicit labels and output contracts; never invent accuracy."""

from abc import ABC, abstractmethod

import numpy as np

from .datasets import sample_arrays


class TaskEvaluator(ABC):
    @abstractmethod
    def evaluate(self, reference, candidate, dataset, config):
        """Return observed metrics and their sample scope."""


class DatasetEvaluator(ABC):
    @abstractmethod
    def evaluate_model(self, session, dataset, config, progress):
        """Evaluate one model against an explicit dataset contract."""


class ClassificationEvaluator(DatasetEvaluator):
    def evaluate_model(self, session, dataset, config, progress=lambda value, message: None):
        if not dataset or not config.output_name:
            raise ValueError("Top1 requires a labeled evaluation dataset and explicit output name")
        count, correct, top5_correct, classes = 0, 0, 0, None
        batches = []
        for index, sample in enumerate(dataset["samples"]):
            feed, labels = sample_arrays(dataset["manifest_path"], sample)
            if labels is None:
                raise ValueError("Top1 evaluation requires labels for every batch")
            a = session.run([config.output_name], feed)[0]
            if (
                a.ndim != 2
                or labels.shape != (a.shape[0],)
                or labels.dtype.kind not in "iu"
                or np.any(labels < 0)
                or np.any(labels >= a.shape[1])
                or a.dtype.kind != "f"
                or a.shape[1] < 1
            ):
                raise ValueError(
                    "Top1 requires [batch, classes] scores and valid [batch] class labels"
                )
            if not np.all(np.isfinite(a)):
                raise ValueError("Top1 scores contain non-finite values")
            if classes is not None and classes != a.shape[1]:
                raise ValueError("Classification output class count changed across batches")
            classes = a.shape[1]
            if any(v >= classes for v in dataset.get("class_mapping", {}).values()):
                raise ValueError("Declared class mapping exceeds model score columns")
            predicted = np.argmax(a, axis=1)
            batch_correct = int(np.count_nonzero(predicted == labels))
            correct += batch_correct
            if classes >= 5:
                top5 = np.argsort(-a, axis=1, kind="stable")[:, :5]
                top5_correct += int(np.count_nonzero(np.any(top5 == labels[:, None], axis=1)))
            batches.append(
                {
                    "batch": index,
                    "samples": int(labels.size),
                    "correct": batch_correct,
                    "labels": labels.tolist(),
                    "predictions": predicted.tolist(),
                }
            )
            count += labels.size
            progress(
                (index + 1) / len(dataset["samples"]),
                f"Classification evaluation batch {index + 1}/{len(dataset['samples'])}",
            )
        if not count:
            raise ValueError("Classification evaluation has no labeled samples")
        return {
            "status": "evaluated",
            "task": "classification",
            "samples": count,
            "classes": classes,
            "correct": correct,
            "incorrect": count - correct,
            "top1_accuracy": correct / count,
            "top5_accuracy": top5_correct / count if classes >= 5 else None,
            "top5_correct": top5_correct if classes >= 5 else None,
            "top5_status": "evaluated" if classes >= 5 else "not_applicable_fewer_than_5_classes",
            "class_mapping": dataset.get("class_mapping", {}),
            "label_contract": "Integer labels are explicit model score column indices; image CSV labels use the saved class mapping",
            "postprocessing": "argmax/top-k on finite [batch,classes] scores; ties use first column; no implicit softmax/class reordering",
            "batches": batches,
            "dataset_id": dataset["id"],
            "dataset_fingerprint": dataset["fingerprint"],
            "scope": "Supplied labels and recorded preprocessing only; representativeness is not inferred from accuracy",
        }


class ClassificationTop1(TaskEvaluator):
    def evaluate(self, reference, candidate, dataset, config, progress=lambda value, message: None):
        evaluator = ClassificationEvaluator()
        a = evaluator.evaluate_model(reference, dataset, config, progress)
        b = evaluator.evaluate_model(candidate, dataset, config, progress)
        accuracy_a, accuracy_b = a["top1_accuracy"], b["top1_accuracy"]
        return {
            "status": "evaluated",
            "task": "classification_top1",
            "samples": a["samples"],
            "baseline_accuracy": accuracy_a,
            "optimized_accuracy": accuracy_b,
            "accuracy_drop": accuracy_a - accuracy_b,
            "accuracy_drop_percentage_points": (accuracy_a - accuracy_b) * 100,
            "accuracy_change_percent": (accuracy_b / accuracy_a - 1) * 100 if accuracy_a else None,
            "baseline_top5_accuracy": a["top5_accuracy"],
            "optimized_top5_accuracy": b["top5_accuracy"],
            "baseline_metrics": a,
            "optimized_metrics": b,
            "scope": "Top1 against supplied labels only; label provenance and dataset representativeness are user responsibilities",
        }


def evaluate_single(session, dataset, config, progress=lambda value, message: None):
    if config.task == "none":
        return {"status": "not_evaluated", "reason": "No evaluator/label contract selected"}
    return ClassificationEvaluator().evaluate_model(session, dataset, config, progress)


def evaluate_task(reference, candidate, dataset, config, progress=lambda value, message: None):
    if config.task == "none":
        return {
            "status": "not_evaluated",
            "reason": "No task evaluator/ground-truth contract selected; numerical errors are not accuracy",
        }
    return ClassificationTop1().evaluate(reference, candidate, dataset, config, progress)


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
