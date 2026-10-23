"""Backend-independent, immutable NPZ dataset snapshots; one batch resident at a time."""

import hashlib
import json
import re
import time
import uuid
from pathlib import Path

import numpy as np

from .benchmark import create_session, digest_file
from .lab_inputs import check_feed, load_npz, prepare_inputs
from .lab_reports import verify_artifact, write_json
from .lab_types import BenchmarkConfig, DatasetConfig, InputConfig
from .model_artifact import ModelArtifact
from .model_tasks import ModelTasks
from .storage import RunStore

MAX_DATASET_BYTES = 1024**3


def content_digest(feed):
    digest = hashlib.sha256()
    for name, array in sorted(feed.items()):
        digest.update(json.dumps([name, str(array.dtype), list(array.shape)]).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def dataset_folder(project, id):
    if not re.fullmatch(r"[0-9a-f]{32}", id):
        raise ValueError("Invalid dataset ID")
    root = Path(project).resolve(strict=True)
    folder = (root / ".opweaver" / "datasets" / id).resolve()
    if not folder.is_relative_to(root):
        raise ValueError("Dataset path escapes the project")
    return folder


def sample_arrays(manifest_path, sample):
    root = Path(manifest_path).resolve(strict=True).parent
    path = (root / sample["path"]).resolve(strict=True)
    if not path.is_relative_to(root) or digest_file(path) != sample["sha256"]:
        raise ValueError("Dataset sample is missing or changed; create a new snapshot")
    arrays = load_npz(path)
    labels = arrays.pop("__opweaver_labels__", None)
    if content_digest(arrays) != sample["content_sha256"]:
        raise ValueError("Dataset content fingerprint changed")
    return arrays, labels


def load_manifest(path, expected_hash=None):
    path = Path(path).resolve(strict=True)
    if path.stat().st_size > 4 * 1024**2:
        raise ValueError("Dataset manifest exceeds 4 MiB")
    if expected_hash and digest_file(path) != expected_hash:
        raise ValueError("Dataset manifest changed; create a new snapshot")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "1.0" or not 1 <= len(manifest["samples"]) <= 512:
        raise ValueError("Unsupported or empty dataset manifest")
    return manifest


def load_dataset(project, id, verify=True):
    records = RunStore(Path(project)).metadata_list("dataset:")
    record = next((r for r in records if r["id"] == id), None)
    if not record:
        raise ValueError(f"Dataset {id} is not registered in this project")
    folder = dataset_folder(project, id)
    path = (folder / "manifest.json").resolve(strict=True)
    if not path.is_relative_to(folder):
        raise ValueError("Dataset manifest escapes the snapshot directory")
    manifest = load_manifest(path, record["manifest_sha256"])
    if verify:
        for sample in manifest["samples"]:
            sample_arrays(path, sample)
    return {**manifest, "manifest_path": str(path), "manifest_sha256": record["manifest_sha256"]}


class DatasetFeeds:
    def __init__(self, record):
        self.record = record

    def __len__(self):
        return len(self.record["samples"])

    def __iter__(self):
        for sample in self.record["samples"]:
            yield sample_arrays(self.record["manifest_path"], sample)[0]

    def __getitem__(self, index):
        return sample_arrays(self.record["manifest_path"], self.record["samples"][index])[0]


def create_dataset(project, raw):
    config = DatasetConfig.model_validate(raw)
    if len(config.files) > 512:
        raise ValueError("A dataset supports at most 512 NPZ batches")
    model = ModelTasks().load(project, config.source_model_id)
    artifact = ModelArtifact.model_validate(model["artifact"])
    session, _ = create_session(verify_artifact(project, artifact), BenchmarkConfig())
    metadata = session.get_inputs()
    names = {v.name for v in metadata}
    if "__opweaver_labels__" in names:
        raise ValueError("Reserved dataset label name conflicts with model input")
    mapping = config.mapping or {item.name: item.name for item in metadata}
    if set(mapping) != names or len(set(mapping.values())) != len(names):
        raise ValueError("Mapping must assign every model input to a distinct NPZ key")
    if config.label_key in mapping.values():
        raise ValueError("The label key must differ from input keys")
    if config.mode == "npz" and len(config.files) != config.count:
        raise ValueError("NPZ file count must equal the requested batch count")
    if config.mode == "random" and config.label_key:
        raise ValueError("Synthetic inputs have no ground-truth labels")
    id = uuid.uuid4().hex
    folder = dataset_folder(project, id)
    folder.mkdir(parents=True, exist_ok=False)
    samples, total, examples = [], 0, 0
    deadline = time.monotonic() + 45
    for index in range(config.count):
        if time.monotonic() > deadline:
            raise TimeoutError("Dataset preparation exceeded 45 seconds; use smaller batches")
        if config.mode == "random":
            feeds, _ = prepare_inputs(
                metadata,
                InputConfig(mode="random", shapes=config.shapes, seed=config.seed + index),
                1,
                folder / "generated",
                BenchmarkConfig(),
            )
            feed, labels, source = feeds[0], None, "synthetic seeded input"
        else:
            source = str(Path(config.files[index]).resolve(strict=True))
            arrays = load_npz(Path(source))
            expected = set(mapping.values()) | ({config.label_key} if config.label_key else set())
            if set(arrays) != expected:
                raise ValueError(f"NPZ keys differ from input mapping/label key: {source}")
            feed = {name: arrays[key] for name, key in mapping.items()}
            labels = arrays.get(config.label_key) if config.label_key else None
        check_feed(metadata, feed, config.shapes)
        for name, array in feed.items():
            if not array.size or not np.all(np.isfinite(array)):
                raise ValueError(f"Input {name} contains empty/non-finite data")
        batch = next((feed[item.name].shape[0] for item in metadata if feed[item.name].ndim), 1)
        if labels is not None and (
            labels.dtype.kind not in "iu" or labels.shape != (batch,) or np.any(labels < 0)
        ):
            raise ValueError("Classification labels must be nonnegative integer [batch] tensors")
        total += sum(v.nbytes for v in feed.values()) + (labels.nbytes if labels is not None else 0)
        if total > MAX_DATASET_BYTES:
            raise ValueError("Dataset exceeds the 1 GiB tensor budget")
        target = folder / f"batch-{index:04d}.npz"
        np.savez(target, **feed, **({"__opweaver_labels__": labels} if labels is not None else {}))
        samples.append(
            {
                "index": index,
                "path": target.name,
                "sha256": digest_file(target),
                "content_sha256": content_digest(feed),
                "source": source,
                "tensors": [
                    {
                        "name": n,
                        "shape": list(v.shape),
                        "dtype": str(v.dtype),
                        "min": float(v.min()),
                        "max": float(v.max()),
                        "mean": float(np.mean(v, dtype=np.float64)),
                    }
                    for n, v in feed.items()
                ],
                "labels": int(labels.size) if labels is not None else 0,
            }
        )
        examples += batch
    warnings = []
    if config.count < 32:
        warnings.append(
            "Small dataset: calibration ranges and quality may not represent production data"
        )
    if config.mode == "random":
        warnings.append(
            "Synthetic seeded data; representativeness and task accuracy are not established"
        )
    manifest = {
        "schema_version": "1.0",
        "id": id,
        "name": config.name,
        "format": "npz-batches-v1",
        "role": config.role,
        "config": config.model_dump(),
        "model_hash": artifact.sha256,
        "created_at": time.time(),
        "batch_count": len(samples),
        "example_count": examples,
        "example_count_scope": "Axis 0 of the first non-scalar model input; scalar-only feeds count as one example per batch",
        "tensor_bytes": total,
        "fingerprint": hashlib.sha256(
            "".join(s["content_sha256"] for s in samples).encode()
        ).hexdigest(),
        "samples": samples,
        "warnings": warnings,
    }
    write_json(folder / "manifest.json", manifest)
    record = {
        **manifest,
        "manifest_path": str(folder / "manifest.json"),
        "manifest_sha256": digest_file(folder / "manifest.json"),
    }
    RunStore(Path(project)).set_metadata(f"dataset:{id}", record)
    return record


def dataset_pair(project, config):
    calibration = (
        load_dataset(project, config.optimization.calibration_dataset_id)
        if config.optimization.calibration_dataset_id
        else None
    )
    evaluation = (
        load_dataset(project, config.evaluation.dataset_id)
        if config.evaluation.dataset_id
        else None
    )
    overlap = []
    if calibration and evaluation:
        left = {s["content_sha256"] for s in calibration["samples"]}
        overlap = sorted(left & {s["content_sha256"] for s in evaluation["samples"]})
        if overlap and not config.evaluation.allow_calibration_reuse:
            raise ValueError(
                "Calibration/evaluation samples overlap; choose independent data or explicitly allow reuse"
            )
    return {
        "calibration": calibration,
        "evaluation": evaluation,
        "independent": bool(calibration and evaluation and not overlap),
        "overlapping_batches": len(overlap),
        "warning": "Reused calibration data is not independent evaluation" if overlap else None,
    }


def availability(project, datasets):
    result = {}
    for role in ("calibration", "evaluation"):
        item = (datasets or {}).get(role)
        if item:
            try:
                current = load_dataset(project, item["id"])
                if current["manifest_sha256"] != item["manifest_sha256"]:
                    raise ValueError("Recorded dataset manifest changed")
                result[role] = {"available": True}
            except (OSError, ValueError, KeyError) as exc:
                result[role] = {"available": False, "reason": str(exc)}
    return result
