"""Single-device training, cancellation and exact batch-boundary checkpoint resume."""

import hashlib
import json
import os
import platform
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import torch
from pydantic import Field
from torch import nn

from .builder import GraphModule
from .ir import Graph, Spec
from .storage import RunStore


class TrainingConfig(Spec):
    epochs: int = Field(default=20, ge=1, le=10000)
    batch_size: int = Field(default=32, ge=1, le=65536)
    learning_rate: float = Field(default=0.01, gt=0, le=10)
    optimizer: Literal["AdamW", "SGD"] = "AdamW"
    loss: Literal["CrossEntropyLoss", "MSELoss"] = "CrossEntropyLoss"
    samples: int = Field(default=512, ge=16, le=1000000)
    seed: int = Field(default=42, ge=0, le=2147483647)
    device: Literal["cpu", "cuda"] = "cpu"


def environment() -> dict:
    available = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "pytorch": str(torch.__version__),
        "cuda_available": available,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if available else "CPU",
        "notice": "CUDA available; CPU is the default."
        if available
        else "CUDA is unavailable. CPU modeling and training remain available.",
    }


def graph_digest(graph: Graph) -> str:
    return hashlib.sha256(json.dumps(graph.model_dump(), sort_keys=True).encode()).hexdigest()


def synthetic_data(graph: Graph, config: TrainingConfig):
    inputs = [n for n in graph.nodes if n.operator.id == "core.input"]
    outputs = [n for n in graph.nodes if n.operator.id == "core.output"]
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError(
            "The synthetic training adapter requires one Input and one Output. The model builder supports multiple inputs/outputs."
        )
    shape = inputs[0].params.get("shape", ["B", 8])
    if len(shape) != 2 or shape[0] != "B" or not isinstance(shape[1], int):
        raise ValueError("Synthetic training requires input shape ['B', features]")
    from .registry import builtin_registry
    from .validation import validate

    report = validate(graph, builtin_registry())
    spec = report.tensors.get(f"{outputs[0].id}:in", {})
    output_shape = spec.get("shape", [])
    if len(output_shape) != 2 or output_shape[0] != "B" or not isinstance(output_shape[1], int):
        raise ValueError("Synthetic training requires output shape ['B', classes/targets]")
    count = output_shape[1]
    if config.loss == "CrossEntropyLoss" and count < 2:
        raise ValueError("Classification requires at least two output classes")
    generator = torch.Generator().manual_seed(config.seed + 731)
    x = torch.randn(config.samples, shape[1], generator=generator)
    teacher = torch.randn(shape[1], count, generator=generator)
    y = x @ teacher
    if config.loss == "CrossEntropyLoss":
        y = y.argmax(dim=1)
    return x, y


class TrainingEngine:
    def __init__(self, emit: Callable[[dict], None] | None = None):
        self.emit = emit or (lambda event: None)
        self.lock = threading.RLock()
        self.jobs: dict[str, dict] = {}
        self.cancels: dict[str, threading.Event] = {}
        self.threads: dict[str, threading.Thread] = {}
        self.stores: dict[str, RunStore] = {}

    def active(self) -> bool:
        with self.lock:
            return any(thread.is_alive() for thread in self.threads.values())

    def _update(self, id: str, kind: str = "state", **values) -> None:
        with self.lock:
            if (
                values.get("state") in ("VALIDATING", "QUEUED", "RUNNING")
                and self.cancels[id].is_set()
            ):
                values["state"] = "STOPPING"
            self.jobs[id].update(values)
            snapshot = json.loads(json.dumps(self.jobs[id]))
        self.stores[id].save(snapshot)
        self.emit({"event": kind, "job_id": id, "data": snapshot})

    def start(
        self,
        graph: Graph,
        config: TrainingConfig,
        project: Path,
        resume: str | None = None,
    ) -> dict:
        with self.lock:
            if self.active():
                raise ValueError(
                    "A training job is already running. Stop it before starting another."
                )
            id = uuid.uuid4().hex
            self.stores[id] = RunStore(project)
            self.stores[id].set_metadata(
                "project",
                {
                    "path": str(project.resolve()),
                    "graph_id": graph.id,
                    "model": graph.metadata.model_dump(),
                },
            )
            self.jobs[id] = {
                "id": id,
                "state": "CREATED",
                "config": config.model_dump(),
                "graph_id": graph.id,
                "graph_digest": graph_digest(graph),
                "metrics": [],
                "logs": [],
                "checkpoint": None,
                "error": None,
                "environment": environment(),
                "created_at": time.time(),
            }
            self.cancels[id] = threading.Event()
            self._update(id)
            thread = threading.Thread(
                target=self._run,
                args=(
                    id,
                    graph.model_copy(deep=True),
                    config,
                    project.resolve(),
                    resume,
                ),
                daemon=True,
            )
            self.threads[id] = thread
            thread.start()
            return self.snapshot(id)

    def snapshot(self, id: str) -> dict:
        with self.lock:
            if id not in self.jobs:
                raise ValueError(f"Unknown live job: {id}; load project history for previous runs")
            return json.loads(json.dumps(self.jobs[id]))

    def stop(self, id: str) -> dict:
        with self.lock:
            state = self.snapshot(id)["state"]
            if state not in ("COMPLETED", "FAILED", "CANCELLED"):
                self.cancels[id].set()
                self._update(id, state="STOPPING")
            return self.snapshot(id)

    def shutdown(self) -> None:
        for id in list(self.jobs):
            self.stop(id)
        for thread in list(self.threads.values()):
            thread.join(timeout=8)

    def _log(self, id: str, message: str) -> None:
        with self.lock:
            self.jobs[id]["logs"].append({"time": time.time(), "message": message})
            self.jobs[id]["logs"] = self.jobs[id]["logs"][-500:]
        self._update(id, kind="log")

    def _run(
        self,
        id: str,
        graph: Graph,
        config: TrainingConfig,
        project: Path,
        resume: str | None,
    ):
        try:
            self._update(id, state="VALIDATING")
            if self.cancels[id].is_set():
                self._update(id, state="CANCELLED")
                return
            if config.device == "cuda" and not torch.cuda.is_available():
                raise ValueError("CUDA is unavailable. Select CPU in Training settings.")
            torch.manual_seed(config.seed)
            model = GraphModule(graph).to(config.device)
            x, y = synthetic_data(graph, config)
            optimizer_type = {"SGD": torch.optim.SGD, "AdamW": torch.optim.AdamW}[config.optimizer]
            optimizer = optimizer_type(model.parameters(), lr=config.learning_rate)
            criterion = {
                "MSELoss": nn.MSELoss,
                "CrossEntropyLoss": nn.CrossEntropyLoss,
            }[config.loss]()
            epoch, next_batch, step = 0, 0, 0
            metrics = []
            elapsed_base = 0.0
            if resume:
                checkpoint_path = Path(resume).resolve()
                if not checkpoint_path.is_relative_to(project / "checkpoints"):
                    raise ValueError(
                        "Checkpoint must be inside this project's checkpoints directory"
                    )
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                if checkpoint.get("format_version") != 1 or checkpoint[
                    "graph_digest"
                ] != graph_digest(graph):
                    raise ValueError(
                        "Checkpoint format or graph differs from the current model. Restore its saved graph first."
                    )
                previous = checkpoint["config"]
                for key, value in config.model_dump().items():
                    if key not in ("epochs", "device") and previous[key] != value:
                        raise ValueError(
                            f"Resume configuration mismatch: {key}. Only total epochs and device may change."
                        )
                model.load_state_dict(checkpoint["model"])
                optimizer.load_state_dict(checkpoint["optimizer"])
                epoch, next_batch, step = (
                    checkpoint["epoch"],
                    checkpoint["next_batch"],
                    checkpoint["step"],
                )
                if config.epochs <= epoch:
                    raise ValueError(
                        f"Checkpoint already finished {epoch} epochs. Increase total epochs to continue."
                    )
                metrics = checkpoint["metrics"]
                elapsed_base = checkpoint.get("elapsed_seconds", 0.0)
                torch.set_rng_state(checkpoint["rng"])
                if config.device == "cuda" and checkpoint.get("cuda_rng"):
                    torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
                self._log(
                    id,
                    f"Restored checkpoint at epoch {epoch + 1}, batch {next_batch}, step {step}",
                )
            self._update(id, state="QUEUED", metrics=metrics)
            artifact_dir = project / "checkpoints"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            artifact = artifact_dir / f"{id}.pt"
            (artifact_dir / f"{id}.graph.json").write_text(
                graph.model_dump_json(indent=2), encoding="utf-8"
            )
            started = time.monotonic()
            self._update(id, state="RUNNING")
            self._log(
                id,
                f"Training {sum(p.numel() for p in model.parameters()):,} parameters on {config.device}; offline synthetic {config.loss}",
            )

            def save_checkpoint():
                payload = {
                    "format_version": 1,
                    "graph_digest": graph_digest(graph),
                    "graph": graph.model_dump(),
                    "config": config.model_dump(),
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "next_batch": next_batch,
                    "step": step,
                    "metrics": metrics,
                    "rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if config.device == "cuda" else [],
                    "elapsed_seconds": elapsed_base + time.monotonic() - started,
                    "environment": environment(),
                }
                temporary = artifact.with_suffix(".tmp")
                torch.save(payload, temporary)
                os.replace(temporary, artifact)
                self._update(id, checkpoint=str(artifact))

            model.train()
            batches = (config.samples + config.batch_size - 1) // config.batch_size
            while epoch < config.epochs:
                order = torch.randperm(
                    config.samples,
                    generator=torch.Generator().manual_seed(config.seed + epoch),
                )
                while next_batch < batches:
                    if self.cancels[id].is_set():
                        save_checkpoint()
                        self._log(
                            id,
                            "Stopped at a batch boundary; model, optimizer and RNG state saved.",
                        )
                        self._update(id, state="CANCELLED")
                        return
                    indices = order[
                        next_batch * config.batch_size : (next_batch + 1) * config.batch_size
                    ]
                    xb, yb = x[indices].to(config.device), y[indices].to(config.device)
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model(xb)
                    if (
                        prediction.ndim != 2
                        or prediction.shape[0] != len(indices)
                        or (config.loss == "MSELoss" and prediction.shape != yb.shape)
                    ):
                        raise ValueError("Training output does not match the synthetic dataset")
                    loss = criterion(prediction, yb)
                    if not torch.isfinite(loss):
                        raise ValueError(
                            "Non-finite loss; check learning rate, data and graph parameters"
                        )
                    loss.backward()
                    optimizer.step()
                    step += 1
                    next_batch += 1
                    accuracy = (
                        float((prediction.argmax(1) == yb).float().mean().item())
                        if config.loss == "CrossEntropyLoss"
                        else None
                    )
                    metric = {
                        "step": step,
                        "epoch": epoch + 1,
                        "batch": next_batch,
                        "loss": float(loss.item()),
                        "accuracy": accuracy,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "elapsed_seconds": elapsed_base + time.monotonic() - started,
                    }
                    metrics.append(metric)
                    metrics = metrics[-10000:]
                    self._update(id, kind="metric", metrics=metrics)
                self._log(
                    id,
                    f"Epoch {epoch + 1}/{config.epochs} · loss {metrics[-1]['loss']:.6f} · step {step}",
                )
                epoch += 1
                next_batch = 0
                save_checkpoint()
            self._update(id, state="COMPLETED")
            self._log(id, "Training completed. Checkpoint is ready to resume or inspect.")
        except Exception as exc:
            # Failure reporting must survive an unwritable experiment database.
            with self.lock:
                self.jobs[id].update(state="FAILED", error=str(exc))
                self.jobs[id]["logs"].append(
                    {"time": time.time(), "message": f"{type(exc).__name__}: {exc}"}
                )
            snapshot = self.snapshot(id)
            try:
                self.stores[id].save(snapshot)
            except Exception as storage_error:
                snapshot["error"] += f" (could not persist failure: {storage_error})"
            self.emit({"event": "state", "job_id": id, "data": snapshot})
