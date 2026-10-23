import pytest
import torch
from opweaver.examples import mlp
from opweaver.storage import RunStore
from opweaver.training import TrainingConfig, TrainingEngine
from opweaver.worker import Worker


def run(engine, tmp_path, config, resume=None):
    job = engine.start(mlp(), config, tmp_path, resume)
    engine.threads[job["id"]].join(timeout=60)
    result = engine.snapshot(job["id"])
    assert result["state"] == "COMPLETED", result
    return result


def test_training_loss_decreases_and_checkpoint_restores(tmp_path):
    torch.set_num_threads(2)
    config = TrainingConfig(epochs=12, samples=256)
    engine = TrainingEngine()
    trained = run(engine, tmp_path, config)
    losses = [m["loss"] for m in trained["metrics"]]
    assert sum(losses[-8:]) < sum(losses[:8]) * 0.6
    checkpoint = torch.load(trained["checkpoint"], weights_only=True)
    assert checkpoint["epoch"] == 12
    resumed = run(
        TrainingEngine(),
        tmp_path,
        config.model_copy(update={"epochs": 14}),
        trained["checkpoint"],
    )
    assert resumed["metrics"][0] == trained["metrics"][0]
    assert resumed["metrics"][-1]["step"] == 14 * 8
    assert len(RunStore(tmp_path).history()) == 2


def test_resume_matches_uninterrupted(tmp_path):
    config = TrainingConfig(epochs=4, samples=64)
    full = run(TrainingEngine(), tmp_path / "full", config)
    half = run(TrainingEngine(), tmp_path / "split", config.model_copy(update={"epochs": 2}))
    resumed = run(TrainingEngine(), tmp_path / "split", config, half["checkpoint"])
    state_a = torch.load(full["checkpoint"], weights_only=True)["model"]
    state_b = torch.load(resumed["checkpoint"], weights_only=True)["model"]
    assert all(torch.equal(state_a[key], state_b[key]) for key in state_a)


def test_stop_and_resume_mid_epoch(tmp_path):
    engine = TrainingEngine()

    def on_event(event):
        if event["event"] == "metric" and event["data"]["metrics"][-1]["step"] == 3:
            engine.stop(event["job_id"])

    engine.emit = on_event
    config = TrainingConfig(epochs=3, samples=256)
    job = engine.start(mlp(), config, tmp_path)
    engine.threads[job["id"]].join(timeout=60)
    stopped = engine.snapshot(job["id"])
    assert stopped["state"] == "CANCELLED"
    assert stopped["checkpoint"]
    resumed = run(TrainingEngine(), tmp_path, config, stopped["checkpoint"])
    assert resumed["metrics"][-1]["step"] == 24


@pytest.mark.parametrize("loss", ["MSELoss", "CrossEntropyLoss"])
@pytest.mark.parametrize("optimizer", ["SGD", "AdamW"])
def test_loss_optimizer_combinations(tmp_path, loss, optimizer):
    result = run(
        TrainingEngine(),
        tmp_path,
        TrainingConfig(epochs=1, samples=32, loss=loss, optimizer=optimizer),
    )
    assert result["metrics"]


def test_failure_is_contained(tmp_path):
    engine = TrainingEngine()
    graph = mlp()
    graph.nodes[1].params["in_features"] = 200
    job = engine.start(graph, TrainingConfig(), tmp_path)
    engine.threads[job["id"]].join(timeout=30)
    assert engine.snapshot(job["id"])["state"] == "FAILED"
    assert run(engine, tmp_path, TrainingConfig(epochs=1, samples=32))["state"] == "COMPLETED"


def test_crash_recovery_marks_interrupted_run(tmp_path):
    store = RunStore(tmp_path)
    store.save({"id": "crashed", "state": "RUNNING"})
    assert store.recover(set())[0]["state"] == "FAILED"


def test_worker_error_then_valid_call():
    worker = Worker(lambda event: None)
    with pytest.raises(ValueError, match="Unknown method"):
        worker.dispatch("not-a-method", {})
    assert worker.dispatch("forward", {"graph": mlp().model_dump()})["outputs"][0]["shape"] == [
        4,
        3,
    ]


def test_cpu_without_cuda_and_clear_gpu_error(tmp_path, monkeypatch):
    from opweaver.training import environment

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert not environment()["cuda_available"]
    assert "CPU" in environment()["notice"]
    engine = TrainingEngine()
    run(engine, tmp_path, TrainingConfig(epochs=1, samples=32))
    job = engine.start(mlp(), TrainingConfig(device="cuda"), tmp_path)
    engine.threads[job["id"]].join(timeout=30)
    failure = engine.snapshot(job["id"])
    assert failure["state"] == "FAILED" and "CUDA is unavailable" in failure["error"]


def test_resume_rejects_graph_and_configuration_changes(tmp_path):
    engine = TrainingEngine()
    completed = run(engine, tmp_path, TrainingConfig(epochs=1, samples=32))
    job = engine.start(
        mlp(),
        TrainingConfig(epochs=2, samples=32, learning_rate=0.02),
        tmp_path,
        completed["checkpoint"],
    )
    engine.threads[job["id"]].join(timeout=30)
    assert "configuration mismatch" in engine.snapshot(job["id"])["error"]
    graph = mlp()
    graph.nodes[2].operator.id = "core.gelu"
    job = engine.start(
        graph, TrainingConfig(epochs=2, samples=32), tmp_path, completed["checkpoint"]
    )
    engine.threads[job["id"]].join(timeout=30)
    assert "graph differs" in engine.snapshot(job["id"])["error"]


def test_mid_epoch_dropout_resume_is_exact(tmp_path):
    from opweaver.examples import node
    from opweaver.ir import Edge

    graph = mlp()
    graph.nodes.append(node("drop", "dropout", p=0.3))
    graph.edges[2].source = "drop"
    graph.edges.append(Edge(id="drop_edge", source="relu", target="drop"))
    config = TrainingConfig(epochs=3, samples=96)

    def execute(engine, folder, resume=None):
        job = engine.start(graph, config, folder, resume)
        engine.threads[job["id"]].join(60)
        return engine.snapshot(job["id"])

    whole = execute(TrainingEngine(), tmp_path / "whole")
    interrupted = TrainingEngine()
    interrupted.emit = lambda event: (
        interrupted.stop(event["job_id"])
        if event["event"] == "metric" and event["data"]["metrics"][-1]["step"] == 2
        else None
    )
    partial = execute(interrupted, tmp_path / "partial")
    assert partial["state"] == "CANCELLED"
    resumed = execute(TrainingEngine(), tmp_path / "partial", partial["checkpoint"])
    assert whole["state"] == resumed["state"] == "COMPLETED"
    a, b = [
        torch.load(result["checkpoint"], weights_only=True)["model"] for result in (whole, resumed)
    ]
    assert all(torch.equal(a[key], b[key]) for key in a)


def test_storage_failure_still_reports_terminal_state(tmp_path, monkeypatch):
    original = RunStore.save

    def failing_save(self, record):
        if record["state"] != "CREATED":
            raise OSError("disk is full")
        return original(self, record)

    monkeypatch.setattr(RunStore, "save", failing_save)
    events = []
    engine = TrainingEngine(events.append)
    job = engine.start(mlp(), TrainingConfig(epochs=1), tmp_path)
    engine.threads[job["id"]].join(30)
    assert engine.snapshot(job["id"])["state"] == "FAILED"
    assert events[-1]["data"]["state"] == "FAILED"
    assert "disk is full" in events[-1]["data"]["error"]
