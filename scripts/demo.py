"""Offline CPU acceptance run; does not require the UI or download data."""

import argparse
import json
from pathlib import Path

import torch
from opweaver.builder import GraphModule
from opweaver.examples import mlp, residual
from opweaver.training import TrainingConfig, TrainingEngine

parser = argparse.ArgumentParser()
parser.add_argument("--project", default="projects/headless-demo")
parser.add_argument("--residual", action="store_true")
args = parser.parse_args()
torch.set_num_threads(2)
graph = residual() if args.residual else mlp()
root = Path(args.project)
root.mkdir(parents=True, exist_ok=True)
(root / "graph.json").write_text(graph.model_dump_json(indent=2), encoding="utf-8")
(root / "layout.json").write_text(
    json.dumps(
        {
            "version": 1,
            "positions": {n.id: {"x": 40 + i * 230, "y": 120} for i, n in enumerate(graph.nodes)},
        }
    ),
    encoding="utf-8",
)
(root / "project.opweaver.json").write_text(
    json.dumps(
        {
            "project_version": 1,
            "name": graph.metadata.name,
            "graph_file": "graph.json",
            "layout_file": "layout.json",
        }
    ),
    encoding="utf-8",
)
model = GraphModule(graph)
print("Forward:", tuple(model(torch.randn(4, 8)).shape))
engine = TrainingEngine(
    lambda event: print(event["data"]["logs"][-1]["message"]) if event["event"] == "log" else None
)
job = engine.start(graph, TrainingConfig(epochs=12), root)
engine.threads[job["id"]].join()
result = engine.snapshot(job["id"])
if result["state"] != "COMPLETED":
    raise SystemExit(result["error"])
print("Checkpoint:", result["checkpoint"])
print(
    "Mean first/last epoch loss:",
    sum(m["loss"] for m in result["metrics"][:16]) / 16,
    sum(m["loss"] for m in result["metrics"][-16:]) / 16,
)
