"""The Python IR is the source; regenerate JSON Schema and example graphs."""

import json
from pathlib import Path

from opweaver.examples import mlp, residual
from opweaver.ir import Graph
from opweaver.model_artifact import OnnxReport
from opweaver.source import AnalysisGraph, SourceMapping
from opweaver.training import TrainingConfig

root = Path(__file__).resolve().parents[1]
(root / "schemas").mkdir(exist_ok=True)
(root / "schemas" / "graph-ir.schema.json").write_text(
    json.dumps(Graph.model_json_schema(), indent=2) + "\n", encoding="utf-8"
)
(root / "schemas" / "training-config.schema.json").write_text(
    json.dumps(TrainingConfig.model_json_schema(), indent=2) + "\n", encoding="utf-8"
)
for name, model in [
    ("analysis-graph", AnalysisGraph),
    ("source-mapping", SourceMapping),
    ("onnx-report", OnnxReport),
]:
    (root / "schemas" / f"{name}.schema.json").write_text(
        json.dumps(model.model_json_schema(), indent=2) + "\n", encoding="utf-8"
    )
for name, graph in [("synthetic-mlp", mlp()), ("residual-mlp", residual())]:
    folder = root / "examples" / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "graph.json").write_text(graph.model_dump_json(indent=2) + "\n", encoding="utf-8")
    positions = {
        node.id: {"x": 60 + index * 235, "y": 150 if node.id != "sum" else 220}
        for index, node in enumerate(graph.nodes)
    }
    (folder / "layout.json").write_text(
        json.dumps({"version": 1, "positions": positions}, indent=2) + "\n",
        encoding="utf-8",
    )
    (folder / "project.opweaver.json").write_text(
        json.dumps(
            {
                "project_version": 1,
                "name": graph.metadata.name,
                "graph_file": "graph.json",
                "layout_file": "layout.json",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
