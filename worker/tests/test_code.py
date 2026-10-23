from pathlib import Path

import pytest
import torch
from opweaver.builder import GraphModule
from opweaver.capture import capture, load_model
from opweaver.code_tasks import CaptureTasks
from opweaver.codegen import generate
from opweaver.examples import mlp, node, residual
from opweaver.registry import builtin_registry
from opweaver.source import digest, inspect_source, patch_parameter

SOURCE = """# 用户的注释必须保留
import torch
from torch import nn

def unrelated_helper():
    return "keep me"

class ResidualModel(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.fc = nn.Linear(8, width)  # preserve comment
        self.norm = nn.LayerNorm(width)
        self.activation = nn.GELU()
        self.head = nn.Linear(width, 3)

    def forward(self, x):
        return self.head(self.activation(self.norm(self.fc(x))) + x)
"""


def write_source(tmp_path, text=SOURCE):
    folder = tmp_path / "src"
    folder.mkdir(parents=True, exist_ok=True)
    file = folder / "model.py"
    file.write_bytes(text.encode("utf-8"))
    return file


def test_a_capture_layers_residual_and_source_mapping(tmp_path):
    file = write_source(tmp_path)
    result = capture(str(file), "ResidualModel", {}, [{"shape": [4, 8]}])
    assert result["success"] and result["design_graph"]
    labels = [n["label"] for n in result["analysis"]["nodes"]]
    assert all(op in labels for op in ("Linear", "LayerNorm", "GELU", "Add"))
    assert len(result["analysis"]["edges"]) == 7
    assert result["forward"]["outputs"][0]["shape"] == [4, 3]
    mapping = next(m for m in result["source_map"] if m["module_path"] == "fc")
    assert mapping["status"] == "exact" and mapping["start_line"] == 11


def test_b_saved_activation_change_updates_same_node(tmp_path):
    file = write_source(tmp_path, SOURCE.replace("nn.GELU()", "nn.ReLU()"))
    before = capture(str(file), "ResidualModel", {}, [{"shape": [2, 8]}])
    file.write_text(SOURCE, encoding="utf-8")
    after = capture(str(file), "ResidualModel", {}, [{"shape": [2, 8]}])
    a = next(n for n in before["analysis"]["nodes"] if n["module_path"] == "activation")
    b = next(n for n in after["analysis"]["nodes"] if n["module_path"] == "activation")
    assert a["id"] == b["id"] and a["operator"] == "core.relu" and b["operator"] == "core.gelu"


@pytest.mark.parametrize("factory", [mlp, residual])
def test_c_graph_generated_code_forward_and_round_trip(tmp_path, factory):
    graph = factory()
    graph.nodes[-2].params["out_features"] = 5
    generated = generate(graph)
    assert "out_features=5" in generated["text"]
    file = write_source(tmp_path, generated["text"])
    captured = capture(
        str(file), generated["class_name"], {}, [{"shape": [4, 8], "design_shape": ["B", 8]}]
    )
    assert captured["design_graph"], captured["diagnostics"]
    from opweaver.ir import Graph

    recovered = Graph.model_validate(captured["design_graph"])
    assert {n.id: (n.operator, n.params) for n in graph.nodes} == {
        n.id: (n.operator, n.params) for n in recovered.nodes
    }
    a = {(e.source, e.source_port, e.target, e.target_port) for e in graph.edges}
    b = {(e.source, e.source_port, e.target, e.target_port) for e in recovered.edges}
    assert a == b
    direct = load_model(file, generated["class_name"], {})
    eager = GraphModule(graph)
    for id, module in eager.layers.items():
        if not module.state_dict():
            continue
        owner = next(n for n in graph.nodes if n.id == id).share_with or id
        module.load_state_dict(getattr(direct, owner).state_dict())
    x = torch.randn(4, 8)
    torch.testing.assert_close(direct(x), eager(x))


def test_static_open_does_not_execute_python(tmp_path):
    marker = tmp_path / "executed.txt"
    text = f"from pathlib import Path\nPath({str(marker)!r}).write_text('bad')\n" + SOURCE
    result = inspect_source(text)
    assert not marker.exists()
    assert any(e["name"] == "ResidualModel" for e in result["entries"])
    assert inspect_source("class Bad(:")["diagnostics"][0]["line"] == 1


def test_local_patch_preserves_unrelated_source_and_rejects_conflict():
    result = patch_parameter(SOURCE, "ResidualModel", "head", "out_features", 5, digest(SOURCE))
    assert 'return "keep me"' in result["text"]
    assert "# 用户的注释必须保留" in result["text"]
    assert "# preserve comment" in result["text"]
    assert "self.head = nn.Linear(width, 5)" in result["text"]
    assert result["diff"].count("\n-") == 1
    with pytest.raises(ValueError, match="conflict"):
        patch_parameter(SOURCE + "\n", "ResidualModel", "head", "out_features", 5, digest(SOURCE))
    with pytest.raises(ValueError, match="computed"):
        patch_parameter(SOURCE, "ResidualModel", "head", "in_features", 16, digest(SOURCE))


def test_conv_batchnorm_sequential_and_concat(tmp_path):
    text = """import torch
from torch import nn
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4), nn.ReLU())
    def forward(self, x):
        y = self.features(x)
        return torch.cat((y, y), dim=1)
"""
    result = capture(str(write_source(tmp_path, text)), "CNN", {}, [{"shape": [2, 3, 8, 8]}])
    assert result["design_graph"], result["diagnostics"]
    assert result["forward"]["outputs"][0]["shape"] == [2, 8, 6, 6]
    assert any(
        m["status"] == "inferred" and m["module_path"] == "features.0" for m in result["source_map"]
    )


def test_unknown_operator_stays_visible_and_analysis_only(tmp_path):
    text = SOURCE.replace(
        "return self.head(self.activation(self.norm(self.fc(x))) + x)",
        "return torch.sin(self.fc(x))",
    )
    result = capture(str(write_source(tmp_path, text)), "ResidualModel", {}, [{"shape": [2, 8]}])
    assert result["success"] and result["design_graph"] is None
    assert any("sin" in n["operator"] for n in result["analysis"]["nodes"])
    assert result["diagnostics"]


def task(tasks, root, text, entry="ResidualModel", timeout=20):
    file = write_source(root, text)
    job = tasks.start(
        {
            "project": str(root),
            "file_path": str(file),
            "entry": entry,
            "inputs": [{"shape": [2, 8]}],
            "expected_hash": digest(text),
            "execute_confirmed": True,
            "timeout": timeout,
        }
    )
    tasks.threads[job["id"]].join(timeout + 10)
    return tasks.snapshot(job["id"])


def test_d_dynamic_control_flow_failure_does_not_overwrite_source(tmp_path):
    text = SOURCE.replace(
        "return self.head(self.activation(self.norm(self.fc(x))) + x)",
        "if x.sum() > 0:\n            return self.fc(x)\n        return x",
    )
    result = task(CaptureTasks(), tmp_path, text)
    assert result["state"] == "FAILED"
    assert "control flow" in result["error"]
    assert (tmp_path / "src/model.py").read_text(encoding="utf-8") == text
    assert Path(result["artifact_dir"], "result.json").exists()


def test_capture_timeout_and_next_task_recovers(tmp_path):
    tasks = CaptureTasks()
    timeout = task(tasks, tmp_path, "import time\ntime.sleep(30)\n" + SOURCE, timeout=3)
    assert timeout["state"] == "FAILED" and "timed out" in timeout["error"]
    valid = task(tasks, tmp_path, SOURCE)
    assert valid["state"] == "COMPLETED"


def test_capture_cancellation(tmp_path):
    tasks = CaptureTasks()
    text = "import time\ntime.sleep(30)\n" + SOURCE
    file = write_source(tmp_path, text)
    job = tasks.start(
        {
            "project": str(tmp_path),
            "file_path": str(file),
            "entry": "ResidualModel",
            "inputs": [{"shape": [2, 8]}],
            "expected_hash": digest(text),
            "execute_confirmed": True,
        }
    )
    tasks.stop(job["id"])
    tasks.threads[job["id"]].join(10)
    assert tasks.snapshot(job["id"])["state"] == "CANCELLED"


def test_custom_operator_codegen_example(tmp_path):
    from opweaver.custom_example import register_example

    registry = builtin_registry()
    register_example(registry)
    graph = mlp()
    graph.nodes[2] = node("relu", "relu")
    graph.nodes[2].operator.id = "example.square"
    model = GraphModule(graph, registry)
    assert model(torch.randn(2, 8)).shape == (2, 3)
    generated = generate(graph, registry)
    assert "Square()" in generated["text"]
    file = write_source(tmp_path, generated["text"])
    result = capture(str(file), "OpWeaverModel", {}, [{"shape": [2, 8]}], registry)
    assert result["design_graph"]


def test_windows_crlf_bom_capture_and_exact_source_hash(tmp_path):
    source = "\ufeff" + SOURCE.replace("\n", "\r\n")
    file = write_source(tmp_path, "")
    file.write_bytes(source.encode("utf-8"))
    tasks = CaptureTasks()
    job = tasks.start(
        {
            "project": str(tmp_path),
            "file_path": str(file),
            "entry": "ResidualModel",
            "inputs": [{"shape": [2, 8]}],
            "expected_hash": digest(source),
            "execute_confirmed": True,
        }
    )
    tasks.threads[job["id"]].join(30)
    result = tasks.snapshot(job["id"])
    assert result["state"] == "COMPLETED", result["error"]
    assert result["result"]["source_hash"] == digest(source)
    patched = patch_parameter(source, "ResidualModel", "head", "out_features", 5, digest(source))
    assert patched["text"].startswith("\ufeff")
    assert patched["text"].count("\r\n") == source.count("\r\n")


def test_ambiguous_constructor_assignment_cannot_be_patched():
    source = SOURCE.replace(
        "self.head = nn.Linear(width, 3)",
        "self.head = nn.Linear(width, 3)\n        self.head = nn.Linear(width, 4)",
    )
    with pytest.raises(ValueError, match="no unique"):
        patch_parameter(source, "ResidualModel", "head", "out_features", 5, digest(source))


def test_capture_uses_constructor_kwargs_and_checks_reference_forward(tmp_path):
    source = """from torch import nn
class Configurable(nn.Module):
    def __init__(self, features=8, classes=3):
        super().__init__()
        self.head = nn.Linear(features, classes)
    def forward(self, x):
        return self.head(x)
"""
    result = capture(
        str(write_source(tmp_path, source)),
        "Configurable",
        {"features": 4, "classes": 2},
        [{"shape": [2, 4]}],
    )
    assert result["forward"]["outputs"][0]["shape"] == [2, 2]
    assert result["validation"]["passed"]
    assert "min" in result["forward"]["outputs"][0]


def test_source_patch_validates_operator_parameter_constraints():
    from opweaver.worker import Worker

    worker = Worker(lambda event: None)
    with pytest.raises(ValueError, match="out_features"):
        worker.dispatch(
            "code.patch",
            {
                "source": SOURCE,
                "class_name": "ResidualModel",
                "module_path": "head",
                "key": "out_features",
                "value": -5,
                "expected_hash": digest(SOURCE),
                "operator_id": "core.linear",
                "parameters": {"in_features": 8, "out_features": 3, "bias": True},
            },
        )
