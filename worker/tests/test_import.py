import hashlib
import json
import time
import uuid
from pathlib import Path

import onnx
import pytest
from opweaver.importers import OnnxImporter, builtin_importers
from opweaver.model_artifact import GraphImportResult
from opweaver.model_tasks import ModelTasks
from opweaver.onnx_examples import generate_examples, residual
from opweaver.onnx_import import import_onnx
from opweaver.storage import RunStore


@pytest.fixture
def fixtures(tmp_path):
    source = tmp_path / "source"
    generate_examples(source)
    return source


def read_model(tmp_path, source):
    project = tmp_path / "project"
    folder = project / ".opweaver" / "imports" / uuid.uuid4().hex
    return import_onnx(source, project, folder)


def test_cnn_structure_tensors_weights_and_original_bytes(tmp_path, fixtures):
    source = fixtures / "tiny-cnn.onnx"
    before = source.read_bytes()
    result = read_model(tmp_path, source)
    assert source.read_bytes() == before
    assert result["artifact"]["sha256"] == hashlib.sha256(before).hexdigest()
    assert result["analysis"]["origin"] == "onnx" and result["analysis"]["read_only"]
    assert result["statistics"]["nodes"] == 5
    assert result["statistics"]["graph_nodes"] == 11
    assert result["statistics"]["edges"] == 10
    assert result["statistics"]["initializer_elements"] == 113
    assert result["statistics"]["logical_tensor_bytes"] == 452
    assert result["statistics"]["weight_dtypes"] == {"float32": 113}
    assert result["outputs"][0]["shape"] == ["B", 3]
    pool = next(n for n in result["analysis"]["nodes"] if n["details"].get("op_type") == "MaxPool")
    assert pool["tensors"]["shape"] == ["B", 2, 3, 3]
    assert pool["parameters"]["strides"]["values"] == [2, 2]
    assert result["validation"] == {
        "checker": "passed",
        "shape_inference": "inferred",
        "runtime": "not_executed",
    }
    copied = tmp_path / "project" / result["artifact"]["file_path"]
    assert copied.read_bytes() == before


def test_residual_merge_has_both_real_predecessors(tmp_path, fixtures):
    result = read_model(tmp_path, fixtures / "residual.onnx")
    add = next(n for n in result["analysis"]["nodes"] if n["details"].get("op_type") == "Add")
    incoming = [e for e in result["analysis"]["edges"] if e["target"] == add["id"]]
    assert len(incoming) == 2 and {e["source"] for e in incoming} == {"input_0", "onnx_0"}


def test_dynamic_shape_preserves_symbol_and_unknown_dimension(tmp_path, fixtures):
    result = read_model(tmp_path, fixtures / "dynamic.onnx")
    assert result["inputs"][0]["shape"] == ["B", None, 4]
    assert result["outputs"][0]["shape"][0] == "B"
    assert not isinstance(result["outputs"][0]["shape"][1], int)
    assert any(d["code"] == "DYNAMIC_SHAPE" for d in result["diagnostics"])


def test_custom_operator_retains_domain_attributes_and_edges(tmp_path, fixtures):
    result = read_model(tmp_path, fixtures / "custom.onnx")
    custom = next(n for n in result["analysis"]["nodes"] if n["kind"] == "onnx_node")
    assert custom["operator"] == "com.example::Fancy"
    assert custom["parameters"]["alpha"] == 0.25
    assert custom["details"]["known"] is False
    assert len(result["analysis"]["edges"]) == 2
    assert any(
        d["code"] == "UNKNOWN_OPERATOR" and d["node"] == custom["id"] for d in result["diagnostics"]
    )


def test_unregistered_standard_operator_is_retained(tmp_path):
    model = residual()
    model.graph.node[0].op_type = "FutureIdentity"
    source = tmp_path / "future.onnx"
    onnx.save_model(model, source)
    result = read_model(tmp_path, source)
    assert result["validation"]["checker"] == "incomplete"
    assert result["statistics"]["unknown_operators"] == {"ai.onnx::FutureIdentity": 1}


def test_multi_output_and_optional_input_ports_are_not_miswired(tmp_path, fixtures):
    result = read_model(tmp_path, fixtures / "multi-output.onnx")
    assert len(result["outputs"]) == 2
    edges = result["analysis"]["edges"]
    assert any(
        e["source"] == "onnx_0" and e["source_port"] == "out_1" and e["target_port"] == "in_1"
        for e in edges
    )
    split = next(n for n in result["analysis"]["nodes"] if n["id"] == "onnx_0")
    assert split["inputs"] == ["in_0"] and split["outputs"] == ["out_0", "out_1"]


def test_if_subgraphs_preserve_control_flow_and_outer_captures(tmp_path, fixtures):
    result = read_model(tmp_path, fixtures / "conditional.onnx")
    assert result["statistics"]["nodes"] == 1
    assert len(result["subgraphs"]) == 2
    assert {g["name"] for g in result["subgraphs"].values()} == {"Then branch", "Else branch"}
    conditional = next(n for n in result["analysis"]["nodes"] if n["kind"] == "onnx_node")
    assert len(conditional["details"]["subgraphs"]) == 2
    assert any(
        v["implicit_capture"] and v["name"] == "x"
        for v in conditional["details"]["input_tensors"].values()
    )
    for graph in result["subgraphs"].values():
        assert any(n["kind"] == "captured_input" for n in graph["nodes"])
        assert len(graph["edges"]) >= 2


def test_external_weights_are_copied_without_embedding_data(tmp_path, fixtures):
    original = (fixtures / "weights.bin").read_bytes()
    result = read_model(tmp_path, fixtures / "external-cnn.onnx")
    artifact = result["artifact"]
    assert artifact["external_files"][0]["sha256"] == hashlib.sha256(original).hexdigest()
    copy = tmp_path / "project" / artifact["file_path"]
    assert (copy.parent / "weights.bin").read_bytes() == original
    assert artifact["total_file_bytes"] == copy.stat().st_size + len(original)
    assert all(
        not t.raw_data for t in onnx.load_model(copy, load_external_data=False).graph.initializer
    )
    assert "raw_data" not in json.dumps(result)
    assert result["statistics"]["initializer_elements"] == 113


@pytest.mark.parametrize("case", ["corrupt", "missing", "traversal", "invalid", "range"])
def test_invalid_models_and_external_paths_fail_cleanly(tmp_path, fixtures, case):
    path = fixtures / "corrupt.onnx"
    if case in ("missing", "traversal", "range"):
        path = fixtures / "external-cnn.onnx"
        if case == "missing":
            (fixtures / "weights.bin").unlink()
        else:
            model = onnx.load_model(path, load_external_data=False)
            target = next(
                v
                for v in model.graph.initializer[0].external_data
                if v.key == ("location" if case == "traversal" else "offset")
            )
            target.value = "../outside.bin" if case == "traversal" else "999999"
            onnx.save_model(model, path)
    elif case == "invalid":
        path = fixtures / "bad.onnx"
        model = residual()
        model.graph.node[0].input[0] = "missing_tensor"
        onnx.save_model(model, path)
    with pytest.raises(Exception):
        read_model(tmp_path, path)


def wait_task(tasks, id):
    deadline = time.monotonic() + 35
    while time.monotonic() < deadline:
        current = tasks.snapshot(id)
        if current["state"] in ("COMPLETED", "FAILED", "CANCELLED"):
            return current
        time.sleep(0.03)
    raise AssertionError("Import did not terminate")


def test_async_import_history_reopen_and_failure_recovery(tmp_path, fixtures):
    project = tmp_path / "workspace"
    project.mkdir()
    (project / "project.opweaver.json").write_text("{}")
    tasks = ModelTasks()
    first = tasks.start({"project": str(project), "source": str(fixtures / "external-cnn.onnx")})
    result = wait_task(tasks, first["id"])
    assert result["state"] == "COMPLETED", result.get("error")
    failed = tasks.start({"project": str(project), "source": str(fixtures / "corrupt.onnx")})
    assert wait_task(tasks, failed["id"])["state"] == "FAILED"
    tasks.shutdown()
    reopened = ModelTasks()
    assert len(reopened.history(project)) == 2
    loaded = reopened.load(project, first["id"])
    assert loaded["artifact"]["id"] == first["id"]
    metadata = RunStore(project).metadata_list("model-import:")
    assert all("result" not in row for row in metadata)
    with pytest.raises(ValueError):
        reopened.load(project, "../escape")


def test_importer_registry_and_pytorch_reuses_capture(tmp_path):
    registry = builtin_importers()
    assert registry.resolve("model.py").requires_execution
    with pytest.raises(ValueError, match="already registered"):
        registry.register(OnnxImporter())
    with pytest.raises(ValueError, match="Automatic deserialization is disabled"):
        registry.resolve("weights.pth")
    project = tmp_path / "project"
    source = tmp_path / "source.py"
    source.write_text(
        "from torch import nn\nclass Model(nn.Module):\n def __init__(self,width=3):\n  super().__init__()\n  self.head=nn.Linear(8,width)\n def forward(self,x):\n  return self.head(x)\n"
    )
    probe = registry.probe(source)
    assert probe["entries"][0]["name"] == "Model"
    folder = project / ".opweaver" / "imports" / uuid.uuid4().hex
    folder.mkdir(parents=True)
    result = registry.resolve(source).import_model(
        {
            "project": str(project),
            "file_path": str(source),
            "entry": "Model",
            "constructor": {"width": 5},
            "inputs": [{"shape": [4, 8]}],
            "execute_confirmed": True,
        },
        folder,
    )
    assert result["artifact"]["format"] == "pytorch"
    assert result["analysis"]["origin"] == "pytorch.fx"
    assert result["pytorch"]["forward"]["outputs"][0]["shape"] == [4, 5]
    assert result["statistics"]["learnable_parameters"] == 45
    assert result["statistics"]["nodes"] == 1
    assert result["pytorch"]["capture"]["constructor"] == {"width": 5}


def test_model_report_schema_is_current():
    root = Path(__file__).resolve().parents[2]
    assert (
        json.loads((root / "schemas" / "onnx-report.schema.json").read_text())
        == GraphImportResult.model_json_schema()
    )


def test_large_graph_contains_real_nodes_and_no_weight_payloads(tmp_path, fixtures):
    result = read_model(tmp_path, fixtures / "large-chain.onnx")
    assert result["statistics"]["nodes"] == 600
    assert len(result["analysis"]["nodes"]) == 602
    assert result["analysis"]["source_model_id"] == result["artifact"]["id"]
