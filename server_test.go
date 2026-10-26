package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestLocalAdapterRejectsForeignOriginsAndHosts(t *testing.T) {
	handler := localOnly(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(200) }))
	for _, example := range []struct {
		host, origin string
		status       int
	}{
		{"127.0.0.1:34115", "http://127.0.0.1:34115", 200},
		{"localhost:34115", "http://127.0.0.1:5173", 200},
		{"localhost:34115", "https://foreign.example", 403},
		{"foreign.example:34115", "", 403},
	} {
		r := httptest.NewRequest("GET", "http://"+example.host+"/api/events", nil)
		r.Header.Set("Origin", example.origin)
		w := httptest.NewRecorder()
		handler.ServeHTTP(w, r)
		if w.Code != example.status {
			t.Fatalf("%s %s: got %d", example.host, example.origin, w.Code)
		}
	}
}

func TestRealModelImportThroughAppAPI(t *testing.T) {
	if os.Getenv("OPWEAVER_INTEGRATION") != "1" {
		t.Skip("set OPWEAVER_INTEGRATION=1 for real ONNX import")
	}
	root, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	app := NewApp(root)
	defer app.worker.Close()
	call := func(method string, params any) map[string]any {
		t.Helper()
		payload, _ := json.Marshal(params)
		raw, err := app.Request(method, string(payload))
		if err != nil {
			t.Fatalf("%s: %v", method, err)
		}
		var result map[string]any
		if err = json.Unmarshal([]byte(raw), &result); err != nil {
			t.Fatal(err)
		}
		return result
	}
	example := call("example", map[string]any{"name": "synthetic-mlp"})
	project := filepath.Join(t.TempDir(), "project")
	call("project.create", map[string]any{"path": project, "name": "Import integration", "graph": example["graph"], "layout": example["layout"]})
	job := call("model.import.start", map[string]any{"project": project, "source": filepath.Join(root, "examples", "onnx", "external-cnn.onnx")})
	deadline := time.Now().Add(30 * time.Second)
	for job["state"] != "COMPLETED" && time.Now().Before(deadline) {
		if job["state"] == "FAILED" {
			t.Fatal(job["error"])
		}
		time.Sleep(30 * time.Millisecond)
		job = call("model.import.status", map[string]any{"id": job["id"]})
	}
	if job["state"] != "COMPLETED" {
		t.Fatal("import did not complete")
	}
	loaded := call("model.load", map[string]any{"project": project, "id": job["id"]})
	stats := loaded["statistics"].(map[string]any)
	if stats["initializer_elements"] != float64(113) {
		t.Fatal(stats)
	}
	lab := call("lab.start", map[string]any{"project": project, "config": map[string]any{"source_model_id": job["id"], "inputs": map[string]any{"shapes": map[string]any{"image": []int{2, 3, 8, 8}}}, "benchmark": map[string]any{"runs": 8, "warmup": 2}}})
	deadline = time.Now().Add(30 * time.Second)
	for lab["state"] != "COMPLETED" && time.Now().Before(deadline) {
		if lab["state"] == "FAILED" {
			t.Fatal(lab["error"])
		}
		time.Sleep(30 * time.Millisecond)
		lab = call("lab.status", map[string]any{"id": lab["id"]})
	}
	experiment := call("lab.load", map[string]any{"project": project, "id": lab["id"]})
	if experiment["status"] != "Succeeded" || !experiment["validation"].(map[string]any)["passed"].(bool) {
		t.Fatal(experiment)
	}
	artifact := experiment["optimization"].(map[string]any)["artifact"].(map[string]any)
	call("model.load", map[string]any{"project": project, "id": artifact["id"]})
	exported := call("lab.export", map[string]any{"project": project, "id": lab["id"], "directory": t.TempDir()})
	if _, err := os.Stat(exported["json"].(string)); err != nil {
		t.Fatal(err)
	}
	comparison := call("lab.compare", map[string]any{"project": project, "reference_id": lab["id"], "candidate_id": lab["id"]})
	if comparison["comparison"].(map[string]any)["outcome"] != "Unchanged" {
		t.Fatal(comparison)
	}
	interpreter := os.Getenv("OPWEAVER_OLIVE_PYTHON")
	if interpreter == "" {
		interpreter = filepath.Join(root, ".tools", "olive-env", "Scripts", "python.exe")
	}
	if _, err := os.Stat(interpreter); err == nil {
		environment := call("lab.environment.configure", map[string]any{"project": project, "environment": map[string]any{"python_path": interpreter}})
		if environment["status"] != "Available" {
			t.Fatal(environment)
		}
		config := map[string]any{"source_model_id": job["id"], "inputs": map[string]any{"shapes": map[string]any{"image": []int{2, 3, 8, 8}}}, "benchmark": map[string]any{"runs": 8, "warmup": 2}, "optimization": map[string]any{"backend_id": "microsoft.olive", "workflow": map[string]any{"passes": []any{map[string]any{"id": "peephole", "type": "OnnxPeepholeOptimizer"}}}}}
		preview := call("lab.workflow.preview", map[string]any{"project": project, "config": config})
		if preview["native_workflow"] == nil {
			t.Fatal(preview)
		}
		call("lab.workflow.save", map[string]any{"project": project, "config": config})
		oliveJob := call("lab.start", map[string]any{"project": project, "config": config})
		deadline = time.Now().Add(60 * time.Second)
		for oliveJob["state"] != "COMPLETED" && time.Now().Before(deadline) {
			if oliveJob["state"] == "FAILED" {
				t.Fatal(oliveJob["error"])
			}
			time.Sleep(100 * time.Millisecond)
			oliveJob = call("lab.status", map[string]any{"id": oliveJob["id"]})
		}
		oliveResult := call("lab.load", map[string]any{"project": project, "id": oliveJob["id"]})
		if oliveResult["status"] != "Succeeded" || oliveResult["workflow_execution"] == nil {
			t.Fatal(oliveResult)
		}
		if !oliveResult["validation"].(map[string]any)["passed"].(bool) {
			t.Fatal(oliveResult)
		}
		call("lab.export", map[string]any{"project": project, "id": oliveJob["id"], "directory": t.TempDir()})
		calibration := call("lab.dataset.create", map[string]any{"project": project, "config": map[string]any{"source_model_id": job["id"], "name": "Calibration", "mode": "random", "count": 4, "seed": 42, "shapes": map[string]any{"image": []int{2, 3, 8, 8}}}})
		evaluation := call("lab.dataset.create", map[string]any{"project": project, "config": map[string]any{"source_model_id": job["id"], "name": "Evaluation", "role": "evaluation", "mode": "random", "count": 3, "seed": 1042, "shapes": map[string]any{"image": []int{2, 3, 8, 8}}}})
		call("lab.dataset.load", map[string]any{"project": project, "id": calibration["id"]})
		config["optimization"].(map[string]any)["calibration_dataset_id"] = calibration["id"]
		config["optimization"].(map[string]any)["workflow"] = map[string]any{"passes": []any{map[string]any{"id": "quant", "type": "OnnxStaticQuantization"}}}
		config["evaluation"] = map[string]any{"dataset_id": evaluation["id"], "gates": map[string]any{"max_mean_abs_error": 0}}
		quantJob := call("lab.start", map[string]any{"project": project, "config": config})
		deadline = time.Now().Add(60 * time.Second)
		for quantJob["state"] != "COMPLETED" && time.Now().Before(deadline) {
			if quantJob["state"] == "FAILED" {
				t.Fatal(quantJob["error"])
			}
			time.Sleep(100 * time.Millisecond)
			quantJob = call("lab.status", map[string]any{"id": quantJob["id"]})
		}
		quantResult := call("lab.load", map[string]any{"project": project, "id": quantJob["id"]})
		if quantResult["status"] != "Succeeded" || !quantResult["validation"].(map[string]any)["execution_valid"].(bool) || !quantResult["quantization"].(map[string]any)["present"].(bool) {
			t.Fatal(quantResult)
		}
		call("lab.export", map[string]any{"project": project, "id": quantJob["id"], "directory": t.TempDir()})
		diagnosticData := call("lab.dataset.create", map[string]any{"project": project, "config": map[string]any{"source_model_id": job["id"], "name": "Diagnostic", "role": "diagnostic", "mode": "random", "count": 3, "seed": 2042, "shapes": map[string]any{"image": []int{2, 3, 8, 8}}}})
		quantArtifact := quantResult["optimization"].(map[string]any)["artifact"].(map[string]any)
		diagnosticJob := call("diagnostic.start", map[string]any{"project": project, "config": map[string]any{"fp32_model_id": job["id"], "quantized_model_id": quantArtifact["id"], "dataset_id": diagnosticData["id"], "max_samples": 2}})
		deadline = time.Now().Add(30 * time.Second)
		for diagnosticJob["state"] != "COMPLETED" && time.Now().Before(deadline) {
			if diagnosticJob["state"] == "FAILED" {
				t.Fatal(diagnosticJob["error"])
			}
			time.Sleep(100 * time.Millisecond)
			diagnosticJob = call("diagnostic.status", map[string]any{"id": diagnosticJob["id"]})
		}
		diagnosticResult := call("diagnostic.load", map[string]any{"project": project, "id": diagnosticJob["id"]})
		if diagnosticResult["status"] != "Succeeded" || len(diagnosticResult["captures"].([]any)) != 2 || diagnosticResult["matching_summary"].(map[string]any)["comparable_tensors"].(float64) < 1 {
			t.Fatal(diagnosticResult)
		}
		call("diagnostic.export", map[string]any{"project": project, "id": diagnosticJob["id"], "directory": t.TempDir()})
	}
	call("project.load", map[string]any{"path": project})
	if _, err := app.Request("model.load", `{"project":"`+filepath.ToSlash(project)+`","id":"../escape"}`); err == nil {
		t.Fatal("invalid artifact ID accepted")
	}
}
