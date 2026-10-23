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
	call("project.load", map[string]any{"path": project})
	if _, err := app.Request("model.load", `{"project":"`+filepath.ToSlash(project)+`","id":"../escape"}`); err == nil {
		t.Fatal("invalid artifact ID accepted")
	}
}
