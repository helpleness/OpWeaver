package project

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestSaveLoadRevisions(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "model")
	graph := json.RawMessage(`{"nodes":[{"id":"a"},{"id":"b"}],"edges":[{"source":"a","target":"b"}]}`)
	layout := json.RawMessage(`{"version":1,"positions":{"a":{"x":1,"y":2}}}`)
	first, err := Create(dir, "test", graph, layout)
	if err != nil {
		t.Fatal(err)
	}
	second, err := Save(dir, "test 2", graph, layout)
	if err != nil {
		t.Fatal(err)
	}
	if first.Manifest.GraphFile == second.Manifest.GraphFile {
		t.Fatal("revision was overwritten")
	}
	loaded, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	if string(loaded.Graph) != string(graph) || string(loaded.Layout) != string(layout) || loaded.Manifest.Name != "test 2" {
		t.Fatal("round trip lost data")
	}
	if _, err = Create(dir, "clobber", graph, layout); err == nil {
		t.Fatal("overwrote existing project")
	}
}

func TestRejectArtifactTraversal(t *testing.T) {
	dir := t.TempDir()
	data := []byte(`{"project_version":1,"graph_file":"../secret.json","layout_file":"layout.json"}`)
	if err := os.WriteFile(filepath.Join(dir, ManifestName), data, 0644); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(dir); err == nil {
		t.Fatal("accepted traversal")
	}
}

func TestRejectMalformedLayoutWithoutCreatingProject(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "broken")
	_, err := Create(dir, "broken", json.RawMessage(`{}`), json.RawMessage(`{"version":1,"positions":"bad"}`))
	if err == nil {
		t.Fatal("accepted malformed layout")
	}
	if _, err = os.Stat(dir); !os.IsNotExist(err) {
		t.Fatal("created a partial project for invalid layout")
	}
}
