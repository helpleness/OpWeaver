// Package project writes immutable graph/layout revisions and atomically publishes a manifest.
package project

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

const ManifestName = "project.opweaver.json"

type Manifest struct {
	Version    int    `json:"project_version"`
	Name       string `json:"name"`
	GraphFile  string `json:"graph_file"`
	LayoutFile string `json:"layout_file"`
}

type Snapshot struct {
	Path     string          `json:"path"`
	Manifest Manifest        `json:"manifest"`
	Graph    json.RawMessage `json:"graph"`
	Layout   json.RawMessage `json:"layout"`
}

func validateLayout(data json.RawMessage) error {
	var layout struct {
		Version   int `json:"version"`
		Positions map[string]struct {
			X float64 `json:"x"`
			Y float64 `json:"y"`
		} `json:"positions"`
	}
	if err := json.Unmarshal(data, &layout); err != nil {
		return fmt.Errorf("invalid layout: %w", err)
	}
	if layout.Version != 1 || layout.Positions == nil {
		return errors.New("layout requires version 1 and a positions map")
	}
	return nil
}

func atomicWrite(path string, data []byte) error {
	file, err := os.CreateTemp(filepath.Dir(path), ".opweaver-*")
	if err != nil {
		return err
	}
	tmp := file.Name()
	defer os.Remove(tmp)
	if _, err = file.Write(data); err != nil {
		file.Close()
		return err
	}
	if err = file.Sync(); err != nil {
		file.Close()
		return err
	}
	if err = file.Close(); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

func Save(root, name string, graph, layout json.RawMessage) (*Snapshot, error) {
	if !json.Valid(graph) || !json.Valid(layout) {
		return nil, errors.New("graph and layout must be valid JSON")
	}
	if err := validateLayout(layout); err != nil {
		return nil, err
	}
	root, err := filepath.Abs(root)
	if err != nil {
		return nil, err
	}
	if _, err := os.Stat(filepath.Join(root, ManifestName)); err != nil {
		return nil, fmt.Errorf("open or create a project before saving: %w", err)
	}
	stamp := fmt.Sprint(time.Now().UnixNano())
	manifest := Manifest{1, name, "graph." + stamp + ".json", "layout." + stamp + ".json"}
	if err = atomicWrite(filepath.Join(root, manifest.GraphFile), graph); err != nil {
		return nil, err
	}
	if err = atomicWrite(filepath.Join(root, manifest.LayoutFile), layout); err != nil {
		return nil, err
	}
	data, _ := json.MarshalIndent(manifest, "", "  ")
	if err = atomicWrite(filepath.Join(root, ManifestName), data); err != nil {
		return nil, err
	}
	return &Snapshot{root, manifest, graph, layout}, nil
}

func Create(root, name string, graph, layout json.RawMessage) (*Snapshot, error) {
	if !json.Valid(graph) {
		return nil, errors.New("invalid graph JSON")
	}
	if err := validateLayout(layout); err != nil {
		return nil, err
	}
	if root == "" || name == "" {
		return nil, errors.New("project directory and name are required")
	}
	entries, err := os.ReadDir(root)
	if err != nil && !os.IsNotExist(err) {
		return nil, err
	}
	if len(entries) > 0 {
		return nil, errors.New("new project directory must be empty; choose a new folder")
	}
	if err = os.MkdirAll(root, 0755); err != nil {
		return nil, err
	}
	file, err := os.OpenFile(filepath.Join(root, ManifestName), os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0644)
	if err != nil {
		return nil, err
	}
	file.Close()
	return Save(root, name, graph, layout)
}

func Load(root string) (*Snapshot, error) {
	if filepath.Base(root) == ManifestName {
		root = filepath.Dir(root)
	}
	root, err := filepath.Abs(root)
	if err != nil {
		return nil, err
	}
	data, err := os.ReadFile(filepath.Join(root, ManifestName))
	if err != nil {
		return nil, err
	}
	var manifest Manifest
	if err = json.Unmarshal(data, &manifest); err != nil {
		return nil, err
	}
	if manifest.Version != 1 {
		return nil, errors.New("unsupported project format")
	}
	read := func(name string) (json.RawMessage, error) {
		if name == "" || filepath.Base(name) != name || name == "." || name == ".." {
			return nil, errors.New("project artifact must be a filename within the project")
		}
		data, err := os.ReadFile(filepath.Join(root, name))
		if err != nil {
			return nil, err
		}
		if !json.Valid(data) {
			return nil, errors.New("invalid project artifact JSON: " + name)
		}
		return data, nil
	}
	graph, err := read(manifest.GraphFile)
	if err != nil {
		return nil, err
	}
	layout, err := read(manifest.LayoutFile)
	if err != nil {
		return nil, err
	}
	if err = validateLayout(layout); err != nil {
		return nil, err
	}
	return &Snapshot{root, manifest, graph, layout}, nil
}
