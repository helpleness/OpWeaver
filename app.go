package main

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/wailsapp/wails/v2/pkg/runtime"
	"opweaver/internal/project"
	"opweaver/internal/worker"
)

type App struct {
	ctx         context.Context
	root        string
	worker      *worker.Client
	mu          sync.Mutex
	subscribers map[chan json.RawMessage]bool
	desktop     bool
	projectMu   sync.Mutex
}

func NewApp(root string) *App {
	app := &App{root: root, subscribers: make(map[chan json.RawMessage]bool)}
	app.worker = worker.New(root, app.emit)
	return app
}

func (a *App) startup(ctx context.Context)  { a.ctx = ctx; a.desktop = true }
func (a *App) shutdown(ctx context.Context) { a.worker.Close() }

func (a *App) emit(data json.RawMessage) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.desktop {
		runtime.EventsEmit(a.ctx, "worker:event", string(data))
	}
	for ch := range a.subscribers {
		select {
		case ch <- data:
		default:
		}
	}
}

// Request keeps Wails and local HTTP adapters on exactly the same API.
func (a *App) Request(method, payload string) (string, error) {
	var params map[string]json.RawMessage
	if err := json.Unmarshal([]byte(payload), &params); err != nil {
		return "", err
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	readString := func(key string) string { var value string; _ = json.Unmarshal(params[key], &value); return value }
	var result any
	switch method {
	case "app.info":
		result = map[string]any{"root": a.root, "default_project_dir": filepath.Join(a.root, "projects"), "desktop": a.desktop}
	case "dialog.open":
		if !a.desktop {
			return "", errors.New("native file dialog is available in the desktop app; enter a directory path instead")
		}
		path, err := runtime.OpenFileDialog(a.ctx, runtime.OpenDialogOptions{Title: "Open OpWeaver project", Filters: []runtime.FileFilter{{DisplayName: "OpWeaver project", Pattern: "*.opweaver.json"}}})
		if err != nil {
			return "", err
		}
		result = path
	case "dialog.directory":
		if !a.desktop {
			return "", errors.New("native directory dialog is available in the desktop app")
		}
		path, err := runtime.OpenDirectoryDialog(a.ctx, runtime.OpenDialogOptions{Title: "Choose parent directory for new project"})
		if err != nil {
			return "", err
		}
		result = path
	case "dialog.python":
		if !a.desktop {
			return "", errors.New("enter a Python file path in browser mode")
		}
		path, err := runtime.OpenFileDialog(a.ctx, runtime.OpenDialogOptions{Title: "Import Python source (does not execute it)", Filters: []runtime.FileFilter{{DisplayName: "Python", Pattern: "*.py"}}})
		if err != nil {
			return "", err
		}
		result = path
	case "workspace.list":
		files, err := project.ListSources(readString("project"))
		if err != nil {
			return "", err
		}
		result = files
	case "workspace.read":
		file, err := project.ReadSource(readString("project"), readString("path"))
		if err != nil {
			return "", err
		}
		result = file
	case "workspace.write":
		a.projectMu.Lock()
		defer a.projectMu.Unlock()
		file, err := project.WriteSource(readString("project"), readString("path"), readString("text"), readString("expected_hash"))
		if err != nil {
			return "", err
		}
		result = file
	case "workspace.import":
		a.projectMu.Lock()
		defer a.projectMu.Unlock()
		sourcePath := readString("source")
		if filepath.Ext(sourcePath) != ".py" {
			return "", errors.New("only Python source can be imported")
		}
		file, err := os.Open(sourcePath)
		if err != nil {
			return "", err
		}
		defer file.Close()
		data, err := io.ReadAll(io.LimitReader(file, 2*1024*1024+1))
		if err != nil {
			return "", err
		}
		if len(data) > 2*1024*1024 {
			return "", errors.New("source exceeds 2 MiB")
		}
		imported, err := project.WriteSource(readString("project"), "src/"+filepath.Base(sourcePath), string(data), "")
		if err != nil {
			return "", err
		}
		result = imported
	case "project.load":
		a.projectMu.Lock()
		defer a.projectMu.Unlock()
		snapshot, err := project.Load(readString("path"))
		if err != nil {
			return "", err
		}
		if _, err = a.worker.Call(ctx, "validate", map[string]any{"graph": snapshot.Graph}); err != nil {
			return "", err
		}
		result = snapshot
	case "project.create", "project.save":
		a.projectMu.Lock()
		defer a.projectMu.Unlock()
		if _, err := a.worker.Call(ctx, "validate", map[string]any{"graph": params["graph"]}); err != nil {
			return "", err
		}
		var err error
		if method == "project.create" {
			result, err = project.Create(readString("path"), readString("name"), params["graph"], params["layout"])
		} else {
			result, err = project.Save(readString("path"), readString("name"), params["graph"], params["layout"])
		}
		if err != nil {
			return "", err
		}
	case "example":
		name := readString("name")
		if name != "synthetic-mlp" && name != "residual-mlp" {
			return "", errors.New("unknown example")
		}
		snapshot, err := project.Load(filepath.Join(a.root, "examples", name))
		if err != nil {
			return "", err
		}
		result = snapshot
	default:
		codeMethods := map[string]bool{"code.inspect": true, "code.generate": true, "code.patch": true, "code.capture.start": true, "code.capture.status": true, "code.capture.stop": true}
		if codeMethods[method] {
			if method == "code.capture.start" {
				path, err := project.WorkspacePath(readString("project"), readString("path"))
				if err != nil {
					return "", err
				}
				file, err := project.ReadSource(readString("project"), readString("path"))
				if err != nil {
					return "", err
				}
				if file.Hash != readString("expected_hash") {
					return "", errors.New("source conflict: file changed before capture")
				}
				params["file_path"], _ = json.Marshal(path)
			}
			data, err := a.worker.Call(ctx, method, params)
			return string(data), err
		}
		allowed := map[string]bool{"environment": true, "registry": true, "validate": true, "forward": true, "train.start": true, "train.stop": true, "train.status": true, "train.history": true}
		if !allowed[method] {
			return "", errors.New("unknown API method: " + method)
		}
		if method == "train.start" || method == "train.history" {
			if _, err := project.Load(readString("project")); err != nil {
				return "", err
			}
		}
		data, err := a.worker.Call(ctx, method, params)
		return string(data), err
	}
	data, err := json.Marshal(result)
	return string(data), err
}

func findRoot() string {
	if root := os.Getenv("OPWEAVER_HOME"); root != "" {
		return root
	}
	cwd, _ := os.Getwd()
	executable, _ := os.Executable()
	for _, start := range []string{cwd, filepath.Dir(executable)} {
		for dir := start; ; dir = filepath.Dir(dir) {
			if _, err := os.Stat(filepath.Join(dir, "worker", "opweaver", "worker.py")); err == nil {
				return dir
			}
			if filepath.Dir(dir) == dir {
				break
			}
		}
	}
	return cwd
}
