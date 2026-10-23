package worker

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func testClient(t *testing.T) *Client {
	t.Helper()
	root, err := filepath.Abs("../..")
	if err != nil {
		t.Fatal(err)
	}
	if os.Getenv("OPWEAVER_INTEGRATION") != "1" {
		t.Skip("set OPWEAVER_INTEGRATION=1 to test against installed Python/PyTorch")
	}
	c := New(root, nil)
	t.Cleanup(c.Close)
	return c
}

func TestRealWorkerErrorRecoveryAndRestart(t *testing.T) {
	c := testClient(t)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	if _, err := c.Call(ctx, "invalid.method", map[string]any{}); err == nil {
		t.Fatal("expected protocol error")
	}
	result, err := c.Call(ctx, "environment", map[string]any{})
	if err != nil {
		t.Fatal(err)
	}
	var env map[string]any
	if err = json.Unmarshal(result, &env); err != nil {
		t.Fatal(err)
	}
	if env["pytorch"] == nil {
		t.Fatal("missing real environment")
	}
	c.mu.Lock()
	process := c.cmd.Process
	c.mu.Unlock()
	if err = process.Kill(); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(10 * time.Second)
	for {
		c.mu.Lock()
		stopped := c.cmd == nil
		c.mu.Unlock()
		if stopped {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("worker exit was not detected")
		}
		time.Sleep(10 * time.Millisecond)
	}
	if _, err = c.Call(ctx, "registry", map[string]any{}); err != nil {
		t.Fatalf("worker failed to restart: %v", err)
	}
}

func TestBundledWorkerFindsRepositoryVirtualenv(t *testing.T) {
	t.Setenv("OPWEAVER_PYTHON", "")
	repo := t.TempDir()
	interpreter := filepath.Join(repo, ".venv", "Scripts", "python.exe")
	if err := os.MkdirAll(filepath.Dir(interpreter), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(interpreter, []byte("test"), 0644); err != nil {
		t.Fatal(err)
	}
	if actual := resolvePython(filepath.Join(repo, "build", "bin")); actual != interpreter {
		t.Fatalf("got %q, expected %q", actual, interpreter)
	}
	t.Setenv("OPWEAVER_PYTHON", "explicit-python")
	if actual := resolvePython(repo); actual != "explicit-python" {
		t.Fatal("explicit interpreter was not respected")
	}
}
