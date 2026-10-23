package project

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func workspaceProject(t *testing.T) string {
	t.Helper()
	root := filepath.Join(t.TempDir(), "project")
	_, err := Create(root, "code", json.RawMessage(`{}`), json.RawMessage(`{"version":1,"positions":{}}`))
	if err != nil {
		t.Fatal(err)
	}
	return root
}

func TestSourceRoundTripAndConflict(t *testing.T) {
	root := workspaceProject(t)
	first, err := WriteSource(root, "src/model.py", "# 手写代码\n", "")
	if err != nil {
		t.Fatal(err)
	}
	second, err := WriteSource(root, "src/model.py", first.Text+"x = 1\n", first.Hash)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = WriteSource(root, "src/model.py", "overwrite", first.Hash); err == nil {
		t.Fatal("stale source overwrote a newer file")
	}
	read, err := ReadSource(root, "src/model.py")
	if err != nil || read.Hash != second.Hash {
		t.Fatal("source round trip failed", err)
	}
	list, err := ListSources(root)
	if err != nil || len(list) != 1 {
		t.Fatal(list, err)
	}
}

func TestSourcePathsCannotEscapeProject(t *testing.T) {
	root := workspaceProject(t)
	for _, name := range []string{"../outside.py", "src/../../outside.py", "project.opweaver.json", "src/model.exe"} {
		if _, err := WriteSource(root, name, "bad", ""); err == nil {
			t.Fatal("accepted path", name)
		}
	}
	outside := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "src"), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, filepath.Join(root, "src", "linked")); err == nil {
		if _, err = WriteSource(root, "src/linked/bad.py", "bad", ""); err == nil {
			t.Fatal("followed symlink outside project")
		}
	}
}
