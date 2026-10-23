package project

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
)

type SourceFile struct {
	Path string `json:"path"`
	Text string `json:"text"`
	Hash string `json:"hash"`
}

func ContentHash(data []byte) string { sum := sha256.Sum256(data); return hex.EncodeToString(sum[:]) }

func within(root, candidate string) bool {
	relative, err := filepath.Rel(root, candidate)
	return err == nil && relative != ".." && !strings.HasPrefix(relative, ".."+string(filepath.Separator)) && !filepath.IsAbs(relative)
}

// WorkspacePath restricts editable files and follows existing symlinks before access.
func WorkspacePath(root, relative string) (string, error) {
	if _, err := Load(root); err != nil {
		return "", err
	}
	root, err := filepath.Abs(root)
	if err != nil {
		return "", err
	}
	root, err = filepath.EvalSymlinks(root)
	if err != nil {
		return "", err
	}
	clean := filepath.ToSlash(filepath.Clean(relative))
	if filepath.IsAbs(relative) || !(strings.HasPrefix(clean, "src/") && strings.HasSuffix(clean, ".py") || clean == ".opweaver/code-state.json") {
		return "", errors.New("workspace files must be src/*.py or the versioned code-state document")
	}
	target := filepath.Join(root, filepath.FromSlash(clean))
	if !within(root, target) {
		return "", errors.New("workspace path escapes the project")
	}
	for directory := target; ; directory = filepath.Dir(directory) {
		if _, err := os.Lstat(directory); err == nil {
			resolved, err := filepath.EvalSymlinks(directory)
			if err != nil {
				return "", err
			}
			if !within(root, resolved) {
				return "", errors.New("workspace symlink escapes the project")
			}
			break
		} else if !os.IsNotExist(err) {
			return "", err
		}
		if directory == root {
			break
		}
	}
	return target, nil
}

func ReadSource(root, relative string) (*SourceFile, error) {
	path, err := WorkspacePath(root, relative)
	if err != nil {
		return nil, err
	}
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	data, err := io.ReadAll(io.LimitReader(file, 2*1024*1024+1))
	if err != nil {
		return nil, err
	}
	if len(data) > 2*1024*1024 {
		return nil, errors.New("source file exceeds the 2 MiB editor limit")
	}
	return &SourceFile{filepath.ToSlash(relative), string(data), ContentHash(data)}, nil
}

func WriteSource(root, relative, text, expectedHash string) (*SourceFile, error) {
	path, err := WorkspacePath(root, relative)
	if err != nil {
		return nil, err
	}
	if len(text) > 2*1024*1024 {
		return nil, errors.New("source file exceeds the 2 MiB editor limit")
	}
	existing, err := os.ReadFile(path)
	if err == nil && (expectedHash == "" || ContentHash(existing) != expectedHash) {
		return nil, errors.New("source conflict: file changed on disk; reload it before saving")
	}
	if err != nil && !os.IsNotExist(err) {
		return nil, err
	}
	if os.IsNotExist(err) && expectedHash != "" {
		return nil, errors.New("source conflict: previously opened file was removed")
	}
	if err = os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return nil, err
	}
	if err = atomicWrite(path, []byte(text)); err != nil {
		return nil, err
	}
	return &SourceFile{filepath.ToSlash(relative), text, ContentHash([]byte(text))}, nil
}

func ListSources(root string) ([]string, error) {
	if _, err := Load(root); err != nil {
		return nil, err
	}
	var sources = []string{}
	err := filepath.WalkDir(filepath.Join(root, "src"), func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			if os.IsNotExist(err) {
				return nil
			}
			return err
		}
		if entry.Type()&os.ModeSymlink != 0 {
			if entry.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		if !entry.IsDir() && strings.HasSuffix(path, ".py") {
			relative, err := filepath.Rel(root, path)
			if err != nil {
				return err
			}
			sources = append(sources, filepath.ToSlash(relative))
		}
		return nil
	})
	return sources, err
}
