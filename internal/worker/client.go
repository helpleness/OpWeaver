// Package worker owns the Python process, request correlation and crash recovery.
package worker

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"sync"
	"sync/atomic"
	"time"
)

type response struct {
	Result json.RawMessage `json:"result"`
	Error  *struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
}

type Client struct {
	Root     string
	Emit     func(json.RawMessage)
	mu       sync.Mutex
	writeMu  sync.Mutex
	cmd      *exec.Cmd
	stdin    io.WriteCloser
	pending  map[string]chan response
	sequence atomic.Uint64
	closed   bool
}

func New(root string, emit func(json.RawMessage)) *Client {
	return &Client{Root: root, Emit: emit, pending: make(map[string]chan response)}
}

func (c *Client) event(value any) {
	if c.Emit != nil {
		data, _ := json.Marshal(value)
		c.Emit(data)
	}
}

func resolvePython(root string) string {
	if configured := os.Getenv("OPWEAVER_PYTHON"); configured != "" {
		return configured
	}
	// A development build may live under build/bin with bundled worker resources,
	// while its virtual environment remains at the repository root.
	for directory := root; ; directory = filepath.Dir(directory) {
		for _, relative := range []string{filepath.Join(".venv", "Scripts", "python.exe"), filepath.Join(".venv", "bin", "python")} {
			candidate := filepath.Join(directory, relative)
			if info, err := os.Stat(candidate); err == nil && !info.IsDir() {
				return candidate
			}
		}
		if filepath.Dir(directory) == directory {
			break
		}
	}
	return "python"
}

func (c *Client) startLocked() error {
	if c.closed {
		return errors.New("worker manager is closed")
	}
	if c.cmd != nil {
		return nil
	}
	python := resolvePython(c.Root)
	cmd := exec.Command(python, "-u", "-m", "opweaver.worker")
	configureProcess(cmd)
	cmd.Dir = c.Root
	cmd.Env = append(os.Environ(), "PYTHONPATH="+filepath.Join(c.Root, "worker"), "PYTHONUNBUFFERED=1", "PYTHONIOENCODING=utf-8")
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return err
	}
	if err = cmd.Start(); err != nil {
		return fmt.Errorf("start Python worker: %w; set OPWEAVER_PYTHON to a Python with torch and pydantic", err)
	}
	c.cmd, c.stdin = cmd, stdin
	readDone := make(chan struct{})
	go func() { c.read(stdout, cmd); close(readDone) }()
	go func() {
		scanner := bufio.NewScanner(stderr)
		for scanner.Scan() {
			c.event(map[string]any{"event": "worker.stderr", "message": scanner.Text()})
		}
	}()
	go func() {
		// Drain stdout before Wait closes its pipe; terminal responses must not be lost.
		<-readDone
		err := cmd.Wait()
		c.mu.Lock()
		if c.cmd == cmd {
			c.cmd, c.stdin = nil, nil
		}
		for id, waiting := range c.pending {
			waiting <- response{Error: &struct {
				Code    string `json:"code"`
				Message string `json:"message"`
			}{"WorkerExited", "Python worker exited; the next request will restart it"}}
			delete(c.pending, id)
		}
		closed := c.closed
		c.mu.Unlock()
		if !closed {
			c.event(map[string]any{"event": "worker.exited", "message": fmt.Sprint(err)})
		}
	}()
	return nil
}

func (c *Client) read(reader io.Reader, cmd *exec.Cmd) {
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 65536), 32*1024*1024)
	for scanner.Scan() {
		data := append(json.RawMessage(nil), scanner.Bytes()...)
		var envelope struct {
			ID    string `json:"id"`
			Event string `json:"event"`
			response
		}
		if err := json.Unmarshal(data, &envelope); err != nil {
			c.event(map[string]any{"event": "worker.stderr", "message": "Invalid worker response: " + err.Error()})
			continue
		}
		if envelope.Event != "" {
			if c.Emit != nil {
				c.Emit(data)
			}
			continue
		}
		c.mu.Lock()
		if ch, ok := c.pending[envelope.ID]; ok {
			ch <- envelope.response
			delete(c.pending, envelope.ID)
		}
		c.mu.Unlock()
	}
	if err := scanner.Err(); err != nil {
		c.event(map[string]any{"event": "worker.stderr", "message": err.Error()})
		_ = cmd.Process.Kill()
	}
}

func (c *Client) Call(ctx context.Context, method string, params any) (json.RawMessage, error) {
	c.mu.Lock()
	if err := c.startLocked(); err != nil {
		c.mu.Unlock()
		return nil, err
	}
	id := strconv.FormatUint(c.sequence.Add(1), 10)
	ch := make(chan response, 1)
	c.pending[id] = ch
	stdin := c.stdin
	c.mu.Unlock()
	defer func() { c.mu.Lock(); delete(c.pending, id); c.mu.Unlock() }()
	data, err := json.Marshal(map[string]any{"protocol": 1, "id": id, "method": method, "params": params})
	if err != nil {
		return nil, err
	}
	c.writeMu.Lock()
	_, err = stdin.Write(append(data, '\n'))
	c.writeMu.Unlock()
	if err != nil {
		return nil, fmt.Errorf("write worker request: %w", err)
	}
	select {
	case answer := <-ch:
		if answer.Error != nil {
			return nil, fmt.Errorf("%s: %s", answer.Error.Code, answer.Error.Message)
		}
		return answer.Result, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func (c *Client) Close() {
	c.mu.Lock()
	running := c.cmd != nil
	c.mu.Unlock()
	if running {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		_, _ = c.Call(ctx, "shutdown", map[string]any{})
		cancel()
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.closed = true
	if c.cmd != nil {
		_ = c.cmd.Process.Kill()
	}
}
