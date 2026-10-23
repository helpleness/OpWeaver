package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"log"
	"net/http"
	"os"
	"os/signal"
	"time"
)

func localOnly(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		allowedHosts := map[string]bool{"127.0.0.1:34115": true, "localhost:34115": true, "127.0.0.1:5173": true, "localhost:5173": true}
		allowedOrigins := map[string]bool{"": true, "http://127.0.0.1:34115": true, "http://localhost:34115": true, "http://127.0.0.1:5173": true, "http://localhost:5173": true}
		if !allowedHosts[r.Host] || !allowedOrigins[r.Header.Get("Origin")] {
			http.Error(w, "untrusted local development origin", http.StatusForbidden)
			return
		}
		next.ServeHTTP(w, r)
	})
}

// Local-only development adapter. Native builds use Wails bindings directly.
func serve(app *App, frontend fs.FS) {
	address := "127.0.0.1:34115"
	mux := http.NewServeMux()
	mux.HandleFunc("/api/rpc", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" || r.Header.Get("X-OpWeaver") != "1" {
			http.Error(w, "unsupported request", 403)
			return
		}
		var request struct {
			Method string          `json:"method"`
			Params json.RawMessage `json:"params"`
		}
		if err := json.NewDecoder(io.LimitReader(r.Body, 8*1024*1024)).Decode(&request); err != nil {
			http.Error(w, err.Error(), 400)
			return
		}
		result, err := app.Request(request.Method, string(request.Params))
		w.Header().Set("Content-Type", "application/json")
		if err != nil {
			_ = json.NewEncoder(w).Encode(map[string]any{"error": err.Error()})
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"result": json.RawMessage(result)})
	})
	mux.HandleFunc("/api/events", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "streaming unsupported", 500)
			return
		}
		ch := make(chan json.RawMessage, 32)
		app.mu.Lock()
		app.subscribers[ch] = true
		app.mu.Unlock()
		defer func() { app.mu.Lock(); delete(app.subscribers, ch); app.mu.Unlock() }()
		fmt.Fprint(w, ": connected\n\n")
		flusher.Flush()
		for {
			select {
			case data := <-ch:
				fmt.Fprintf(w, "data: %s\n\n", data)
				flusher.Flush()
			case <-r.Context().Done():
				return
			}
		}
	})
	mux.Handle("/", http.FileServer(http.FS(frontend)))
	server := &http.Server{Addr: address, Handler: localOnly(mux), ReadHeaderTimeout: 5 * time.Second}
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt)
	go func() { <-stop; app.worker.Close(); _ = server.Shutdown(context.Background()) }()
	log.Printf("OpWeaver development server: http://%s", address)
	if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Print(err)
	}
	app.worker.Close()
}
