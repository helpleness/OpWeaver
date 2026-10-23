package main

import (
	"net/http"
	"net/http/httptest"
	"testing"
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
