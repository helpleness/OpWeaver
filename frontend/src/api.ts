export interface WorkerEvent {
  event: string;
  job_id?: string;
  data?: any;
  message?: string;
}

declare global {
  interface Window {
    go?: {
      main: {
        App: { Request(method: string, payload: string): Promise<string> };
      };
    };
    runtime?: {
      EventsOn(name: string, callback: (data: string) => void): () => void;
    };
  }
}

export async function rpc<T = any>(
  method: string,
  params: unknown = {},
): Promise<T> {
  if (window.go?.main?.App) {
    return JSON.parse(
      await window.go.main.App.Request(method, JSON.stringify(params)),
    );
  }
  const response = await fetch("/api/rpc", {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-OpWeaver": "1" },
    body: JSON.stringify({ method, params }),
  });
  if (!response.ok)
    throw new Error(
      `Backend unavailable (${response.status}). Start the Go host.`,
    );
  const envelope = await response.json();
  if (envelope.error) throw new Error(envelope.error);
  return envelope.result as T;
}

export function subscribe(callback: (event: WorkerEvent) => void): () => void {
  if (window.runtime?.EventsOn) {
    return window.runtime.EventsOn("worker:event", (data) =>
      callback(JSON.parse(data)),
    );
  }
  const events = new EventSource("/api/events");
  events.onmessage = (event) => callback(JSON.parse(event.data));
  return () => events.close();
}
