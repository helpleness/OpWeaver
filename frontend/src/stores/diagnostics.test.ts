// @vitest-environment jsdom
import { beforeEach, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { rpc } from "../api";
import { diagnosticDefaults } from "../diagnostic-types";
import { useEditor } from "./editor";
import { useDiagnostics } from "./diagnostics";
import { useModels } from "./models";
vi.mock("../api", () => ({ rpc: vi.fn() }));
beforeEach(() => {
  setActivePinia(createPinia());
  localStorage.clear();
  vi.mocked(rpc).mockResolvedValue([]);
});
it("isolates diagnostics events and drops late running events after terminal state", async () => {
  const store = useDiagnostics(),
    editor = useEditor();
  editor.job = { id: "training" } as any;
  await store.accept({
    id: "other",
    project: "different",
    state: "RUNNING",
    status: "Running",
    created_at: 1,
  });
  expect(store.job).toBeNull();
  await store.accept({
    id: "saved",
    project: "",
    state: "COMPLETED",
    status: "Succeeded",
    created_at: 1,
    result: { id: "saved", config: diagnosticDefaults(), records: [] } as any,
  });
  await store.accept({
    id: "saved",
    project: "",
    state: "RUNNING",
    status: "Running",
    created_at: 1,
  });
  expect(store.running).toBe(false);
  expect(store.report?.id).toBe("saved");
  expect(editor.job?.id).toBe("training");
});
it("restores configuration and a report, then handles worker exit without fabricating statistics", async () => {
  const store = useDiagnostics();
  const config = { ...diagnosticDefaults(), max_samples: 7 };
  vi.mocked(rpc).mockResolvedValueOnce({ id: "report", config, records: [] });
  await store.open("report");
  expect(store.config.max_samples).toBe(7);
  store.job = {
    id: "live",
    project: "",
    state: "RUNNING",
    status: "Running",
    created_at: 1,
  };
  store.workerExited();
  expect(store.running).toBe(false);
  expect(store.job.state).toBe("FAILED");
  expect(store.report?.id).toBe("report");
});
it("opens the actual counterpart graph and attaches diagnostic node focus/badges", async () => {
  const models = useModels();
  const report = {
    id: "diag",
    models: {
      fp32: { id: "original", sha256: "hash" },
      quantized: { id: "quant", sha256: "hashq" },
    },
    records: [
      {
        id: "row",
        kind: "activation",
        fp32_node_id: "onnx_0",
        quantized_node_id: "onnx_5",
        aggregate: { mae: 2, valid_samples: 1 },
      },
    ],
  } as any;
  vi.mocked(rpc).mockResolvedValueOnce({
    artifact: { id: "quant", sha256: "hashq" },
    analysis: { nodes: [] },
  });
  await models.openDiagnostic(report, report.records[0], "quantized", 0.1, 1);
  expect(models.focusRequest?.nodeId).toBe("onnx_5");
  expect(models.diagnosticBadges).toEqual({ onnx_5: "high" });
  expect(models.diagnosticContext?.report.id).toBe("diag");
});
