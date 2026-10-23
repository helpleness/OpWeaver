// @vitest-environment jsdom
import { beforeEach, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { flushPromises } from "@vue/test-utils";
import { rpc } from "../api";
import { useLab } from "./lab";
import { useEditor } from "./editor";
import { defaultConfig, metric, percent } from "../lab-types";
vi.mock("../api", () => ({ rpc: vi.fn() }));
beforeEach(() => {
  setActivePinia(createPinia());
  localStorage.clear();
  vi.mocked(rpc).mockResolvedValue([]);
});
it("late experiment events cannot replace the active project or training job", async () => {
  const lab = useLab(),
    editor = useEditor();
  editor.job = { id: "training" } as any;
  await lab.accept({
    id: "other",
    project: "different",
    state: "COMPLETED",
    status: "Succeeded",
    created_at: 1,
  });
  expect(lab.job).toBeNull();
  expect(editor.job?.id).toBe("training");
});
it("terminal snapshots cannot regress to an earlier running event", async () => {
  const lab = useLab();
  const result = { id: "x", status: "Failed", config: defaultConfig() } as any;
  await lab.accept({
    id: "x",
    project: "",
    state: "FAILED",
    status: "Failed",
    created_at: 1,
    error: "Invalid input",
    result,
  });
  await lab.accept({
    id: "x",
    project: "",
    state: "RUNNING",
    status: "Running",
    created_at: 1,
  });
  expect(lab.running).toBe(false);
  expect(lab.experiment?.status).toBe("Failed");
  expect(lab.error).toBe("Invalid input");
});
it("worker disconnection ends the visible job and keeps prior reports", () => {
  const lab = useLab();
  lab.job = {
    id: "x",
    project: "",
    state: "RUNNING",
    status: "Running",
    created_at: 1,
  };
  lab.workerExited();
  expect(lab.job.state).toBe("FAILED");
  expect(lab.running).toBe(false);
});
it("experiment configuration and report reopen from the backend", async () => {
  const lab = useLab();
  const config = defaultConfig();
  config.benchmark.runs = 53;
  vi.mocked(rpc)
    .mockResolvedValueOnce({ id: "saved", source_model_id: "source", config })
    .mockResolvedValueOnce({ artifact: { id: "source" } });
  await lab.open("saved");
  expect(lab.config.benchmark.runs).toBe(53);
  expect(lab.experiment?.id).toBe("saved");
  expect(lab.source?.artifact.id).toBe("source");
});
it("missing metrics stay unavailable and negative changes remain negative", () => {
  expect(metric(null)).toBe("—");
  expect(metric(NaN)).toBe("—");
  expect(percent(-12)).toBe("-12.00%");
  expect(percent(20)).toBe("+20.00%");
});

it("workspace recovery stays loading until the saved experiment and model are restored", async () => {
  const lab = useLab(),
    editor = useEditor();
  editor.projectPath = "restore-project";
  await flushPromises();
  localStorage.setItem("opweaver.lab.restore-project", "saved");
  let release!: (report: any) => void;
  vi.mocked(rpc).mockImplementation((method) => {
    if (method === "lab.history") return Promise.resolve([{ id: "saved" }]);
    if (method === "lab.load")
      return new Promise((resolve) => (release = resolve));
    if (method === "model.load")
      return Promise.resolve({ artifact: { id: "source" } });
    return Promise.resolve([]);
  });
  const restoring = lab.loadWorkspace();
  await flushPromises();
  expect(lab.loading).toBe(true);
  expect(lab.experiment).toBeNull();
  const config = defaultConfig();
  config.evaluation.dataset_id = "a".repeat(32);
  release({ id: "saved", source_model_id: "source", config });
  await restoring;
  expect(lab.loading).toBe(false);
  expect(lab.experiment?.id).toBe("saved");
  expect(lab.config.evaluation.dataset_id).toBe("a".repeat(32));
});

it("cancel during workflow preflight stops the new job instead of a previous experiment", async () => {
  const lab = useLab(),
    editor = useEditor();
  editor.projectPath = "project";
  await flushPromises();
  lab.source = { artifact: { id: "a".repeat(32) } } as any;
  lab.config.source_model_id = "a".repeat(32);
  lab.job = {
    id: "previous",
    project: "project",
    state: "COMPLETED",
    status: "Succeeded",
    created_at: 1,
  };
  let release!: (job: any) => void;
  vi.mocked(rpc).mockImplementation((method) => {
    if (method === "lab.start")
      return new Promise((resolve) => (release = resolve));
    if (method === "lab.stop")
      return Promise.resolve({
        id: "new",
        project: "project",
        state: "CANCELLED",
        status: "Cancelled",
        created_at: 2,
      });
    return Promise.resolve([]);
  });
  const pending = lab.start("optimize");
  await lab.stop();
  expect(
    vi.mocked(rpc).mock.calls.filter(([method]) => method === "lab.stop"),
  ).toEqual([]);
  release({
    id: "new",
    project: "project",
    state: "RUNNING",
    status: "Running",
    created_at: 2,
  });
  await pending;
  expect(rpc).toHaveBeenCalledWith("lab.stop", { id: "new" });
  expect(lab.job?.state).toBe("CANCELLED");
});
