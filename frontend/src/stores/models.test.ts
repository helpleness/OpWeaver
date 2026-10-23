// @vitest-environment jsdom
import { beforeEach, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { rpc } from "../api";
import { useModels } from "./models";
vi.mock("../api", () => ({ rpc: vi.fn() }));
beforeEach(() => {
  setActivePinia(createPinia());
  vi.mocked(rpc).mockResolvedValue([]);
});
it("failed imports retain the last valid analysis", async () => {
  const store = useModels();
  store.report = { artifact: { id: "valid" }, analysis: { nodes: [] } } as any;
  await store.accept({
    id: "bad",
    project: "",
    state: "FAILED",
    file_path: "bad.onnx",
    created_at: 0,
    error: "Invalid protobuf",
  });
  expect(store.report?.artifact.id).toBe("valid");
  expect(store.error).toContain("Invalid protobuf");
});
it("late model events cannot replace the active project", async () => {
  const store = useModels();
  await store.accept({
    id: "other",
    project: "different-project",
    state: "COMPLETED",
    file_path: "x.onnx",
    created_at: 0,
    error: null,
    result: { success: true } as any,
  });
  expect(store.job).toBeNull();
  expect(store.report).toBeNull();
});
