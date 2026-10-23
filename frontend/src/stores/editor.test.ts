// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { useEditor, emptyGraph } from "./editor";
import type { Operator } from "../types";
import { rpc } from "../api";

vi.mock("../api", () => ({
  rpc: vi
    .fn()
    .mockResolvedValue({ valid: true, issues: [], order: [], tensors: {} }),
}));
const operator = (
  id: string,
  inputs: string[],
  outputs: string[],
  defaults = {},
): Operator => ({
  id,
  version: "1",
  label: id.split(".")[1],
  category: "Layers",
  inputs,
  outputs,
  defaults,
  parameters: { properties: {} },
  description: "",
  capabilities: {
    dtypes: ["float32"],
    devices: ["cpu"],
    autograd: true,
    torch_export: false,
    onnx: false,
    custom_kernel: false,
    backends: ["pytorch.eager"],
  },
});
beforeEach(() => {
  setActivePinia(createPinia());
  vi.clearAllMocks();
});
describe("Graph editing state", () => {
  it("adds nodes and syncs parameter edits without polluting IR with layout", () => {
    const store = useEditor();
    store.registry = [
      operator("core.linear", ["in"], ["out"], {
        in_features: 8,
        out_features: 16,
      }),
    ];
    const id = store.addNode("core.linear", { x: 100, y: 200 });
    expect(store.graph.nodes).toHaveLength(1);
    store.updateParams(id, "out_features", 3);
    expect(store.graph.nodes?.[0].params?.out_features).toBe(3);
    expect(store.layout.positions[id]).toEqual({ x: 100, y: 200 });
    expect(JSON.stringify(store.graph)).not.toContain("position");
    store.undo();
    expect(store.graph.nodes?.[0].params?.out_features).toBe(16);
    store.redo();
    expect(store.graph.nodes?.[0].params?.out_features).toBe(3);
  });
  it("connects, rejects duplicate inputs and cycles, and removes attached edges", () => {
    const store = useEditor();
    store.registry = [operator("core.relu", ["in"], ["out"])];
    const a = store.addNode("core.relu"),
      b = store.addNode("core.relu");
    store.connect(a, b, "out", "in");
    expect(() => store.connect(a, b, "out", "in")).toThrow("already");
    expect(() => store.connect(b, a, "out", "in")).toThrow("cycle");
    store.removeNodes([a]);
    expect(store.graph.edges).toHaveLength(0);
    store.undo();
    expect(store.graph.edges).toHaveLength(1);
  });
  it("saves semantic graph and layout separately without dropping connections", async () => {
    const store = useEditor();
    store.registry = [operator("core.relu", ["in"], ["out"])];
    const a = store.addNode("core.relu"),
      b = store.addNode("core.relu");
    store.connect(a, b, "out", "in");
    store.projectPath = "/project";
    await store.save();
    expect(store.dirty).toBe(false);
    const payload = vi
      .mocked(rpc)
      .mock.calls.find((c) => c[0] === "project.save")?.[1] as any;
    expect(payload.graph.edges).toHaveLength(1);
    expect(payload.layout.positions[a]).toBeDefined();
    const snapshot = JSON.parse(
      JSON.stringify({ graph: store.graph, layout: store.layout }),
    );
    store.replace(emptyGraph(), { version: 1, positions: {} });
    store.replace(snapshot.graph, snapshot.layout, "/project", true);
    expect(store.graph.edges).toHaveLength(1);
    expect(store.dirty).toBe(false);
  });
  it("surfaces worker validation errors", async () => {
    const store = useEditor();
    vi.mocked(rpc).mockRejectedValueOnce(new Error("worker disconnected"));
    await expect(store.validateNow()).rejects.toThrow("disconnected");
    expect(store.validation.valid).toBe(false);
    expect(store.validation.issues[0].message).toContain("disconnected");
  });
});
