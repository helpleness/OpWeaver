// @vitest-environment jsdom
import { beforeEach, expect, it, vi } from "vitest";
import { setActivePinia, createPinia } from "pinia";
import { rpc } from "../api";
import { useEditor } from "./editor";
import { useCode } from "./code";

vi.mock("../api", () => ({ rpc: vi.fn() }));
beforeEach(() => {
  setActivePinia(createPinia());
  vi.clearAllMocks();
  vi.mocked(rpc).mockImplementation(async (method) => {
    if (method === "workspace.list") return [];
    if (method === "workspace.read") throw new Error("not found");
    if (method === "workspace.write")
      return {
        path: ".opweaver/code-state.json",
        hash: "state-hash",
        text: "",
      };
    if (method === "code.inspect") return { entries: [], diagnostics: [] };
    return { valid: true, issues: [], order: [], tensors: {} };
  });
});

it("saving uses a content hash and does not lose an unsaved file when switching", async () => {
  const code = useCode();
  code.documents = {
    "src/a.py": {
      path: "src/a.py",
      text: "old",
      saved: "old",
      hash: "old-hash",
      revision: 0,
    },
    "src/b.py": {
      path: "src/b.py",
      text: "b",
      saved: "b",
      hash: "b-hash",
      revision: 0,
    },
  };
  code.activeFile = "src/a.py";
  code.edit("hand edit");
  await code.open("src/b.py");
  expect(code.documents["src/a.py"].text).toBe("hand edit");
  expect(code.dirty).toBe(true);
  code.activeFile = "src/a.py";
  vi.mocked(rpc).mockResolvedValueOnce({
    path: "src/a.py",
    text: "hand edit",
    hash: "new-hash",
  });
  await code.saveActive();
  expect(
    vi
      .mocked(rpc)
      .mock.calls.some(
        ([method, params]) =>
          method === "workspace.write" &&
          (params as any).expected_hash === "old-hash",
      ),
  ).toBe(true);
  expect(code.dirty).toBe(false);
});

it("a failed capture retains the previous valid graph and source", async () => {
  const code = useCode();
  const editor = useEditor();
  const previous = {
    success: true,
    analysis: { nodes: [{ id: "old" }], edges: [] },
    source_map: [],
  } as any;
  code.lastResult = previous;
  await code.acceptJob({
    id: "failure",
    project: "",
    state: "FAILED",
    error: "TraceError: dynamic control flow",
    result: null,
    logs: "",
    source_hash: "",
    file_path: "",
    artifact_dir: "",
  });
  expect(code.lastResult).toEqual(previous);
  expect(code.status).toBe("parse_failed");
  expect(code.error).toContain("control flow");
  expect(editor.graph.nodes).toEqual([]);
});

it("handwritten source is not eligible for full regeneration", async () => {
  const code = useCode();
  const editor = useEditor();
  editor.projectPath = "/project";
  await code.loadWorkspace();
  code.files = ["src/model.py"];
  vi.mocked(rpc).mockImplementation(async (method) =>
    method === "code.generate"
      ? { text: "generated" }
      : { path: "src/model.py", text: "# hand written", hash: "external-hash" },
  );
  await expect(code.previewGeneration()).rejects.toThrow("hand-edited");
  expect(code.proposal).toBeNull();
});

it("node and source navigation preserve explicitly unmapped locations", async () => {
  const code = useCode();
  code.lastResult = {
    source_map: [
      {
        file_path: "src/a.py",
        start_line: 12,
        end_line: 13,
        module_path: "layer",
        graph_node_id: "a",
        status: "exact",
        reason: "",
      },
      {
        file_path: "src/a.py",
        start_line: null,
        end_line: null,
        module_path: "",
        graph_node_id: "unknown",
        status: "unmapped",
        reason: "",
      },
    ],
  } as any;
  await code.locateNode("a");
  expect(code.jump?.line).toBe(12);
  await code.locateNode("unknown");
  expect(code.jump?.line).toBe(12);
  const previousJump = code.jump;
  code.locateLine(13);
  expect(code.selectedAnalysis).toBe("a");
  expect(code.jump).toEqual(previousJump);
});

it("a disk conflict retains the reviewed diff and unsaved source", async () => {
  const code = useCode();
  code.documents = {
    "src/model.py": {
      path: "src/model.py",
      text: "old",
      saved: "old",
      hash: "old-hash",
      revision: 0,
    },
  };
  code.proposal = {
    path: "src/model.py",
    before: "old",
    text: "new",
    expected_hash: "old-hash",
    diff: "",
    kind: "patch",
  };
  vi.mocked(rpc).mockRejectedValueOnce(
    new Error("source conflict: file changed on disk"),
  );
  await expect(code.applyProposal()).rejects.toThrow("source conflict");
  expect(code.status).toBe("conflict");
  expect(code.documents["src/model.py"].text).toBe("old");
  expect(code.proposal?.text).toBe("new");
});
