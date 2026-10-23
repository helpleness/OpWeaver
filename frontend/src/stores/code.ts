import { computed, ref, watch } from "vue";
import { defineStore } from "pinia";
import { rpc } from "../api";
import { useEditor } from "./editor";
import type {
  CaptureJob,
  CaptureResult,
  CodeProposal,
  Diagnostic,
  SourceFile,
  SourceMapping,
  SyncState,
} from "../code-types";

interface DocumentState extends SourceFile {
  saved: string;
  revision: number;
}
const clone = <T>(value: T): T => JSON.parse(JSON.stringify(value));

export const useCode = defineStore("code", () => {
  const editor = useEditor();
  const mode = ref<"visual" | "code-first">("visual");
  const view = ref<"graph" | "split" | "code">("graph");
  const files = ref<string[]>([]);
  const documents = ref<Record<string, DocumentState>>({});
  const activeFile = ref("");
  const entry = ref("OpWeaverModel");
  const entries = ref<
    { name: string; kind: string; line: number; signature: string }[]
  >([]);
  const constructor = ref("{}");
  const inputConfig = ref(
    '[{"shape":[4,8],"design_shape":["B",8],"dtype":"float32"}]',
  );
  const status = ref<SyncState>("pending");
  const diagnostics = ref<Diagnostic[]>([]);
  const lastResult = ref<CaptureResult | null>(null);
  const captureJob = ref<CaptureJob | null>(null);
  const parsedHash = ref("");
  const codeRevision = ref(0);
  const graphRevision = ref(0);
  const parsedGraphSignature = ref("");
  const managedHashes = ref<Record<string, string>>({});
  const proposal = ref<CodeProposal | null>(null);
  const selectedAnalysis = ref<string | null>(null);
  const analysisSelection = ref<string[]>([]);
  const jump = ref<{ line: number; endLine?: number; token: number } | null>(
    null,
  );
  const error = ref("");
  const stateHash = ref("");
  const lastStateText = ref("");
  let applyingGraph = false;
  let locatingFromSource = false;
  let projectEpoch = 0;
  let staticTimer: ReturnType<typeof setTimeout>;
  let persistQueue: Promise<void> = Promise.resolve();
  const handledTerminal = new Set<string>();

  const document = computed(() => documents.value[activeFile.value]);
  const text = computed(() => document.value?.text ?? "");
  const dirty = computed(() =>
    Object.values(documents.value).some((d) => d.text !== d.saved),
  );
  const activeAnalysis = computed(() =>
    mode.value === "code-first" ? (lastResult.value?.analysis ?? null) : null,
  );
  const readOnly = computed(() => mode.value === "code-first");
  const running = computed(
    () =>
      !!captureJob.value &&
      ["CREATED", "RUNNING", "STOPPING"].includes(captureJob.value.state),
  );
  const sourceMap = computed(() => lastResult.value?.source_map ?? []);
  const syncLabel = computed(
    () =>
      ({
        synced: "Synced",
        pending: "Waiting to parse",
        parsing: "Capturing…",
        parse_failed: "Parse failed · graph stale",
        stale: "Graph stale",
        conflict: "Conflict · reparse required",
        read_only: "Read-only analysis",
      })[status.value],
  );

  async function inspect() {
    const current = activeFile.value;
    const content = text.value;
    const result = await rpc("code.inspect", {
      source: content,
      file_path: current,
    });
    if (current !== activeFile.value || content !== text.value) return;
    entries.value = result.entries;
    diagnostics.value = result.diagnostics;
    if (
      !entries.value.some((e) => e.name === entry.value) &&
      entries.value.length
    )
      entry.value = (
        entries.value.find((e) => e.kind === "class") ?? entries.value[0]
      ).name;
  }
  function edit(content: string) {
    if (!document.value || content === document.value.text) return;
    document.value.text = content;
    document.value.revision++;
    codeRevision.value++;
    status.value =
      mode.value === "visual" &&
      parsedGraphSignature.value &&
      parsedGraphSignature.value !== JSON.stringify(editor.graph)
        ? "conflict"
        : lastResult.value
          ? "stale"
          : "pending";
    proposal.value = null;
    clearTimeout(staticTimer);
    staticTimer = setTimeout(
      () => void inspect().catch((e) => (error.value = String(e))),
      300,
    );
  }
  async function refreshFiles() {
    if (editor.projectPath)
      files.value = await rpc("workspace.list", {
        project: editor.projectPath,
      });
  }
  async function open(path: string) {
    if (!documents.value[path]) {
      const file = await rpc<SourceFile>("workspace.read", {
        project: editor.projectPath,
        path,
      });
      documents.value[path] = { ...file, saved: file.text, revision: 0 };
    }
    activeFile.value = path;
    if (
      parsedHash.value !== document.value.hash ||
      document.value.text !== document.value.saved
    )
      status.value = lastResult.value ? "stale" : "pending";
    await inspect();
  }
  async function saveActive() {
    if (!document.value) return;
    const path = activeFile.value,
      content = text.value;
    try {
      const saved = await rpc<SourceFile>("workspace.write", {
        project: editor.projectPath,
        path,
        text: content,
        expected_hash: document.value.hash,
      });
      documents.value[path].hash = saved.hash;
      documents.value[path].saved = saved.text;
      error.value = "";
      await persist();
    } catch (e) {
      status.value = "conflict";
      error.value = String(e);
      throw e;
    }
  }
  async function saveAll() {
    const current = activeFile.value;
    for (const path of Object.keys(documents.value))
      if (documents.value[path].text !== documents.value[path].saved) {
        activeFile.value = path;
        await saveActive();
      }
    activeFile.value = current;
  }
  async function importFile(source: string) {
    const file = await rpc<SourceFile>("workspace.import", {
      project: editor.projectPath,
      source,
    });
    documents.value[file.path] = { ...file, saved: file.text, revision: 0 };
    await refreshFiles();
    mode.value = "code-first";
    view.value = "split";
    await open(file.path);
    await persist();
  }
  async function newFile(path: string) {
    const file = await rpc<SourceFile>("workspace.write", {
      project: editor.projectPath,
      path,
      text: "# Python model source\nfrom torch import nn\n",
      expected_hash: "",
    });
    documents.value[file.path] = { ...file, saved: file.text, revision: 0 };
    await refreshFiles();
    await open(file.path);
  }
  async function reloadActive() {
    if (!activeFile.value) return;
    const saved = await rpc<SourceFile>("workspace.read", {
      project: editor.projectPath,
      path: activeFile.value,
    });
    documents.value[activeFile.value] = {
      ...saved,
      saved: saved.text,
      revision: (document.value?.revision ?? 0) + 1,
    };
    status.value = lastResult.value ? "stale" : "pending";
    error.value = "";
    proposal.value = null;
    await inspect();
  }
  async function previewGeneration(path = "src/model.py") {
    if (!editor.projectPath)
      throw new Error("Create or open a project before generating Python.");
    const generated = await rpc("code.generate", {
      graph: editor.graph,
      file_path: path,
      class_name: "OpWeaverModel",
    });
    let before = "",
      hash = "";
    if (files.value.includes(path)) {
      const current = await rpc<SourceFile>("workspace.read", {
        project: editor.projectPath,
        path,
      });
      before = current.text;
      hash = current.hash;
      if (
        documents.value[path] &&
        documents.value[path].text !== documents.value[path].saved
      )
        throw new Error(
          "Save your source edits before preparing a graph change.",
        );
      if (managedHashes.value[path] !== current.hash)
        throw new Error(
          "This file contains hand-edited code. Use a new output filename, or Code-First with a local parameter patch. Existing source is preserved.",
        );
    }
    proposal.value = {
      path,
      before,
      text: generated.text,
      expected_hash: hash,
      diff: "",
      kind: "generate",
      class_name: generated.class_name,
      source_map: generated.source_map,
      graph_snapshot: JSON.stringify(editor.graph),
    };
  }
  async function previewParameter(nodeId: string, key: string, value: unknown) {
    if (
      !document.value ||
      !lastResult.value ||
      document.value.hash !== parsedHash.value ||
      document.value.text !== document.value.saved ||
      status.value === "conflict" ||
      status.value === "stale"
    )
      throw new Error(
        "Save and reparse this file before preparing a source edit.",
      );
    const mapping = sourceMap.value.find(
      (m) =>
        m.graph_node_id === nodeId && m.status === "exact" && m.module_path,
    );
    if (!mapping)
      throw new Error(
        "This node has no exact, patchable module definition. Its graph remains read-only.",
      );
    const result = await rpc("code.patch", {
      source: text.value,
      class_name: entry.value,
      module_path: mapping.module_path,
      key,
      value,
      expected_hash: parsedHash.value,
      operator_id: lastResult.value.analysis.nodes.find((n) => n.id === nodeId)
        ?.operator,
      parameters: lastResult.value.analysis.nodes.find((n) => n.id === nodeId)
        ?.parameters,
    });
    proposal.value = {
      path: activeFile.value,
      before: text.value,
      text: result.text,
      expected_hash: document.value.hash,
      diff: result.diff,
      kind: "patch",
    };
  }
  async function applyProposal() {
    const candidate = proposal.value;
    if (!candidate) return;
    if (
      candidate.kind === "generate" &&
      candidate.graph_snapshot !== JSON.stringify(editor.graph)
    ) {
      status.value = "conflict";
      throw new Error(
        "The graph changed after this preview. Generate a fresh diff.",
      );
    }
    if (
      documents.value[candidate.path] &&
      documents.value[candidate.path].text !== candidate.before
    )
      throw new Error(
        "Source buffer changed after the diff was prepared. Review a fresh diff.",
      );
    let saved: SourceFile;
    try {
      saved = await rpc<SourceFile>("workspace.write", {
        project: editor.projectPath,
        path: candidate.path,
        text: candidate.text,
        expected_hash: candidate.expected_hash,
      });
    } catch (e) {
      status.value = "conflict";
      error.value = String(e);
      throw e;
    }
    documents.value[saved.path] = {
      ...saved,
      saved: saved.text,
      revision: (documents.value[saved.path]?.revision ?? 0) + 1,
    };
    if (candidate.kind === "generate") {
      managedHashes.value[saved.path] = saved.hash;
      entry.value = candidate.class_name!;
      mode.value = "visual";
      parsedGraphSignature.value = JSON.stringify(editor.graph);
    }
    proposal.value = null;
    activeFile.value = saved.path;
    parsedHash.value = "";
    status.value = "pending";
    codeRevision.value++;
    await refreshFiles();
    await inspect();
    await persist();
  }
  async function startCapture() {
    if (!document.value)
      throw new Error("Open or generate a Python file first.");
    if (
      mode.value === "visual" &&
      parsedGraphSignature.value &&
      parsedGraphSignature.value !== JSON.stringify(editor.graph)
    ) {
      status.value = "conflict";
      throw new Error(
        "The Design Graph changed since the last synchronization. Generate and review Graph → Code, or explicitly switch to Code-First before parsing source.",
      );
    }
    if (dirty.value) await saveAll();
    const constructorValue = JSON.parse(constructor.value);
    const inputs = JSON.parse(inputConfig.value);
    if (
      !Array.isArray(inputs) ||
      !constructorValue ||
      Array.isArray(constructorValue) ||
      typeof constructorValue !== "object"
    )
      throw new Error(
        "Constructor must be a JSON object; inputs must be a JSON array.",
      );
    let result: CaptureJob;
    try {
      result = await rpc<CaptureJob>("code.capture.start", {
        project: editor.projectPath,
        path: activeFile.value,
        entry: entry.value,
        constructor: constructorValue,
        inputs,
        expected_hash: document.value.hash,
        execute_confirmed: true,
        timeout: 30,
      });
    } catch (e) {
      if (/conflict|changed|hash/i.test(String(e))) status.value = "conflict";
      error.value = String(e);
      throw e;
    }
    captureJob.value = result;
    status.value = "parsing";
    error.value = "";
    await persist();
  }
  async function acceptJob(job: CaptureJob) {
    const normalize = (path: string) =>
      path.replaceAll("\\", "/").toLowerCase();
    if (normalize(job.project) !== normalize(editor.projectPath)) return;
    if (handledTerminal.has(job.id)) return;
    if (captureJob.value && captureJob.value.id !== job.id && running.value)
      return;
    captureJob.value = job;
    if (!["COMPLETED", "FAILED", "CANCELLED"].includes(job.state)) return;
    handledTerminal.add(job.id);
    if (job.state !== "COMPLETED" || !job.result?.success) {
      status.value = "parse_failed";
      error.value = job.error ?? "Capture cancelled; previous graph retained.";
      diagnostics.value = job.result?.diagnostics ?? [
        { severity: "error", message: error.value },
      ];
      await persist();
      return;
    }
    if (
      job.source_hash !== document.value?.hash ||
      document.value.text !== document.value.saved
    ) {
      status.value = "stale";
      error.value =
        "Source changed during capture. The previous graph is retained; parse the current saved file.";
      return;
    }
    lastResult.value = job.result;
    diagnostics.value = job.result.diagnostics;
    parsedHash.value = job.result.source_hash;
    selectedAnalysis.value = null;
    if (mode.value === "visual" && job.result.design_graph) {
      applyingGraph = true;
      editor.remember();
      const projected = clone(job.result.design_graph);
      projected.id = editor.graph.id;
      projected.metadata = editor.graph.metadata;
      editor.graph = projected;
      (projected.nodes ?? []).forEach(
        (n, i) =>
          (editor.layout.positions[n.id] ??= {
            x: 60 + (i % 4) * 245,
            y: 80 + Math.floor(i / 4) * 200,
          }),
      );
      applyingGraph = false;
      await editor.validateNow();
      try {
        await editor.save();
        status.value = "synced";
      } catch (e) {
        status.value = "stale";
        error.value = `Captured graph could not be saved: ${String(e)}`;
      }
    } else if (mode.value === "visual") {
      mode.value = "code-first";
      status.value = "read_only";
    } else status.value = "read_only";
    parsedGraphSignature.value = JSON.stringify(editor.graph);
    graphRevision.value++;
    await persist();
  }
  async function pollCapture() {
    if (running.value && captureJob.value) {
      const job = await rpc<CaptureJob>("code.capture.status", {
        id: captureJob.value.id,
      });
      if (
        job.state !== captureJob.value.state ||
        ["COMPLETED", "FAILED", "CANCELLED"].includes(job.state)
      )
        await acceptJob(job);
    }
  }
  async function stopCapture() {
    if (captureJob.value)
      captureJob.value = await rpc("code.capture.stop", {
        id: captureJob.value.id,
      });
  }
  async function locateNode(id: string) {
    const map = sourceMap.value.find(
      (m) => m.graph_node_id === id && m.start_line,
    );
    if (!map) return;
    const normalize = (path: string) => path.replaceAll("\\", "/");
    const root = normalize(editor.projectPath) + "/";
    const mapped = normalize(map.file_path);
    const relative = mapped.toLowerCase().startsWith(root.toLowerCase())
      ? mapped.slice(root.length)
      : mapped;
    if (relative !== activeFile.value && files.value.includes(relative))
      await open(relative);
    else if (
      activeFile.value &&
      relative !== activeFile.value &&
      editor.projectPath
    )
      return;
    selectedAnalysis.value = id;
    analysisSelection.value = [id];
    jump.value = {
      line: map.start_line!,
      endLine: map.end_line ?? undefined,
      token: Date.now(),
    };
  }
  function locateLine(line: number) {
    const maps = sourceMap.value.filter(
      (m) =>
        m.start_line &&
        line >= m.start_line &&
        line <= (m.end_line ?? m.start_line),
    );
    if (maps.length) {
      selectedAnalysis.value = maps[0].graph_node_id;
      analysisSelection.value = maps.map((m) => m.graph_node_id);
      if (mode.value === "visual") {
        locatingFromSource = true;
        editor.selected = maps[0].graph_node_id;
        editor.selection = maps.map((m) => m.graph_node_id);
        locatingFromSource = false;
      }
    }
  }
  async function persistCurrent() {
    if (!editor.projectPath) return;
    const project = editor.projectPath,
      epoch = projectEpoch;
    const payload = {
      version: 1,
      mode: mode.value,
      active_file: activeFile.value,
      entry: entry.value,
      constructor: constructor.value,
      inputs: inputConfig.value,
      status: status.value,
      last_result: lastResult.value,
      parsed_hash: parsedHash.value,
      code_revision: codeRevision.value,
      graph_revision: graphRevision.value,
      parsed_graph: parsedGraphSignature.value,
      managed_hashes: managedHashes.value,
    };
    const encoded = JSON.stringify(payload);
    if (encoded === lastStateText.value) return;
    try {
      const file = await rpc<SourceFile>("workspace.write", {
        project: editor.projectPath,
        path: ".opweaver/code-state.json",
        text: encoded,
        expected_hash: stateHash.value,
      });
      if (epoch === projectEpoch && project === editor.projectPath) {
        stateHash.value = file.hash;
        lastStateText.value = encoded;
      }
    } catch (e) {
      error.value = `Workspace state was not saved: ${String(e)}`;
    }
  }
  function persist() {
    persistQueue = persistQueue.then(persistCurrent).catch((e) => {
      error.value = String(e);
    });
    return persistQueue;
  }
  async function loadWorkspace() {
    const epoch = ++projectEpoch;
    handledTerminal.clear();
    documents.value = {};
    files.value = [];
    activeFile.value = "";
    lastResult.value = null;
    captureJob.value = null;
    managedHashes.value = {};
    stateHash.value = "";
    lastStateText.value = "";
    parsedHash.value = "";
    parsedGraphSignature.value = "";
    codeRevision.value = 0;
    graphRevision.value = 0;
    diagnostics.value = [];
    proposal.value = null;
    status.value = "pending";
    error.value = "";
    mode.value = "visual";
    if (!editor.projectPath) return;
    await refreshFiles();
    try {
      const file = await rpc<SourceFile>("workspace.read", {
        project: editor.projectPath,
        path: ".opweaver/code-state.json",
      });
      if (epoch !== projectEpoch) return;
      const saved = JSON.parse(file.text);
      if (saved.version !== 1)
        throw new Error("Unknown code workspace version");
      stateHash.value = file.hash;
      lastStateText.value = file.text;
      mode.value = saved.mode;
      entry.value = saved.entry;
      constructor.value = saved.constructor;
      inputConfig.value = saved.inputs;
      status.value = saved.status;
      lastResult.value = saved.last_result;
      parsedHash.value = saved.parsed_hash;
      codeRevision.value = saved.code_revision;
      graphRevision.value = saved.graph_revision;
      parsedGraphSignature.value = saved.parsed_graph;
      managedHashes.value = saved.managed_hashes;
      if (files.value.includes(saved.active_file))
        await open(saved.active_file);
    } catch {
      if (files.value.length) await open(files.value[0]);
    }
    if (status.value === "parsing") status.value = "stale";
    if (
      mode.value === "visual" &&
      status.value === "synced" &&
      parsedGraphSignature.value !== JSON.stringify(editor.graph)
    )
      status.value = "stale";
  }
  watch(
    () => editor.projectPath,
    () => void loadWorkspace().catch((e) => (error.value = String(e))),
  );
  watch(
    () => JSON.stringify(editor.graph),
    () => {
      if (
        !applyingGraph &&
        parsedGraphSignature.value &&
        JSON.stringify(editor.graph) !== parsedGraphSignature.value
      ) {
        graphRevision.value++;
        if (mode.value === "visual") status.value = "pending";
      }
    },
    { flush: "sync" },
  );
  watch(
    () => editor.selected,
    (id) => {
      if (id && !locatingFromSource)
        void locateNode(id).catch((e) => (error.value = String(e)));
    },
    { flush: "sync" },
  );
  return {
    mode,
    view,
    files,
    documents,
    activeFile,
    entry,
    entries,
    constructor,
    inputConfig,
    status,
    diagnostics,
    lastResult,
    captureJob,
    parsedHash,
    codeRevision,
    graphRevision,
    proposal,
    selectedAnalysis,
    analysisSelection,
    jump,
    error,
    document,
    text,
    dirty,
    activeAnalysis,
    readOnly,
    running,
    sourceMap,
    syncLabel,
    inspect,
    edit,
    refreshFiles,
    open,
    saveActive,
    saveAll,
    importFile,
    newFile,
    reloadActive,
    previewGeneration,
    previewParameter,
    applyProposal,
    startCapture,
    acceptJob,
    pollCapture,
    stopCapture,
    locateNode,
    locateLine,
    persist,
    loadWorkspace,
  };
});
