import { computed, ref, watch } from "vue";
import { defineStore } from "pinia";
import { rpc } from "../api";
import type {
  Graph,
  Layout,
  Operator,
  Validation,
  TrainingConfig,
  Job,
} from "../types";

const clone = <T>(value: T): T => JSON.parse(JSON.stringify(value));
export const emptyGraph = (name = "Untitled model"): Graph => ({
  ir_version: "1.0",
  id: crypto.randomUUID(),
  metadata: { name, model_version: "0.1.0", description: "" },
  nodes: [],
  edges: [],
  modules: [],
  plugins: [],
});

export const useEditor = defineStore("editor", () => {
  const graph = ref<Graph>(emptyGraph());
  const layout = ref<Layout>({ version: 1, positions: {} });
  const registry = ref<Operator[]>([]);
  const selected = ref<string | null>(null);
  const selection = ref<string[]>([]);
  const projectPath = ref("");
  const savedSignature = ref("");
  const past = ref<{ graph: Graph; layout: Layout }[]>([]);
  const future = ref<{ graph: Graph; layout: Layout }[]>([]);
  const validation = ref<Validation>({
    valid: false,
    issues: [],
    order: [],
    tensors: {},
  });
  const validating = ref(false);
  const validationError = ref("");
  const job = ref<Job | null>(null);
  const history = ref<Job[]>([]);
  const config = ref<TrainingConfig>({
    epochs: 20,
    batch_size: 32,
    learning_rate: 0.01,
    optimizer: "AdamW",
    loss: "CrossEntropyLoss",
    samples: 512,
    seed: 42,
    device: "cpu",
  });
  const signature = computed(() =>
    JSON.stringify({ graph: graph.value, layout: layout.value }),
  );
  const dirty = computed(() => signature.value !== savedSignature.value);
  const running = computed(
    () =>
      !!job.value &&
      !["COMPLETED", "FAILED", "CANCELLED"].includes(job.value.state),
  );
  const selectedNode = computed(() =>
    graph.value.nodes?.find((n) => n.id === selected.value),
  );
  const selectedOperator = computed(() =>
    registry.value.find((op) => op.id === selectedNode.value?.operator.id),
  );
  let revision = 0;
  let timer: ReturnType<typeof setTimeout>;
  function remember() {
    past.value.push({ graph: clone(graph.value), layout: clone(layout.value) });
    if (past.value.length > 80) past.value.shift();
    future.value = [];
  }
  function undo() {
    const state = past.value.pop();
    if (!state) return;
    future.value.push({
      graph: clone(graph.value),
      layout: clone(layout.value),
    });
    graph.value = state.graph;
    layout.value = state.layout;
  }
  function redo() {
    const state = future.value.pop();
    if (!state) return;
    past.value.push({ graph: clone(graph.value), layout: clone(layout.value) });
    graph.value = state.graph;
    layout.value = state.layout;
  }
  function addNode(operatorId: string, position = { x: 100, y: 100 }) {
    const op = registry.value.find((op) => op.id === operatorId);
    if (!op) throw new Error(`Unknown operator: ${operatorId}`);
    remember();
    const id = `${op.label.toLowerCase()}_${crypto.randomUUID().slice(0, 8)}`;
    graph.value.nodes ??= [];
    graph.value.nodes.push({
      id,
      label: op.label,
      operator: { id: op.id, version: op.version },
      params: clone(op.defaults),
      inputs: op.inputs.map((name) => ({ name })),
      outputs: op.outputs.map((name) => ({ name })),
    });
    layout.value.positions[id] = position;
    selected.value = id;
    selection.value = [id];
    return id;
  }
  function removeNodes(ids: string[]) {
    if (!ids.some((id) => graph.value.nodes?.some((n) => n.id === id))) return;
    remember();
    graph.value.nodes = graph.value.nodes?.filter((n) => !ids.includes(n.id));
    graph.value.edges = graph.value.edges?.filter(
      (e) => !ids.includes(e.source) && !ids.includes(e.target),
    );
    ids.forEach((id) => delete layout.value.positions[id]);
    selection.value = selection.value.filter((id) => !ids.includes(id));
    if (selected.value && ids.includes(selected.value)) selected.value = null;
  }
  function updateParams(id: string, key: string, value: unknown) {
    const node = graph.value.nodes?.find((n) => n.id === id);
    if (!node) return;
    remember();
    node.params = { ...node.params, [key]: value };
  }
  function renameNode(id: string, label: string) {
    const node = graph.value.nodes?.find((n) => n.id === id);
    if (node) {
      remember();
      node.label = label;
    }
  }
  function moveNodes(positions: Record<string, { x: number; y: number }>) {
    const changed = Object.entries(positions).some(
      ([id, position]) =>
        layout.value.positions[id]?.x !== position.x ||
        layout.value.positions[id]?.y !== position.y,
    );
    if (!changed) return;
    remember();
    Object.assign(layout.value.positions, positions);
  }
  function connect(
    source: string,
    target: string,
    sourcePort: string,
    targetPort: string,
  ) {
    const a = graph.value.nodes?.find((n) => n.id === source),
      b = graph.value.nodes?.find((n) => n.id === target);
    if (
      !a?.outputs?.some((p) => p.name === sourcePort) ||
      !b?.inputs?.some((p) => p.name === targetPort)
    )
      throw new Error("Choose an output port and an input port.");
    if (source === target) throw new Error("A node cannot connect to itself.");
    if (
      graph.value.edges?.some(
        (e) => e.target === target && e.target_port === targetPort,
      )
    )
      throw new Error("This input already has a connection. Delete it first.");
    const reachable = new Set([target]);
    let changed = true;
    while (changed) {
      const before = reachable.size;
      graph.value.edges?.forEach((e) => {
        if (reachable.has(e.source)) reachable.add(e.target);
      });
      changed = reachable.size !== before;
    }
    if (reachable.has(source))
      throw new Error("This connection would create a cycle.");
    remember();
    graph.value.edges ??= [];
    graph.value.edges.push({
      id: `edge_${crypto.randomUUID()}`,
      source,
      target,
      source_port: sourcePort,
      target_port: targetPort,
    });
  }
  function removeEdges(ids: string[]) {
    if (!graph.value.edges?.some((e) => ids.includes(e.id))) return;
    remember();
    graph.value.edges = graph.value.edges.filter((e) => !ids.includes(e.id));
  }
  function replace(next: Graph, nextLayout: Layout, path = "", saved = false) {
    graph.value = clone(next);
    layout.value = clone(nextLayout);
    projectPath.value = path;
    selected.value = null;
    selection.value = [];
    past.value = [];
    future.value = [];
    job.value = null;
    history.value = [];
    savedSignature.value = saved ? signature.value : "";
  }
  async function validateNow() {
    const current = ++revision;
    validating.value = true;
    try {
      const result = await rpc<Validation>("validate", { graph: graph.value });
      if (current === revision) {
        validation.value = result;
        validationError.value = "";
      }
      return result;
    } catch (error) {
      if (current === revision) {
        validationError.value = String(error);
        validation.value = {
          valid: false,
          issues: [{ severity: "error", message: String(error) }],
          order: [],
          tensors: {},
        };
      }
      throw error;
    } finally {
      if (current === revision) validating.value = false;
    }
  }
  watch(
    () => JSON.stringify(graph.value),
    () => {
      ++revision;
      validating.value = true;
      clearTimeout(timer);
      timer = setTimeout(() => {
        void validateNow().catch(() => {});
      }, 250);
    },
  );
  async function save() {
    if (!projectPath.value)
      throw new Error("Create a project directory before saving.");
    const snapshot = signature.value;
    await rpc("project.save", {
      path: projectPath.value,
      name: graph.value.metadata?.name,
      graph: graph.value,
      layout: layout.value,
    });
    savedSignature.value = snapshot;
  }
  async function load(path: string) {
    const snapshot = await rpc("project.load", { path });
    replace(snapshot.graph, snapshot.layout, snapshot.path, true);
    await loadHistory();
    if (job.value) config.value = { ...job.value.config };
  }
  async function loadHistory() {
    if (!projectPath.value) return;
    history.value = await rpc("train.history", { project: projectPath.value });
    if (!running.value && history.value.length) job.value = history.value[0];
  }
  return {
    graph,
    layout,
    registry,
    selected,
    selection,
    projectPath,
    savedSignature,
    signature,
    dirty,
    past,
    future,
    validation,
    validating,
    validationError,
    job,
    history,
    config,
    running,
    selectedNode,
    selectedOperator,
    remember,
    undo,
    redo,
    addNode,
    removeNodes,
    updateParams,
    renameNode,
    moveNodes,
    connect,
    removeEdges,
    replace,
    validateNow,
    save,
    load,
    loadHistory,
  };
});
