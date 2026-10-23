import { computed, ref, watch } from "vue";
import { defineStore } from "pinia";
import { rpc } from "../api";
import { useEditor } from "./editor";
import type { ModelJob, OnnxReport } from "../model-types";

export const useModels = defineStore("models", () => {
  const editor = useEditor();
  const records = ref<ModelJob[]>([]);
  const report = ref<OnnxReport | null>(null);
  const job = ref<ModelJob | null>(null);
  const selectedNode = ref<string | null>(null);
  const subgraphKey = ref("");
  const error = ref("");
  const loading = ref(false);
  const running = computed(
    () =>
      !!job.value &&
      ["CREATED", "RUNNING", "STOPPING"].includes(job.value.state),
  );
  const completed = computed(() =>
    records.value.filter((r) => r.state === "COMPLETED" && r.artifact),
  );
  let epoch = 0,
    selectionRequest = 0;
  const handled = new Set<string>();
  const storageKey = () => `opweaver.model.${editor.projectPath}`;
  async function refresh() {
    if (!editor.projectPath) return;
    const current = epoch;
    const next = await rpc<ModelJob[]>("model.list", {
      project: editor.projectPath,
    });
    if (current === epoch) records.value = next;
  }
  async function open(id: string) {
    const current = epoch,
      request = ++selectionRequest;
    loading.value = true;
    try {
      const next = await rpc<OnnxReport>("model.load", {
        project: editor.projectPath,
        id,
      });
      if (current !== epoch || request !== selectionRequest) return;
      report.value = next;
      selectedNode.value = null;
      subgraphKey.value = "";
      error.value = "";
      localStorage.setItem(storageKey(), id);
    } catch (e) {
      if (current === epoch && request === selectionRequest)
        error.value = String(e);
    } finally {
      if (current === epoch && request === selectionRequest)
        loading.value = false;
    }
  }
  async function loadWorkspace() {
    const current = ++epoch;
    records.value = [];
    report.value = null;
    job.value = null;
    selectedNode.value = null;
    error.value = "";
    loading.value = false;
    handled.clear();
    if (!editor.projectPath) return;
    await refresh();
    if (current !== epoch) return;
    const previous = localStorage.getItem(storageKey());
    const selected =
      completed.value.find((r) => r.id === previous) ?? completed.value[0];
    if (selected) await open(selected.id);
  }
  async function start(source: string, options: Record<string, unknown> = {}) {
    if (!editor.projectPath)
      throw new Error("Create or open a project before importing ONNX.");
    const current = epoch;
    error.value = "";
    const next = await rpc<ModelJob>("model.import.start", {
      ...options,
      project: editor.projectPath,
      source,
      timeout: 120,
    });
    if (current !== epoch) return;
    job.value = next;
  }
  async function accept(next: ModelJob) {
    const normalize = (path: string) =>
      path.replaceAll("\\", "/").toLowerCase();
    if (normalize(next.project) !== normalize(editor.projectPath)) return;
    if (handled.has(next.id)) return;
    job.value = next;
    if (!["COMPLETED", "FAILED", "CANCELLED"].includes(next.state)) return;
    handled.add(next.id);
    if (next.state === "COMPLETED" && next.result?.success) {
      ++selectionRequest;
      report.value = next.result;
      selectedNode.value = null;
      subgraphKey.value = "";
      loading.value = false;
      error.value = "";
      localStorage.setItem(storageKey(), next.id);
    } else
      error.value =
        next.error ?? "Import cancelled. Previous valid model retained.";
    await refresh();
  }
  async function poll() {
    if (running.value && job.value)
      await accept(
        await rpc<ModelJob>("model.import.status", { id: job.value.id }),
      );
  }
  async function stop() {
    if (job.value)
      await accept(
        await rpc<ModelJob>("model.import.stop", { id: job.value.id }),
      );
  }
  watch(
    () => editor.projectPath,
    () => void loadWorkspace().catch((e) => (error.value = String(e))),
  );
  return {
    records,
    report,
    job,
    selectedNode,
    subgraphKey,
    error,
    loading,
    running,
    completed,
    refresh,
    open,
    start,
    accept,
    poll,
    stop,
    loadWorkspace,
  };
});
