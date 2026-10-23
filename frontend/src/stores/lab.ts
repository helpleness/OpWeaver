import { computed, ref, watch } from "vue";
import { defineStore } from "pinia";
import { rpc } from "../api";
import { useEditor } from "./editor";
import { useModels } from "./models";
import type { OnnxReport } from "../model-types";
import {
  defaultConfig,
  type BackendCapability,
  type Experiment,
  type LabJob,
} from "../lab-types";

const terminal = (state: string) =>
  ["COMPLETED", "FAILED", "CANCELLED"].includes(state);
export const useLab = defineStore("lab", () => {
  const editor = useEditor();
  const models = useModels();
  const backends = ref<BackendCapability[]>([]);
  const records = ref<LabJob[]>([]);
  const source = ref<OnnxReport | null>(null);
  const config = ref(defaultConfig());
  const job = ref<LabJob | null>(null);
  const experiment = ref<Experiment | null>(null);
  const error = ref("");
  const loading = ref(false);
  const starting = ref(false);
  const exportResult = ref("");
  const running = computed(
    () => starting.value || (!!job.value && !terminal(job.value.state)),
  );
  let epoch = 0,
    selection = 0;
  let cancelPendingStart = false;
  const handled = new Set<string>();
  const key = () => `opweaver.lab.${editor.projectPath}`;

  async function probe() {
    const current = epoch;
    const sourceId = source.value?.artifact.id;
    const found = await rpc<BackendCapability[]>("lab.backends", {
      project: editor.projectPath,
      source_model_id: sourceId,
    });
    if (current === epoch && sourceId === source.value?.artifact.id)
      backends.value = found;
  }
  async function refresh() {
    if (!editor.projectPath) return;
    const current = epoch;
    const next = await rpc<LabJob[]>("lab.history", {
      project: editor.projectPath,
    });
    if (current === epoch) records.value = next;
  }
  async function selectSource(id: string) {
    const current = epoch,
      request = ++selection;
    loading.value = true;
    try {
      const report = await rpc<OnnxReport>("model.load", {
        project: editor.projectPath,
        id,
      });
      if (current !== epoch || request !== selection) return;
      source.value = report;
      config.value.source_model_id = id;
      config.value.inputs.shapes = Object.fromEntries(
        report.inputs
          .filter((v) => v.shape?.every((d) => typeof d === "number"))
          .map((v) => [v.name, v.shape as number[]]),
      );
      config.value.benchmark.batch_input = null;
      config.value.benchmark.batch_size = null;
      await probe();
    } finally {
      if (current === epoch && request === selection) loading.value = false;
    }
  }
  async function open(id: string) {
    const current = epoch,
      request = ++selection;
    const next = await rpc<Experiment>("lab.load", {
      project: editor.projectPath,
      id,
    });
    if (current !== epoch || request !== selection) return;
    const report = await rpc<OnnxReport>("model.load", {
      project: editor.projectPath,
      id: next.source_model_id,
    });
    if (current !== epoch || request !== selection) return;
    source.value = report;
    config.value = structuredClone(next.config);
    experiment.value = next;
    exportResult.value = "";
    error.value = "";
    localStorage.setItem(key(), id);
    await probe();
  }
  async function accept(next: LabJob) {
    const normalize = (path: string) =>
      path.replaceAll("\\", "/").toLowerCase();
    if (
      normalize(next.project) !== normalize(editor.projectPath) ||
      handled.has(next.id)
    )
      return;
    job.value = next;
    if (!terminal(next.state)) return;
    handled.add(next.id);
    if (next.result) experiment.value = next.result;
    if (next.error) error.value = next.error;
    localStorage.setItem(key(), next.id);
    await Promise.all([refresh(), models.refresh()]);
  }
  async function start(mode: "baseline" | "optimize") {
    if (running.value) return;
    if (!editor.projectPath || !source.value)
      throw new Error("Select an imported ONNX artifact first");
    error.value = "";
    exportResult.value = "";
    const current = epoch;
    starting.value = true;
    experiment.value = null;
    job.value = null;
    cancelPendingStart = false;
    try {
      config.value.mode = mode;
      const next = await rpc<LabJob>("lab.start", {
        project: editor.projectPath,
        config: config.value,
      });
      if (current === epoch) {
        if (cancelPendingStart && !terminal(next.state))
          await accept(await rpc<LabJob>("lab.stop", { id: next.id }));
        else await accept(next);
      }
    } finally {
      if (current === epoch) starting.value = false;
    }
  }
  async function poll() {
    if (!job.value || terminal(job.value.state)) return;
    await accept(await rpc<LabJob>("lab.status", { id: job.value.id }));
  }
  async function stop() {
    if (starting.value) cancelPendingStart = true;
    if (job.value)
      await accept(await rpc<LabJob>("lab.stop", { id: job.value.id }));
  }
  function workerExited() {
    if (job.value && running.value) {
      job.value.state = "FAILED";
      job.value.status = "Failed";
      job.value.error = error.value =
        "Experiment worker exited; reopen history after reconnecting to recover its report";
    }
    starting.value = false;
    cancelPendingStart = false;
  }
  async function exportReport(directory: string) {
    if (!experiment.value || !directory) return;
    const result = await rpc<{ directory: string }>("lab.export", {
      project: editor.projectPath,
      id: experiment.value.id,
      directory,
    });
    exportResult.value = result.directory;
  }
  async function loadWorkspace() {
    const current = ++epoch;
    ++selection;
    records.value = [];
    source.value = null;
    job.value = null;
    experiment.value = null;
    error.value = "";
    exportResult.value = "";
    loading.value = false;
    starting.value = false;
    config.value = defaultConfig();
    handled.clear();
    if (!editor.projectPath) return;
    loading.value = true;
    try {
      await refresh();
      if (current !== epoch) return;
      const previous = localStorage.getItem(key());
      if (previous && records.value.some((r) => r.id === previous))
        await open(previous);
    } finally {
      if (current === epoch) loading.value = false;
    }
  }
  watch(
    () => editor.projectPath,
    () => void loadWorkspace().catch((e) => (error.value = String(e))),
  );
  return {
    backends,
    records,
    source,
    config,
    job,
    experiment,
    error,
    loading,
    exportResult,
    running,
    probe,
    refresh,
    selectSource,
    open,
    accept,
    start,
    poll,
    stop,
    workerExited,
    exportReport,
    loadWorkspace,
  };
});
