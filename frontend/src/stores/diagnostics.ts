import { computed, ref, watch } from "vue";
import { defineStore } from "pinia";
import { rpc } from "../api";
import { useEditor } from "./editor";
import {
  diagnosticDefaults,
  type DiagnosticConfig,
  type DiagnosticJob,
  type DiagnosticReport,
} from "../diagnostic-types";

const terminal = (state: string) =>
  ["COMPLETED", "FAILED", "CANCELLED"].includes(state);
export const useDiagnostics = defineStore("diagnostics", () => {
  const editor = useEditor();
  const config = ref(diagnosticDefaults()),
    records = ref<DiagnosticJob[]>([]);
  const report = ref<DiagnosticReport | null>(null),
    job = ref<DiagnosticJob | null>(null);
  const error = ref(""),
    exported = ref(""),
    starting = ref(false),
    showPanel = ref(false);
  const running = computed(
    () => starting.value || (!!job.value && !terminal(job.value.state)),
  );
  const handled = new Set<string>();
  let epoch = 0,
    selection = 0,
    cancelPending = false;
  const key = () => `opweaver.diagnostic.${editor.projectPath}`;
  async function refresh() {
    if (!editor.projectPath) return;
    const current = epoch,
      next = await rpc<DiagnosticJob[]>("diagnostic.history", {
        project: editor.projectPath,
      });
    if (current === epoch) records.value = next;
  }
  async function open(id: string) {
    const current = epoch,
      request = ++selection;
    const next = await rpc<DiagnosticReport>("diagnostic.load", {
      project: editor.projectPath,
      id,
    });
    if (current !== epoch || request !== selection) return;
    report.value = next;
    config.value = structuredClone(next.config);
    error.value = "";
    exported.value = "";
    localStorage.setItem(key(), id);
  }
  async function accept(next: DiagnosticJob) {
    const normalize = (p: string) => p.replaceAll("\\", "/").toLowerCase();
    if (
      normalize(next.project) !== normalize(editor.projectPath) ||
      handled.has(next.id)
    )
      return;
    job.value = next;
    if (!terminal(next.state)) return;
    handled.add(next.id);
    if (next.result) {
      report.value = next.result;
      config.value = structuredClone(next.result.config);
    }
    if (next.error) error.value = next.error;
    localStorage.setItem(key(), next.id);
    await refresh();
  }
  async function start(nextConfig: DiagnosticConfig) {
    if (running.value) return;
    const current = epoch;
    starting.value = true;
    cancelPending = false;
    report.value = null;
    job.value = null;
    error.value = "";
    exported.value = "";
    try {
      const next = await rpc<DiagnosticJob>("diagnostic.start", {
        project: editor.projectPath,
        config: nextConfig,
      });
      if (current === epoch) {
        if (cancelPending && !terminal(next.state))
          await accept(
            await rpc<DiagnosticJob>("diagnostic.stop", { id: next.id }),
          );
        else await accept(next);
      }
    } finally {
      if (current === epoch) starting.value = false;
    }
  }
  async function poll() {
    if (job.value && !terminal(job.value.state))
      await accept(
        await rpc<DiagnosticJob>("diagnostic.status", { id: job.value.id }),
      );
  }
  async function stop() {
    cancelPending = true;
    if (job.value)
      await accept(
        await rpc<DiagnosticJob>("diagnostic.stop", { id: job.value.id }),
      );
  }
  function workerExited() {
    if (job.value && running.value) {
      job.value.state = "FAILED";
      job.value.status = "Failed";
      job.value.error = error.value =
        "Diagnostic worker exited; reopen its durable history after reconnecting";
    }
    starting.value = false;
    cancelPending = false;
  }
  async function exportReport(directory: string) {
    if (!report.value) return;
    exported.value = (
      await rpc<{ directory: string }>("diagnostic.export", {
        project: editor.projectPath,
        id: report.value.id,
        directory,
      })
    ).directory;
  }
  async function loadWorkspace() {
    const current = ++epoch;
    ++selection;
    config.value = diagnosticDefaults();
    records.value = [];
    report.value = null;
    job.value = null;
    error.value = "";
    exported.value = "";
    starting.value = false;
    handled.clear();
    showPanel.value = false;
    if (!editor.projectPath) return;
    await refresh();
    if (current !== epoch) return;
    const previous = localStorage.getItem(key());
    const latest =
      records.value.find((v) => v.id === previous) ?? records.value[0];
    if (latest) await open(latest.id);
  }
  watch(
    () => editor.projectPath,
    () => void loadWorkspace().catch((e) => (error.value = String(e))),
  );
  return {
    config,
    records,
    report,
    job,
    error,
    exported,
    running,
    showPanel,
    refresh,
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
