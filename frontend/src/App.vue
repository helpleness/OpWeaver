<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from "vue";
import {
  Activity,
  ArrowRight,
  Box,
  Braces,
  Check,
  ChevronDown,
  ChevronRight,
  CircleHelp,
  CircuitBoard,
  Cpu,
  Database,
  FileCode2,
  FilePlus2,
  FolderOpen,
  GitBranch,
  Layers,
  LoaderCircle,
  Network,
  Play,
  Plug,
  RotateCcw,
  Save,
  Search,
  Settings2,
  ShieldCheck,
  Square,
  Terminal,
  Workflow,
  X,
  Zap,
} from "lucide-vue-next";
import { rpc, subscribe } from "./api";
import type { Job, Operator } from "./types";
import { emptyGraph, useEditor } from "./stores/editor";
import ModelWorkbench from "./components/ModelWorkbench.vue";
import AnalysisInspector from "./components/AnalysisInspector.vue";
import { useCode } from "./stores/code";
import NodeInspector from "./components/NodeInspector.vue";
import LossChart from "./components/LossChart.vue";
import TrainingPanel from "./components/TrainingPanel.vue";

const editor = useEditor();
const code = useCode();
const inspectorVisible = ref(true);
const workspace = ref("Model Graph");
const bottomTab = ref("Training Metrics");
const search = ref("");
const libraryTab = ref("Operators");
const error = ref("");
const busy = ref(false);
const ready = ref(false);
const environment = ref<any>(null);
const appInfo = ref<any>(null);
const forward = ref<any>(null);
const consoleLines = ref<{ time: number; message: string }[]>([]);
const modal = ref<"new" | "open" | "resume" | "discard" | "">("");
const newName = ref("My model");
const newPath = ref("");
const newTemplate = ref("empty");
const openPath = ref("");
const resumePath = ref("");
const resumeEpochs = ref(30);
const resumeConfig = ref<any>(null);
let deferredAction: (() => void) | null = null;
let unsubscribe = () => {};
let poll: ReturnType<typeof setInterval>;
let polling = false;

const workspaces = [
  { label: "Model Graph", icon: Workflow },
  { label: "Dataset", icon: Database },
  { label: "Training", icon: Activity },
  { label: "Optimization", icon: Zap },
  { label: "Profiler", icon: CircuitBoard },
  { label: "Plugins", icon: Plug },
];
const categories = computed(() =>
  [...new Set(editor.registry.map((op) => op.category))]
    .map((name) => ({
      name,
      operators: editor.registry.filter(
        (op) =>
          op.category === name &&
          `${op.label} ${op.description}`
            .toLowerCase()
            .includes(search.value.toLowerCase()),
      ),
    }))
    .filter((c) => c.operators.length),
);
const latestMetric = computed(() => editor.job?.metrics.at(-1));
const consoleOutput = computed(() =>
  [...consoleLines.value, ...(editor.job?.logs ?? [])]
    .sort((a, b) => a.time - b.time)
    .slice(-300),
);
const problemIssues = computed<
  {
    severity: string;
    message: string;
    node?: string;
    edge?: string;
    line?: number;
  }[]
>(() =>
  code.readOnly
    ? code.diagnostics
    : [...editor.validation.issues, ...code.diagnostics],
);
const problemCount = computed(() => problemIssues.value.length);
function locateProblem(issue: { node?: string; line?: number }) {
  if (code.readOnly && issue.node)
    void code.locateNode(issue.node).catch((e) => showError(String(e)));
  else if (issue.line) {
    if (code.view === "graph") code.view = "split";
    code.jump = { line: issue.line, token: Date.now() };
  } else editor.selected = issue.node ?? null;
}

function log(message: string) {
  consoleLines.value.push({ time: Date.now() / 1000, message });
  consoleLines.value = consoleLines.value.slice(-200);
}
function showError(message: string) {
  error.value = message;
  log(message);
}
async function perform(action: () => Promise<unknown>) {
  busy.value = true;
  error.value = "";
  try {
    await action();
  } catch (err) {
    showError(String(err));
  } finally {
    busy.value = false;
  }
}
async function initialize() {
  await perform(async () => {
    const [info, env, operators] = await Promise.all([
      rpc("app.info"),
      rpc("environment"),
      rpc<Operator[]>("registry"),
    ]);
    appInfo.value = info;
    environment.value = env;
    editor.registry = operators;
    ready.value = true;
    log(`Worker connected · Python ${env.python} · PyTorch ${env.pytorch}`);
    log(env.notice);
    const previous = localStorage.getItem("opweaver.lastProject");
    if (previous && !editor.projectPath) {
      try {
        await editor.load(previous);
        log(`Restored project ${previous}`);
        return;
      } catch (err) {
        log(`Could not restore last project: ${String(err)}`);
        localStorage.removeItem("opweaver.lastProject");
      }
    }
    if (!editor.graph.nodes?.length) await loadExample("synthetic-mlp");
  });
}
async function loadExample(name: string) {
  const example = await rpc("example", { name });
  editor.replace(example.graph, example.layout);
  workspace.value = "Model Graph";
  log(
    `Loaded ${example.manifest.name}. Create a project to save your own copy.`,
  );
}
function askDiscard(action: () => void) {
  if (editor.running || code.running) return;
  if ((editor.dirty || code.dirty) && editor.projectPath) {
    deferredAction = action;
    modal.value = "discard";
  } else action();
}
function showNew(template = "empty") {
  askDiscard(() => {
    newTemplate.value = template;
    newName.value =
      template === "current"
        ? (editor.graph.metadata?.name ?? "My model")
        : "My model";
    newPath.value = `${appInfo.value?.default_project_dir ?? ""}/model-${Date.now().toString(36)}`;
    modal.value = "new";
  });
}
async function createProject() {
  await perform(async () => {
    let graph = editor.graph,
      layout = editor.layout;
    if (newTemplate.value === "empty") {
      graph = emptyGraph(newName.value);
      layout = { version: 1, positions: {} };
    } else if (newTemplate.value !== "current") {
      const example = await rpc("example", { name: newTemplate.value });
      graph = example.graph;
      layout = example.layout;
    }
    graph = JSON.parse(JSON.stringify(graph));
    graph.metadata = { ...graph.metadata, name: newName.value };
    const snapshot = await rpc("project.create", {
      path: newPath.value,
      name: newName.value,
      graph,
      layout,
    });
    editor.replace(snapshot.graph, snapshot.layout, snapshot.path, true);
    localStorage.setItem("opweaver.lastProject", snapshot.path);
    modal.value = "";
    workspace.value = "Model Graph";
    log(`Project created: ${snapshot.path}`);
  });
}
function save() {
  if (!editor.projectPath) {
    showNew("current");
    return;
  }
  void perform(async () => {
    await code.saveAll();
    await code.persist();
    await editor.save();
    log("Graph and layout saved as a new project revision.");
  });
}
function openProject() {
  askDiscard(() => {
    openPath.value = "";
    modal.value = "open";
  });
}
async function loadProject() {
  await perform(async () => {
    await editor.load(openPath.value);
    localStorage.setItem("opweaver.lastProject", editor.projectPath);
    modal.value = "";
    workspace.value = "Model Graph";
    log(`Opened ${editor.projectPath}`);
  });
}
function drag(event: DragEvent, op: Operator) {
  event.dataTransfer?.setData("application/opweaver-operator", op.id);
  if (event.dataTransfer) event.dataTransfer.effectAllowed = "copy";
}
function add(op: Operator) {
  if (code.readOnly) {
    showError(
      "Code-First is an analysis graph. Edit Python source, or switch to Visual Authoring before adding design nodes.",
    );
    return;
  }
  workspace.value = "Model Graph";
  editor.addNode(op.id, {
    x: 120 + ((editor.graph.nodes?.length ?? 0) % 4) * 220,
    y: 100 + Math.floor((editor.graph.nodes?.length ?? 0) / 4) * 180,
  });
}
async function runForward() {
  if (
    code.activeFile &&
    workspace.value === "Model Graph" &&
    code.view !== "graph"
  ) {
    await perform(code.startCapture);
    log("Started Python model capture with a real CPU forward pass.");
    return;
  }
  await perform(async () => {
    const report = await editor.validateNow();
    if (!report.valid) {
      bottomTab.value = "Problems";
      throw new Error("Fix graph issues before running the model.");
    }
    forward.value = await rpc("forward", { graph: editor.graph });
    bottomTab.value = "Tensor Inspector";
    log(
      `Forward completed on CPU · ${forward.value.parameters} parameters · ${forward.value.outputs.length} output(s)`,
    );
  });
}
async function train(resume?: string) {
  if (code.readOnly) {
    showError(
      "Training currently uses the editable Design Graph. Code-First capture runs forward analysis; switch to Visual Authoring to train your design.",
    );
    return;
  }
  if (!editor.projectPath) {
    showNew("current");
    return;
  }
  await perform(async () => {
    const report = await editor.validateNow();
    if (!report.valid) {
      bottomTab.value = "Problems";
      throw new Error("Fix graph issues before training.");
    }
    await editor.save();
    const result = await rpc<Job>("train.start", {
      graph: editor.graph,
      config: editor.config,
      project: editor.projectPath,
      resume,
    });
    if (editor.job?.id !== result.id) editor.job = result;
    bottomTab.value = "Training Metrics";
    modal.value = "";
    log(
      `${resume ? "Resuming" : "Started"} experiment ${result.id.slice(0, 8)}`,
    );
  });
}
function prepareResume(path: string) {
  const run = editor.history.find((j) => j.checkpoint === path) ?? editor.job;
  resumePath.value = path;
  resumeConfig.value = run ? { ...run.config } : { ...editor.config };
  resumeEpochs.value = Math.max(
    editor.config.epochs,
    (run?.metrics.at(-1)?.epoch ?? 0) + 10,
  );
  modal.value = "resume";
}
async function resume() {
  editor.config = { ...resumeConfig.value, epochs: resumeEpochs.value };
  await train(resumePath.value);
}
async function refreshHistory() {
  try {
    await editor.loadHistory();
  } catch (err) {
    log(String(err));
  }
}
function keydown(event: KeyboardEvent) {
  if (
    event.defaultPrevented ||
    (event.target as HTMLElement).closest(".monaco-editor")
  )
    return;
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
    event.preventDefault();
    save();
  }
  const target = event.target as HTMLElement;
  if (["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName) || modal.value)
    return;
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
    event.preventDefault();
    event.shiftKey ? editor.redo() : editor.undo();
  }
}
function beforeUnload(event: BeforeUnloadEvent) {
  if (editor.dirty || editor.running || code.dirty || code.running) {
    event.preventDefault();
    event.returnValue = "";
  }
}
onMounted(() => {
  unsubscribe = subscribe((event) => {
    if (event.event === "code.task" && event.data?.id) {
      void code
        .acceptJob(event.data)
        .then(() => {
          if (event.data.state === "COMPLETED" && event.data.result?.forward) {
            forward.value = event.data.result.forward;
            log(
              `Python capture completed · ${forward.value.parameters} parameters · real CPU forward.`,
            );
          } else if (event.data.state === "FAILED")
            log(event.data.error ?? "Python capture failed");
        })
        .catch((e) => showError(String(e)));
    } else if (event.data?.id) {
      editor.job = event.data as Job;
      if (
        event.event === "state" &&
        ["COMPLETED", "FAILED", "CANCELLED"].includes(event.data.state)
      ) {
        void refreshHistory();
        if (event.data.error) showError(event.data.error);
      }
    } else if (event.event === "worker.exited") {
      if (code.running && code.captureJob) {
        code.captureJob.state = "FAILED";
        code.status = "parse_failed";
        code.error =
          "Worker exited; previous source and analysis were retained.";
      }
      ready.value = false;
      if (editor.job && editor.running) {
        editor.job.state = "FAILED";
        editor.job.error = "Worker exited unexpectedly.";
      }
      showError(
        "Python Worker disconnected. Click Reconnect to restart it; the last checkpoint can be resumed.",
      );
    } else if (event.event === "worker.stderr") log(event.message ?? "");
  });
  void initialize();
  poll = setInterval(async () => {
    if (!editor.running || !editor.job || polling) return;
    polling = true;
    try {
      editor.job = await rpc<Job>("train.status", { id: editor.job.id });
      if (!editor.running) await refreshHistory();
    } catch (err) {
      if (editor.job) editor.job.state = "FAILED";
      showError(String(err));
    } finally {
      polling = false;
    }
  }, 2000);
  window.addEventListener("keydown", keydown);
  window.addEventListener("beforeunload", beforeUnload);
});
onUnmounted(() => {
  unsubscribe();
  clearInterval(poll);
  window.removeEventListener("keydown", keydown);
  window.removeEventListener("beforeunload", beforeUnload);
});
</script>

<template>
  <div class="studio">
    <header class="titlebar">
      <div class="brand">
        <div class="brand-mark"><Network :size="23" /></div>
        <strong>OpWeaver<span>Studio</span></strong
        ><span class="version">0.1</span>
      </div>
      <div class="titlebar-center">
        {{ editor.graph.metadata?.name }}<span class="separator">/</span
        ><span class="muted">Model engineering</span>
      </div>
      <button
        class="environment-button"
        title="Refresh worker environment"
        @click="initialize"
      >
        <span class="live-dot" :class="{ warning: !ready }"></span
        >{{
          ready
            ? `Python ${environment?.python} · PyTorch ${environment?.pytorch?.split("+")[0]}`
            : busy
              ? "Connecting to Worker…"
              : "Reconnect Worker"
        }}<ChevronDown :size="12" />
      </button>
    </header>
    <div class="projectbar">
      <div class="project-actions">
        <button
          title="New project"
          :disabled="busy || editor.running || code.running"
          @click="showNew()"
        >
          <FilePlus2 :size="15" /><span>New</span></button
        ><button
          title="Open project"
          :disabled="busy || editor.running || code.running"
          @click="openProject"
        >
          <FolderOpen :size="15" /><span>Open</span></button
        ><button
          title="Save project (Ctrl+S)"
          :disabled="busy || !ready"
          @click="save"
        >
          <Save :size="15" /><span>Save</span></button
        ><i></i
        ><span class="save-state"
          ><span :class="{ 'unsaved-dot': editor.dirty || code.dirty }"></span
          >{{
            editor.projectPath
              ? editor.dirty || code.dirty
                ? "Unsaved changes"
                : "All changes saved"
              : "Example · not saved"
          }}</span
        >
      </div>
      <div class="run-actions">
        <span class="device-chip"
          ><Cpu :size="14" />{{ editor.config.device.toUpperCase() }}</span
        ><button
          class="subtle"
          :disabled="!ready || busy || editor.running"
          @click="runForward"
        >
          <Play :size="13" /> Forward</button
        ><button
          v-if="editor.running"
          class="stop-button"
          @click="
            perform(async () => {
              editor.job = await rpc('train.stop', { id: editor.job?.id });
            })
          "
        >
          <Square :size="12" /> Stop training</button
        ><button
          v-else
          class="primary"
          :disabled="!ready || busy || code.running || code.readOnly"
          @click="train()"
        >
          <Play :size="13" fill="currentColor" /> Train model
        </button>
      </div>
    </div>
    <div class="studio-body">
      <nav class="activitybar">
        <button
          v-for="item in workspaces"
          :key="item.label"
          :title="item.label"
          :class="{ active: workspace === item.label }"
          @click="workspace = item.label"
        >
          <component :is="item.icon" :size="20" />
        </button>
        <div class="activity-spacer"></div>
        <button title="Environment details" @click="bottomTab = 'Output'">
          <CircleHelp :size="19" />
        </button>
      </nav>
      <aside class="library">
        <div class="panel-heading">
          <Layers :size="15" /><strong>Component library</strong>
        </div>
        <div class="library-tabs">
          <button
            v-for="tab in ['Operators', 'Models']"
            :key="tab"
            :class="{ active: libraryTab === tab }"
            @click="libraryTab = tab"
          >
            {{ tab }}
          </button>
        </div>
        <div v-if="libraryTab === 'Operators'" class="library-content">
          <div class="search-field">
            <Search :size="14" /><input
              v-model="search"
              aria-label="Search operators"
              placeholder="Find an operator…"
            /><kbd>⌕</kbd>
          </div>
          <div class="library-hint">
            DRAG TO CANVAS <span>{{ editor.registry.length }}</span>
          </div>
          <div
            v-for="category in categories"
            :key="category.name"
            class="operator-category"
          >
            <h3>
              <ChevronDown :size="11" />{{ category.name
              }}<span>{{ category.operators.length }}</span>
            </h3>
            <button
              v-for="op in category.operators"
              :key="op.id"
              class="operator-item"
              :class="op.category.toLowerCase()"
              draggable="true"
              :title="`${op.description} Click to add or drag to canvas.`"
              @dragstart="drag($event, op)"
              @click="add(op)"
            >
              <span class="operator-glyph"
                ><component
                  :is="
                    op.category === 'Activation'
                      ? Activity
                      : op.category === 'Tensor'
                        ? Braces
                        : op.category === 'Structure'
                          ? GitBranch
                          : Box
                  "
                  :size="15" /></span
              ><span>{{ op.label }}</span
              ><span class="drag-dots">⠿</span>
            </button>
          </div>
          <p v-if="search && !categories.length" class="empty-note">
            No matching operators.
          </p>
        </div>
        <div v-else class="library-content model-library">
          <p class="muted">Offline examples</p>
          <button
            @click="
              askDiscard(() => {
                void perform(() => loadExample('synthetic-mlp'));
              })
            "
          >
            <Network :size="22" /><strong>Synthetic classifier</strong
            ><span>8 → 16 → 3 · MLP</span><ArrowRight :size="14" /></button
          ><button
            @click="
              askDiscard(() => {
                void perform(() => loadExample('residual-mlp'));
              })
            "
          >
            <GitBranch :size="22" /><strong>Residual classifier</strong
            ><span>Branch + Add · DAG</span><ArrowRight :size="14" />
          </button>
          <p class="muted small">Custom module library · Coming Later (P2)</p>
        </div>
        <div class="project-files">
          <div class="section-label">
            PROJECT FILES <FileCode2 :size="12" />
          </div>
          <div>
            <ChevronRight :size="12" /><FolderOpen :size="13" /><span>{{
              editor.graph.metadata?.name
            }}</span>
          </div>
          <div>
            <FileCode2 :size="13" /><span>Graph IR</span><code>v1.0</code>
          </div>
          <div>
            <Braces :size="13" /><span>UI layout</span><code>JSON</code>
          </div>
          <div>
            <Database :size="13" /><span>Experiments</span><code>SQLite</code>
          </div>
          <button
            v-for="file in code.files"
            :key="file"
            class="source-explorer-file"
            :title="file"
            @click="
              workspace = 'Model Graph';
              code.view = 'split';
              perform(() => code.open(file));
            "
          >
            <FileCode2 :size="12" /><span>{{ file }}</span>
          </button>
          <p :title="editor.projectPath">
            {{ editor.projectPath || "Create a project to persist files" }}
          </p>
        </div>
        <div class="library-footer">
          <ShieldCheck :size="13" /> Local execution · explicit plugins
        </div>
      </aside>
      <main class="main-workspace">
        <div class="workspace-tabs">
          <button
            v-for="item in workspaces"
            :key="item.label"
            :class="{ active: workspace === item.label }"
            @click="workspace = item.label"
          >
            <component :is="item.icon" :size="13" />{{ item.label
            }}<span
              v-if="
                ['Optimization', 'Profiler', 'Plugins'].includes(item.label)
              "
              class="later-dot"
            ></span>
          </button>
        </div>
        <div v-if="error" class="error-banner" role="alert">
          <span>{{ error }}</span
          ><button title="Dismiss error" @click="error = ''">
            <X :size="14" />
          </button>
        </div>
        <div class="work-area">
          <ModelWorkbench
            v-if="workspace === 'Model Graph'"
            :desktop="appInfo?.desktop"
            :inspector-visible="inspectorVisible"
            @toggle-inspector="inspectorVisible = !inspectorVisible"
            @error="showError"
          /><TrainingPanel
            v-else-if="workspace === 'Training'"
            :cuda-available="environment?.cuda_available ?? false"
            @train="train()"
            @resume="prepareResume"
          />
          <div
            v-else-if="workspace === 'Dataset'"
            class="information-workspace"
          >
            <div class="info-icon"><Database :size="30" /></div>
            <span class="eyebrow">DATASET / SYNTHETIC V1</span>
            <h1>Everything you need to begin.</h1>
            <p>
              The included dataset is generated entirely offline from your
              graph's input and output dimensions. Fixed seeds make each
              experiment reproducible.
            </p>
            <div class="info-facts">
              <div><span>Input</span><code>[B, features] · float32</code></div>
              <div>
                <span>Classification</span><code>argmax(X @ teacher)</code>
              </div>
              <div><span>Regression</span><code>X @ teacher</code></div>
              <div>
                <span>Samples / Seed</span
                ><code
                  >{{ editor.config.samples }} / {{ editor.config.seed }}</code
                >
              </div>
            </div>
            <button class="primary" @click="workspace = 'Training'">
              Configure dataset & training <ArrowRight :size="14" />
            </button>
            <p class="muted small">
              External datasets, transforms and DataLoader plugins: Coming
              Later.
            </p>
          </div>
          <div v-else class="information-workspace">
            <div class="info-icon">
              <component
                :is="workspaces.find((w) => w.label === workspace)?.icon"
                :size="30"
              />
            </div>
            <span class="eyebrow">COMING LATER</span>
            <h1>{{ workspace }}</h1>
            <p v-if="workspace === 'Optimization'">
              Versioned passes, capability checks and numerical validation form
              the extension boundary. Export, quantization and backend
              compilation are planned for P3.
            </p>
            <p v-else-if="workspace === 'Profiler'">
              Kernel timelines, memory analysis and measured backend comparisons
              are planned for P3–P4. No profiling results have been generated.
            </p>
            <p v-else>
              Built-in operators support visual construction and Python code
              generation. External plugin loading and its explicit trust
              approval are planned for a later stage.
            </p>
            <div class="honest-state">
              <ShieldCheck :size="16" />This workspace is planned and does not
              execute operations.
            </div>
          </div>
        </div>
        <section class="bottom-panel">
          <div class="bottom-tabs">
            <button
              v-for="tab in [
                'Training Metrics',
                'Console',
                'Problems',
                'Tensor Inspector',
                'Output',
              ]"
              :key="tab"
              :class="{ active: bottomTab === tab }"
              @click="bottomTab = tab"
            >
              {{ tab
              }}<span
                v-if="tab === 'Problems' && problemCount"
                class="count-badge"
                >{{ problemCount }}</span
              ></button
            ><span v-if="editor.job" class="job-indicator"
              ><span
                class="live-dot"
                :class="{ warning: editor.job.state === 'FAILED' }"
              ></span
              >{{ editor.job.state }}</span
            >
          </div>
          <div class="bottom-content">
            <template v-if="bottomTab === 'Training Metrics'"
              ><div class="metrics-bar">
                <div class="legend">
                  <span></span>Training loss
                  <small>{{
                    editor.job
                      ? editor.job.id.slice(0, 8)
                      : "no active experiment"
                  }}</small>
                </div>
                <div class="metric-readouts">
                  <span
                    >LOSS
                    <strong>{{
                      latestMetric?.loss.toFixed(4) ?? "—"
                    }}</strong></span
                  ><span
                    >ACC
                    <strong>{{
                      latestMetric?.accuracy != null
                        ? `${(latestMetric.accuracy * 100).toFixed(1)}%`
                        : "—"
                    }}</strong></span
                  ><span
                    >EPOCH
                    <strong
                      >{{ latestMetric?.epoch ?? "—"
                      }}<small v-if="editor.job">
                        / {{ editor.job.config.epochs }}</small
                      ></strong
                    ></span
                  ><span
                    >LR
                    <strong>{{
                      latestMetric?.learning_rate ?? "—"
                    }}</strong></span
                  ><span
                    >TIME
                    <strong>{{
                      latestMetric
                        ? `${latestMetric.elapsed_seconds.toFixed(1)}s`
                        : "—"
                    }}</strong></span
                  ><button
                    v-if="editor.job?.checkpoint"
                    class="subtle compact"
                    :disabled="editor.running"
                    @click="prepareResume(editor.job.checkpoint)"
                  >
                    <RotateCcw :size="12" />Resume
                  </button>
                </div>
              </div>
              <LossChart :metrics="editor.job?.metrics ?? []"
            /></template>
            <div v-else-if="bottomTab === 'Console'" class="console-output">
              <p v-for="(line, i) in consoleOutput" :key="i">
                <time>{{
                  new Date(line.time * 1000).toLocaleTimeString()
                }}</time
                ><span>{{ line.message }}</span>
              </p>
              <p v-if="!consoleOutput.length" class="muted">
                Worker logs will appear here.
              </p>
            </div>
            <div v-else-if="bottomTab === 'Problems'" class="problems-list">
              <div v-if="!problemCount" class="all-clear">
                <Check :size="17" />No graph problems detected.
              </div>
              <button
                v-for="(issue, i) in problemIssues"
                :key="i"
                @click="locateProblem(issue)"
              >
                <span class="error-symbol">×</span
                ><span>{{ issue.message }}</span
                ><code>{{ issue.node ?? issue.edge ?? "graph" }}</code>
              </button>
            </div>
            <div
              v-else-if="bottomTab === 'Tensor Inspector'"
              class="tensor-output"
            >
              <template v-if="forward"
                ><div class="result-heading">
                  <span class="pill positive">FORWARD COMPLETE</span
                  ><span
                    >{{ forward.parameters }} learnable parameters ·
                    {{ (forward.device ?? "cpu").toUpperCase() }}</span
                  >
                </div>
                <div
                  v-for="(tensor, i) in forward.outputs"
                  :key="i"
                  class="tensor-result"
                >
                  <strong>Output {{ i }}</strong
                  ><code>[{{ tensor.shape.join(", ") }}]</code
                  ><span>{{ tensor.dtype }}</span
                  ><span>mean {{ tensor.mean.toFixed(5) }}</span
                  ><span>min {{ tensor.min?.toFixed(5) ?? "—" }}</span
                  ><span>max {{ tensor.max?.toFixed(5) ?? "—" }}</span>
                </div></template
              >
              <p v-else class="muted">
                Run Forward to inspect actual tensor statistics. No tensors have
                been executed yet.
              </p>
            </div>
            <div v-else class="output-details">
              <p>
                <strong>Execution environment</strong>
                {{ environment?.notice ?? "Worker is connecting…" }}
              </p>
              <p>
                Python {{ environment?.python ?? "—" }} · PyTorch
                {{ environment?.pytorch ?? "—" }} ·
                {{ environment?.device ?? "—" }}
              </p>
              <p>
                Project: <code>{{ editor.projectPath || "Not saved" }}</code>
              </p>
              <p>
                Checkpoint:
                <code>{{ editor.job?.checkpoint || "No checkpoint yet" }}</code>
              </p>
              <p class="muted">
                Graph IR v1.0 · JSON Lines RPC v1 · weights remain on disk
              </p>
            </div>
          </div>
        </section>
      </main>
      <AnalysisInspector
        v-if="workspace === 'Model Graph' && inspectorVisible && code.readOnly"
      />
      <NodeInspector
        v-else-if="workspace === 'Model Graph' && inspectorVisible"
        @error="showError"
      />
    </div>
    <footer class="statusbar">
      <span
        ><GitBranch :size="12" />{{
          editor.graph.metadata?.model_version ?? "0.1.0"
        }}</span
      ><span
        ><span class="live-dot" :class="{ warning: !ready }"></span
        >{{ ready ? "Worker connected" : "Worker disconnected" }}</span
      ><span
        >{{
          code.readOnly
            ? (code.activeAnalysis?.nodes.length ?? 0)
            : (editor.graph.nodes?.length ?? 0)
        }}
        nodes ·
        {{
          code.readOnly
            ? (code.activeAnalysis?.edges.length ?? 0)
            : (editor.graph.edges?.length ?? 0)
        }}
        edges</span
      >
      <div></div>
      <span v-if="busy"><LoaderCircle :size="12" class="spin" />Working…</span
      ><span>Graph IR 1.0</span
      ><span class="tagline">Weave Models. Forge Kernels.</span>
    </footer>

    <div v-if="modal" class="modal-backdrop" @click.self="modal = ''">
      <section class="modal" role="dialog" aria-modal="true">
        <button class="modal-close" title="Close dialog" @click="modal = ''">
          <X :size="18" />
        </button>
        <template v-if="modal === 'new'"
          ><div class="modal-icon"><FilePlus2 :size="24" /></div>
          <h2>New OpWeaver project</h2>
          <p>A home for your graph, experiments and checkpoints.</p>
          <label>Project name<input v-model="newName" autofocus /></label
          ><label
            >Project directory
            <div class="input-with-button">
              <input v-model="newPath" /><button
                v-if="appInfo?.desktop"
                title="Choose parent folder"
                @click="
                  perform(async () => {
                    const folder = await rpc('dialog.directory');
                    if (folder) newPath = `${folder}/opweaver-project`;
                  })
                "
              >
                <FolderOpen :size="16" />
              </button></div></label
          ><label
            >Starting point<select v-model="newTemplate">
              <option value="empty">Empty graph</option>
              <option value="current">Current graph</option>
              <option value="synthetic-mlp">
                Synthetic classifier example
              </option>
              <option value="residual-mlp">Residual classifier example</option>
            </select></label
          >
          <p class="small muted">
            Choose a new or empty directory. Existing project files are never
            overwritten.
          </p>
          <button
            class="primary full-width"
            :disabled="busy || !newPath || !newName"
            @click="createProject"
          >
            Create project <ArrowRight :size="14" /></button
        ></template>
        <template v-else-if="modal === 'open'"
          ><div class="modal-icon"><FolderOpen :size="24" /></div>
          <h2>Open project</h2>
          <p>Open a project directory or its project.opweaver.json file.</p>
          <label
            >Project path
            <div class="input-with-button">
              <input v-model="openPath" autofocus /><button
                v-if="appInfo?.desktop"
                title="Browse project"
                @click="
                  perform(async () => {
                    const path = await rpc('dialog.open');
                    if (path) openPath = path;
                  })
                "
              >
                <FolderOpen :size="16" />
              </button></div></label
          ><button
            class="primary full-width"
            :disabled="busy || !openPath"
            @click="loadProject"
          >
            Open project
          </button></template
        >
        <template v-else-if="modal === 'resume'"
          ><div class="modal-icon"><RotateCcw :size="24" /></div>
          <h2>Resume experiment</h2>
          <p>
            Restore model, optimizer, step and RNG state. The saved graph and
            training configuration must match.
          </p>
          <code class="checkpoint-path">{{ resumePath }}</code
          ><label
            >Total target epochs<input
              v-model.number="resumeEpochs"
              type="number"
              min="1"
          /></label>
          <p class="muted small">
            This is the total epoch target, including already completed epochs.
            All other settings are restored from this experiment.
          </p>
          <button class="primary full-width" :disabled="busy" @click="resume">
            Resume training <Play :size="14" /></button
        ></template>
        <template v-else
          ><h2>Unsaved changes</h2>
          <p>
            Save your current graph before switching projects, or discard these
            unsaved edits.
          </p>
          <div class="modal-actions">
            <button @click="modal = ''">Keep editing</button
            ><button
              class="danger"
              @click="
                modal = '';
                deferredAction?.();
              "
            >
              Discard changes</button
            ><button
              class="primary"
              @click="
                perform(async () => {
                  await code.saveAll();
                  await code.persist();
                  await editor.save();
                  modal = '';
                  deferredAction?.();
                })
              "
            >
              Save & continue
            </button>
          </div></template
        >
        <p v-if="error" class="modal-error" role="alert">{{ error }}</p>
      </section>
    </div>
  </div>
</template>
