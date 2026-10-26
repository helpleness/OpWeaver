<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from "vue";
import {
  FlaskConical,
  Play,
  Square,
  Download,
  GitCompareArrows,
  CheckCircle2,
  AlertTriangle,
  Cpu,
  FolderOpen,
  RefreshCw,
  ArrowUpRight,
} from "lucide-vue-next";
import { useLab } from "../stores/lab";
import { useModels } from "../stores/models";
import { useEditor } from "../stores/editor";
import { rpc } from "../api";
import { bytesText, shapeText } from "../model-types";
import { metric, percent } from "../lab-types";
import LatencyChart from "./LatencyChart.vue";
import OliveWorkflow from "./OliveWorkflow.vue";
import DatasetPanel from "./DatasetPanel.vue";
import TaskQualityCard from "./TaskQualityCard.vue";
import DiagnosticsPanel from "./DiagnosticsPanel.vue";
import { useDiagnostics } from "../stores/diagnostics";
import type { DiagnosticRecord, DiagnosticReport } from "../diagnostic-types";
const props = defineProps<{ desktop?: boolean }>();
const emit = defineEmits<{
  viewModel: [id: string];
  importModel: [];
  viewDiagnostic: [
    report: DiagnosticReport,
    row: DiagnosticRecord,
    side: "fp32" | "quantized",
    low: number | null,
    high: number | null,
  ];
}>();
const diagnostics = useDiagnostics();
function viewDiagnostic(
  report: DiagnosticReport,
  row: DiagnosticRecord,
  side: "fp32" | "quantized",
  low: number | null,
  high: number | null,
) {
  emit("viewDiagnostic", report, row, side, low, high);
}
const lab = useLab(),
  models = useModels(),
  editor = useEditor();
const shapes = ref<Record<string, string>>({});
const npzFiles = ref("");
const exportDirectory = ref("");
const busy = ref(false);
const tab = ref("Results");
const crossCompare = ref<any>(null);
const result = computed(() => lab.experiment);
const capability = computed(() =>
  lab.backends.find((b) => b.id === lab.config.optimization.backend_id),
);
const sources = computed(() =>
  models.completed.filter((m) => m.artifact?.format === "onnx"),
);
const disabled = computed(() => lab.running || busy.value || lab.loading);
const canStart = computed(
  () =>
    !disabled.value &&
    !!lab.source &&
    !!capability.value?.available &&
    !editor.running &&
    !models.running &&
    !diagnostics.running,
);
const comparison = computed(
  () => result.value?.comparison as Record<string, any> | null,
);
const graphComparison = computed(
  () => result.value?.graph_comparison as Record<string, any> | null,
);
const validationRows = computed(
  () => (result.value?.validation?.outputs ?? []) as Record<string, any>[],
);
const rows = computed(() => [
  ["Model size", "model_bytes", "bytes"],
  ["Mean latency", "mean_ms", "ms"],
  ["P50 latency", "p50_ms", "ms"],
  ["P95 latency", "p95_ms", "ms"],
  ["Min / Max", "min_ms", "range"],
  ["Throughput", "requests_per_second", "requests/s"],
  ["Batch throughput", "samples_per_second", "samples/s"],
  ["Session initialization", "session_init_ms", "ms"],
  ["Warmup total", "warmup_ms", "ms"],
]);
function display(side: "baseline" | "optimized", key: string, unit: string) {
  const data = result.value?.[side] as Record<string, any> | null;
  const value = data?.[key];
  if (value == null) return "—";
  if (unit === "bytes") return bytesText(value);
  if (unit === "range") return `${metric(value)} / ${metric(data?.max_ms)} ms`;
  return `${metric(value)} ${unit}`;
}
function updateDraft() {
  shapes.value = Object.fromEntries(
    (lab.source?.inputs ?? []).map((t) => [
      t.name,
      JSON.stringify(lab.config.inputs.shapes[t.name] ?? t.shape ?? []),
    ]),
  );
  npzFiles.value = lab.config.inputs.files.join("\n");
}
watch(
  () => [lab.source?.artifact.id, lab.experiment?.id, lab.config.inputs],
  updateDraft,
);
watch(
  () => lab.experiment?.id,
  () => {
    crossCompare.value = null;
  },
);
async function compareExperiment(id: string) {
  if (!id || !lab.experiment) {
    crossCompare.value = null;
    return;
  }
  crossCompare.value = await rpc("lab.compare", {
    project: editor.projectPath,
    reference_id: id,
    candidate_id: lab.experiment.id,
  });
}
async function action(fn: () => Promise<unknown>) {
  busy.value = true;
  try {
    await fn();
  } catch (e) {
    lab.error = String(e);
  } finally {
    busy.value = false;
  }
}
function prepareInputs(reuseDataset = true) {
  const config = lab.config;
  if (reuseDataset && config.evaluation.dataset_id) {
    if (!config.benchmark.batch_size) {
      config.benchmark.batch_size = null;
      config.benchmark.batch_input = null;
    }
    return;
  }
  if (config.inputs.mode === "random") {
    const values: Record<string, number[]> = {};
    for (const input of lab.source?.inputs ?? []) {
      let shape;
      try {
        shape = JSON.parse(shapes.value[input.name] ?? "");
      } catch {
        throw new Error(
          `Enter a concrete JSON shape for ${input.name}, for example [2,4]`,
        );
      }
      if (
        !Array.isArray(shape) ||
        shape.some((d) => !Number.isInteger(d) || d < 0)
      )
        throw new Error(
          `Resolve all dynamic dimensions for ${input.name}; scalar shape is []`,
        );
      values[input.name] = shape;
    }
    config.inputs.shapes = values;
    config.inputs.files = [];
  } else {
    config.inputs.shapes = {};
    config.inputs.files = npzFiles.value
      .split(/\r?\n/)
      .map((s) => s.trim())
      .filter(Boolean);
  }
  // An empty HTML numeric field is represented as an empty string by Vue.
  if (!config.benchmark.batch_size) {
    config.benchmark.batch_size = null;
    config.benchmark.batch_input = null;
  }
}
async function start(mode: "baseline" | "optimize") {
  prepareInputs();
  await lab.start(mode);
  tab.value = "Results";
}
async function chooseInputs() {
  const paths = await rpc<string[]>("dialog.inputs");
  if (paths?.length) npzFiles.value = paths.join("\n");
}
async function exportReport() {
  let path = exportDirectory.value;
  if (props.desktop) path = await rpc<string>("dialog.directory");
  if (path) await lab.exportReport(path);
}
let timer: ReturnType<typeof setInterval>;
let polling = false;
onMounted(() => {
  void action(async () => {
    await Promise.all([lab.probe(), lab.refresh(), models.refresh()]);
    if (!lab.source && !lab.loading) {
      const selected =
        models.report?.artifact.format === "onnx"
          ? models.report.artifact.id
          : sources.value[0]?.id;
      if (selected) await lab.selectSource(selected);
    }
    updateDraft();
  });
  timer = setInterval(async () => {
    if (polling || !lab.running) return;
    polling = true;
    try {
      await lab.poll();
    } catch (e) {
      lab.error = String(e);
    } finally {
      polling = false;
    }
  }, 1000);
});
onUnmounted(() => clearInterval(timer));
</script>
<template>
  <section class="optimization-lab">
    <header class="lab-header">
      <div class="lab-heading-mark"><FlaskConical :size="22" /></div>
      <div>
        <span class="eyebrow">LOCAL INFERENCE ENGINEERING</span>
        <h1>Optimization Lab</h1>
      </div>
      <span class="lab-cpu"><Cpu :size="13" />CPU · concurrency 1</span>
      <button
        title="Refresh backend and experiment history"
        :disabled="disabled"
        @click="
          action(async () => {
            await lab.probe();
            await lab.refresh();
            await models.refresh();
          })
        "
      >
        <RefreshCw :size="14" />
      </button>
    </header>
    <nav class="lab-mode-tabs" aria-label="Optimization Lab modes">
      <button
        :class="{ active: !diagnostics.showPanel }"
        @click="diagnostics.showPanel = false"
      >
        Benchmark & datasets
      </button>
      <button
        :class="{ active: diagnostics.showPanel }"
        :disabled="!editor.projectPath || lab.running"
        @click="diagnostics.showPanel = true"
      >
        Quantization diagnostics
      </button>
    </nav>
    <DiagnosticsPanel
      v-if="diagnostics.showPanel"
      :desktop="desktop"
      @view-node="viewDiagnostic"
    />
    <div v-else-if="!editor.projectPath || !sources.length" class="lab-empty">
      <FlaskConical :size="38" />
      <h2>Measure. Optimize. Verify.</h2>
      <p>
        Select a saved project and import an ONNX model to run a real CPU
        experiment.
      </p>
      <button class="primary" @click="emit('importModel')">
        Import a model <ArrowUpRight :size="14" />
      </button>
    </div>
    <div v-else class="lab-body">
      <aside class="lab-configuration">
        <fieldset :disabled="disabled">
          <div class="lab-step-title">
            <span>01</span><strong>Model & inputs</strong>
          </div>
          <label
            >Imported ModelArtifact<select
              aria-label="Lab source model"
              :value="lab.source?.artifact.id ?? ''"
              @change="
                action(() =>
                  lab.selectSource(($event.target as HTMLSelectElement).value),
                )
              "
            >
              <option disabled value="">Select ONNX model</option>
              <option
                v-for="model in sources"
                :key="model.id"
                :value="model.id"
              >
                {{ model.artifact?.name }} · {{ model.id.slice(0, 8) }}
              </option>
            </select></label
          >
          <template v-if="lab.source">
            <div class="lab-model-facts">
              <span
                >{{ lab.source.artifact.format.toUpperCase() }} ·
                {{ bytesText(lab.source.artifact.total_file_bytes) }}</span
              ><code :title="lab.source.artifact.sha256"
                >SHA256 {{ lab.source.artifact.sha256.slice(0, 16) }}…</code
              ><span :title="lab.source.artifact.source_path">{{
                lab.source.artifact.source_name
              }}</span>
            </div>
            <p
              v-for="output in lab.source.outputs"
              :key="output.name"
              class="lab-io"
            >
              OUT <strong>{{ output.name }}</strong> {{ shapeText(output) }} ·
              {{ output.dtype }}
            </p>
            <details
              v-if="lab.source.diagnostics.length"
              class="lab-diagnostics"
            >
              <summary>
                Model diagnostics ({{ lab.source.diagnostics.length }})
              </summary>
              <p v-for="(d, i) in lab.source.diagnostics" :key="i">
                {{ d.message }}
              </p>
            </details>
          </template>
          <label
            >Input data<select
              aria-label="Lab input mode"
              v-model="lab.config.inputs.mode"
            >
              <option value="random">Seeded random tensors</option>
              <option value="npz">User NPZ samples</option>
            </select></label
          >
          <template v-if="lab.config.inputs.mode === 'random'">
            <label v-for="input in lab.source?.inputs" :key="input.name"
              >{{ input.name }} · {{ input.dtype }} · {{ shapeText(input)
              }}<input
                :aria-label="`Shape for ${input.name}`"
                v-model="shapes[input.name]"
                spellcheck="false"
                placeholder="Concrete shape, e.g. [2,4]"
            /></label>
            <label
              >Random seed<input
                aria-label="Lab seed"
                type="number"
                min="0"
                max="4294967295"
                v-model.number="lab.config.inputs.seed"
            /></label>
            <p class="muted small">
              Resolve symbols explicitly. Random values may be unsuitable for
              indices or masks; use NPZ for meaningful inputs.
            </p>
          </template>
          <template v-else
            ><label
              >NPZ files · one sample per line<textarea
                aria-label="NPZ sample paths"
                rows="4"
                v-model="npzFiles"
                placeholder="C:\data\sample-0.npz"
              /></label
            ><button v-if="desktop" @click="action(chooseInputs)">
              <FolderOpen :size="13" />Choose NPZ files
            </button>
            <p class="muted small">
              Array keys must match all input names. DType and Shape must match.
              Provide at least {{ lab.config.validation.samples }} files for
              validation; no pickle is loaded.
            </p></template
          >
          <div class="lab-step-title">
            <span>02</span><strong>Benchmark conditions</strong>
          </div>
          <div class="lab-form-grid">
            <label
              >Warmup<input
                aria-label="Warmup iterations"
                type="number"
                min="0"
                max="10000"
                v-model.number="lab.config.benchmark.warmup" /></label
            ><label
              >Timed runs<input
                aria-label="Timed iterations"
                type="number"
                min="1"
                max="100000"
                v-model.number="lab.config.benchmark.runs" /></label
            ><label
              >Intra-op threads<input
                aria-label="Intra op threads"
                type="number"
                min="1"
                max="128"
                v-model.number="lab.config.benchmark.intra_op_threads" /></label
            ><label
              >Inter-op threads<input
                aria-label="Inter op threads"
                type="number"
                min="1"
                max="128"
                v-model.number="lab.config.benchmark.inter_op_threads"
            /></label>
          </div>
          <label
            >Both benchmark sessions · graph optimization<select
              aria-label="Benchmark session optimization"
              v-model="lab.config.benchmark.session_optimization"
            >
              <option
                v-for="level in lab.backends.find(
                  (b) => b.id === 'onnxruntime.cpu',
                )?.levels"
                :key="level"
                :value="level"
              >
                {{ level }}
              </option>
            </select></label
          >
          <p class="muted small">
            Disable isolates the saved graph transformation. This setting is
            identical on both benchmark sides.
          </p>
          <div class="lab-form-grid">
            <label
              >Declared batch (optional)<input
                aria-label="Declared batch size"
                type="number"
                min="1"
                v-model.number="lab.config.benchmark.batch_size"
                placeholder="Unspecified" /></label
            ><label
              >Batch input · axis 0<select
                aria-label="Batch input"
                v-model="lab.config.benchmark.batch_input"
              >
                <option :value="null">Unspecified</option>
                <option
                  v-for="input in lab.source?.inputs"
                  :key="input.name"
                  :value="input.name"
                >
                  {{ input.name }}
                </option>
              </select></label
            >
          </div>
          <button
            class="lab-baseline-button"
            :disabled="!canStart"
            @click="action(() => start('baseline'))"
          >
            <Play :size="13" />Run baseline only
          </button>
          <div class="lab-step-title">
            <span>03</span><strong>Optimize & validate</strong>
          </div>
          <label
            >Optimization backend<select
              aria-label="Optimization backend"
              v-model="lab.config.optimization.backend_id"
            >
              <option
                v-for="backend in lab.backends"
                :key="backend.id"
                :value="backend.id"
                :disabled="!backend.implemented"
              >
                {{ backend.name }}
                {{ backend.available ? backend.version : "· unavailable" }}
              </option>
            </select></label
          >
          <p class="muted small" v-if="capability">
            {{ capability.note || capability.reason }}
          </p>
          <OliveWorkflow
            v-if="lab.config.optimization.backend_id === 'microsoft.olive'"
            :prepare-inputs="prepareInputs"
          />
          <DatasetPanel
            v-if="lab.source"
            :prepare-inputs="() => prepareInputs(false)"
            :desktop="desktop"
          />
          <div class="lab-form-grid">
            <label
              v-if="lab.config.optimization.backend_id === 'onnxruntime.cpu'"
              >Artifact optimization level<select
                aria-label="Artifact optimization level"
                v-model="lab.config.optimization.level"
              >
                <option
                  v-for="level in capability?.levels"
                  :key="level"
                  :value="level"
                >
                  {{ level }}
                </option>
              </select></label
            ><label
              >Target device<select
                aria-label="Optimization device"
                v-model="lab.config.optimization.device"
              >
                <option v-for="device in capability?.devices" :key="device">
                  {{ device }}
                </option>
              </select></label
            >
          </div>
          <label
            >Input / output precision<select
              aria-label="Optimization precision"
              v-model="lab.config.optimization.precision"
            >
              <option
                v-for="precision in capability?.precisions"
                :key="precision"
              >
                {{ precision }}
              </option>
            </select></label
          >
          <div class="lab-form-grid">
            <label
              >Validation samples<input
                aria-label="Validation samples"
                type="number"
                min="2"
                max="16"
                v-model.number="lab.config.validation.samples" /></label
            ><label
              >Timeout (seconds)<input
                aria-label="Lab timeout"
                type="number"
                min="3"
                max="1800"
                v-model.number="lab.config.timeout" /></label
            ><label
              >Absolute tolerance<input
                aria-label="Absolute tolerance"
                type="number"
                min="0"
                step="0.00001"
                v-model.number="lab.config.validation.atol" /></label
            ><label
              >Relative tolerance<input
                aria-label="Relative tolerance"
                type="number"
                min="0"
                step="0.0001"
                v-model.number="lab.config.validation.rtol"
            /></label>
          </div>
          <p class="muted small">
            Output: project/.opweaver/optimizations/&lt;experiment&gt; →
            registered as a new ModelArtifact. Originals are retained.
          </p>
          <button
            class="primary full-width"
            :disabled="!canStart"
            @click="action(() => start('optimize'))"
          >
            <FlaskConical :size="14" />Run optimization experiment
          </button>
        </fieldset>
        <button
          v-if="lab.running"
          class="danger full-width"
          @click="action(lab.stop)"
        >
          <Square :size="13" />Cancel experiment
        </button>
        <details class="lab-diagnostics">
          <summary>Backend availability</summary>
          <p v-for="backend in lab.backends" :key="backend.id">
            <strong>{{ backend.name }}</strong> ·
            {{
              backend.available
                ? `Available ${backend.version}`
                : backend.reason
            }}<br />{{ backend.dependencies.join(", ") }}
          </p>
        </details>
      </aside>
      <main class="lab-results">
        <div class="lab-history">
          <label
            >Experiment history<select
              aria-label="Experiment history"
              :value="result?.id ?? ''"
              :disabled="disabled"
              @change="
                action(() =>
                  lab.open(($event.target as HTMLSelectElement).value),
                )
              "
            >
              <option value="" disabled>Select saved experiment</option>
              <option
                v-for="record in lab.records"
                :key="record.id"
                :value="record.id"
              >
                {{ new Date(record.created_at * 1000).toLocaleString() }} ·
                {{ record.status }} · {{ record.id.slice(0, 8) }}
              </option>
            </select></label
          ><button
            :disabled="!result || disabled"
            @click="action(exportReport)"
          >
            <Download :size="14" />Export model & reports
          </button>
        </div>
        <label v-if="!desktop && result"
          >Export parent directory<input
            aria-label="Experiment export directory"
            v-model="exportDirectory"
            placeholder="Existing directory for JSON, Markdown, model and input bundle"
        /></label>
        <p v-if="lab.exportResult" class="lab-exported" role="status">
          Exported: {{ lab.exportResult }}
        </p>
        <div v-if="lab.error" class="lab-error" role="alert">
          <AlertTriangle :size="16" /><span>{{ lab.error }}</span>
        </div>
        <p v-if="lab.running && !lab.job" class="lab-progress" role="status">
          Checking workflow configuration. Cancellation requests will stop the
          experiment as soon as this check returns.
        </p>
        <div v-if="lab.running && lab.job" class="lab-progress" role="status">
          <strong
            >{{ lab.job.status }} · {{ lab.job.phase ?? "environment" }}</strong
          ><span>{{
            lab.job.progress == null
              ? "Running · see raw logs"
              : `${Math.round(lab.job.progress * 100)}% of phase`
          }}</span
          ><progress
            :value="lab.job.progress == null ? undefined : lab.job.progress"
            max="1"
          />
          <div class="lab-phase-list">
            <span v-for="(state, phase) in lab.job.phases" :key="phase"
              >{{ phase }} · {{ state }}</span
            >
          </div>
          <pre>{{ lab.job.logs }}</pre>
        </div>
        <template v-if="result">
          <div class="lab-result-title">
            <span class="eyebrow">EXPERIMENT {{ result.id.slice(0, 8) }}</span
            ><span :class="['lab-status', result.status]">{{
              result.status
            }}</span
            ><span class="muted small"
              >{{ result.config.mode }} ·
              {{
                result.config.mode === "baseline"
                  ? "onnxruntime.cpu"
                  : result.config.optimization.backend_id
              }}
              ·
              {{
                result.optimization?.backend_version ??
                result.environment.onnxruntime
              }}</span
            >
          </div>
          <div
            v-if="result.quantization"
            class="quant-quality-summary"
            aria-label="Quantization quality summary"
          >
            <strong
              >{{ result.quantization.representation }} ·
              {{ (result.quantization.integer_weights as any[])?.length ?? 0 }}
              integer weight tensors</strong
            >
            <div class="quant-quality-badges">
              <span
                >Execution
                {{
                  result.validation?.execution_valid
                    ? "valid"
                    : "invalid / not evaluated"
                }}</span
              ><span
                >Allclose
                {{
                  result.validation?.passed ? "passed" : "outside tolerance"
                }}</span
              ><span
                :class="{
                  'quality-failed':
                    result.validation?.quality_gate?.status === 'failed',
                }"
                >Quality gate
                {{
                  result.validation?.quality_gate?.status ?? "not configured"
                }}</span
              >
            </div>
            <p>
              MAE
              {{
                metric(result.validation?.numerical_summary?.mean_abs_error, 7)
              }}
              · Max error
              {{
                metric(result.validation?.numerical_summary?.max_abs_error, 7)
              }}
              · Changed
              {{
                metric(
                  Number(
                    result.validation?.numerical_summary?.changed_fraction ?? 0,
                  ) * 100,
                  2,
                )
              }}%
            </p>
            <p>
              {{
                result.datasets?.independent
                  ? "Independent evaluation"
                  : "Evaluation independence not established"
              }}
              · Calibration
              {{ (result.datasets?.calibration as any)?.batch_count }} batches ·
              Evaluation
              {{ (result.datasets?.evaluation as any)?.batch_count }} batches
            </p>
            <p>
              Task:
              {{ result.validation?.task_quality?.status ?? "not_evaluated"
              }}<template
                v-if="result.validation?.task_quality?.status === 'evaluated'"
              >
                · Top1
                {{
                  metric(
                    Number(result.validation.task_quality.baseline_accuracy) *
                      100,
                    2,
                  )
                }}% →
                {{
                  metric(
                    Number(result.validation.task_quality.optimized_accuracy) *
                      100,
                    2,
                  )
                }}% · drop
                {{
                  metric(
                    Number(result.validation.task_quality.accuracy_drop) * 100,
                    2,
                  )
                }}
                pp</template
              >
            </p>
            <details>
              <summary>Quantization, datasets & quality evidence</summary>
              <pre>{{
                JSON.stringify(
                  {
                    quantization: result.quantization,
                    datasets: result.datasets,
                    quality: result.validation,
                    data_availability: result.data_availability,
                  },
                  null,
                  2,
                )
              }}</pre>
            </details>
          </div>
          <p
            v-for="(state, role) in result.data_availability ?? {}"
            :key="role"
            v-show="!(state as any).available"
            class="dataset-reference-warning"
          >
            {{ role }} data unavailable: {{ (state as any).reason }}
          </p>
          <TaskQualityCard
            :single="result.baseline_evaluation"
            :paired="result.validation?.task_quality"
            :final="result.final_evaluation"
          />
          <nav class="lab-result-tabs">
            <button
              v-for="name in [
                'Results',
                'Validation',
                'Graph changes',
                'Configuration & logs',
              ]"
              :key="name"
              :class="{ active: tab === name }"
              @click="tab = name"
            >
              {{ name }}
            </button>
          </nav>
          <template v-if="tab === 'Results'">
            <p
              v-if="result.workflow_execution"
              class="lab-workflow-observation"
            >
              {{
                (result.workflow_execution.candidates as any[])?.some(
                  (c) => c.observed_graph_change,
                )
                  ? "Observed ONNX graph change"
                  : "No applicable graph transformation observed"
              }}
              ·
              {{
                (result.workflow_execution.executed_passes as any[])?.length ??
                0
              }}
              passes recorded by Olive. Performance is reported from the
              measurements below.
            </p>
            <div class="lab-metric-cards">
              <div
                :class="{ regression: comparison?.outcome === 'Regression' }"
              >
                <span>Latency speedup</span
                ><strong>{{
                  comparison?.speedup == null
                    ? "—"
                    : `${metric(comparison.speedup, 3)}×`
                }}</strong
                ><small>{{ comparison?.outcome ?? "Baseline only" }}</small>
              </div>
              <div>
                <span>Latency change</span
                ><strong>{{
                  percent(comparison?.latency_change_percent)
                }}</strong
                ><small>Optimized vs baseline</small>
              </div>
              <div>
                <span>Model size change</span
                ><strong>{{ percent(comparison?.size_change_percent) }}</strong
                ><small>Includes external weights</small>
              </div>
            </div>
            <p v-if="comparison" class="muted small">{{ comparison.reason }}</p>
            <table class="lab-table">
              <thead>
                <tr>
                  <th>Measured metric</th>
                  <th>Baseline</th>
                  <th>Optimized</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="[label, key, unit] in rows" :key="key">
                  <td>{{ label }}</td>
                  <td>{{ display("baseline", key, unit) }}</td>
                  <td>{{ display("optimized", key, unit) }}</td>
                </tr>
                <tr>
                  <td>Numerical consistency</td>
                  <td>Reference</td>
                  <td>
                    {{
                      result.validation
                        ? result.validation.passed
                          ? "Passed on recorded samples"
                          : result.quantization
                            ? "Outside tolerance · see measured quality"
                            : "Failed"
                        : "Not performed"
                    }}
                  </td>
                </tr>
              </tbody>
            </table>
            <p class="muted small">
              Batch
              {{
                result.config.benchmark.batch_size ??
                "unspecified (samples/s unavailable)"
              }}
              · concurrency 1 · {{ result.config.benchmark.provider }} · session
              optimization {{ result.config.benchmark.session_optimization }} ·
              throughput change
              {{ percent(comparison?.throughput_change_percent) }}
            </p>
            <LatencyChart
              v-if="result.baseline"
              :baseline="result.baseline.latencies_ms"
              :optimized="result.optimized?.latencies_ms"
            />
            <label class="lab-compare-picker"
              >Compare saved experiments<select
                aria-label="Compare with experiment"
                :disabled="disabled"
                :value="crossCompare?.reference_id ?? ''"
                @change="
                  action(() =>
                    compareExperiment(
                      ($event.target as HTMLSelectElement).value,
                    ),
                  )
                "
              >
                <option value="">Select reference experiment</option>
                <option
                  v-for="record in lab.records.filter(
                    (r) => r.id !== result?.id,
                  )"
                  :key="record.id"
                  :value="record.id"
                >
                  {{ record.id.slice(0, 8) }} · {{ record.status }} ·
                  {{ new Date(record.created_at * 1000).toLocaleString() }}
                </option>
              </select></label
            >
            <div v-if="crossCompare" class="lab-cross-comparison">
              <strong
                >{{ crossCompare.comparison.outcome }} ·
                {{ metric(crossCompare.comparison.speedup, 3) }}×</strong
              >
              <p class="muted small">{{ crossCompare.comparison.reason }}</p>
              <table class="lab-table">
                <thead>
                  <tr>
                    <th>Saved measured result</th>
                    <th>
                      Reference {{ crossCompare.reference_id.slice(0, 8) }}
                    </th>
                    <th>Current {{ result.id.slice(0, 8) }}</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>ModelArtifact</td>
                    <td>
                      {{ crossCompare.reference.model_id?.slice(0, 8) ?? "—" }}
                    </td>
                    <td>
                      {{ crossCompare.candidate.model_id?.slice(0, 8) ?? "—" }}
                    </td>
                  </tr>
                  <tr>
                    <td>Mean ms</td>
                    <td>{{ metric(crossCompare.reference.mean_ms) }}</td>
                    <td>{{ metric(crossCompare.candidate.mean_ms) }}</td>
                  </tr>
                  <tr>
                    <td>P95 ms</td>
                    <td>{{ metric(crossCompare.reference.p95_ms) }}</td>
                    <td>{{ metric(crossCompare.candidate.p95_ms) }}</td>
                  </tr>
                  <tr>
                    <td>requests/s</td>
                    <td>
                      {{ metric(crossCompare.reference.requests_per_second) }}
                    </td>
                    <td>
                      {{ metric(crossCompare.candidate.requests_per_second) }}
                    </td>
                  </tr>
                </tbody>
              </table>
              <p class="muted small">
                Uses each experiment's optimized benchmark when present,
                otherwise its baseline. Comparability checks recorded
                conditions; measurements from separate runs remain subject to
                noise.
              </p>
            </div>
          </template>
          <template v-else-if="tab === 'Validation'"
            ><div class="lab-validation-heading">
              <CheckCircle2
                v-if="result.validation?.passed"
                :size="21"
              /><AlertTriangle v-else :size="21" />
              <div>
                <h2>
                  {{
                    result.validation
                      ? result.validation.passed
                        ? "Numerical consistency passed"
                        : "Numerical consistency failed"
                      : "Validation not performed"
                  }}
                </h2>
                <p>
                  {{
                    result.validation?.scope ??
                    "Baseline-only runs do not compare outputs."
                  }}
                </p>
              </div>
            </div>
            <p class="muted small">
              {{ result.validation?.rule }} · atol
              {{ result.validation?.atol }} · rtol
              {{ result.validation?.rtol }} ·
              {{ result.validation?.samples ?? 0 }} samples
            </p>
            <div class="lab-table-scroll">
              <table class="lab-table">
                <thead>
                  <tr>
                    <th>Sample / output</th>
                    <th>Type / Shape</th>
                    <th>Max abs</th>
                    <th>Mean abs</th>
                    <th>Max rel</th>
                    <th>Result</th>
                  </tr>
                </thead>
                <tbody>
                  <tr
                    v-for="(row, i) in validationRows"
                    :key="i"
                    :title="row.reason"
                  >
                    <td>{{ row.sample }} / {{ row.name }}</td>
                    <td>
                      {{ row.dtype }}
                      {{
                        shapeText({
                          name: row.name ?? "",
                          dtype: row.dtype ?? "",
                          shape: row.shape ?? null,
                        })
                      }}
                    </td>
                    <td>{{ metric(row.max_abs_error, 8) }}</td>
                    <td>{{ metric(row.mean_abs_error, 8) }}</td>
                    <td>{{ metric(row.max_relative_error, 8) }}</td>
                    <td>
                      {{ row.passed ? "Passed" : "Failed" }}
                      {{ row.comparison?.includes("exact") ? "(exact)" : "" }}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
            <details>
              <summary>Full per-output diagnostics</summary>
              <pre>{{ JSON.stringify(result.validation, null, 2) }}</pre>
            </details></template
          >
          <template v-else-if="tab === 'Graph changes'"
            ><div class="lab-graph-links">
              <GitCompareArrows :size="22" />
              <div>
                <h2>Inspect the saved model graphs</h2>
                <p>Uses the P0-B Graph Canvas and Model Inspector.</p>
              </div>
              <button @click="emit('viewModel', result.source_model_id)">
                View original graph <ArrowUpRight :size="13" /></button
              ><button
                :disabled="!result.optimization"
                @click="
                  result.optimization &&
                  emit('viewModel', result.optimization.artifact.id)
                "
              >
                View optimized graph <ArrowUpRight :size="13" />
              </button>
            </div>
            <template v-if="graphComparison"
              ><table class="lab-table">
                <thead>
                  <tr>
                    <th>Structure</th>
                    <th>Original</th>
                    <th>Optimized</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>ONNX operator nodes</td>
                    <td>{{ graphComparison.original_nodes }}</td>
                    <td>{{ graphComparison.optimized_nodes }}</td>
                  </tr>
                  <tr>
                    <td>Model bytes</td>
                    <td>{{ bytesText(graphComparison.original_bytes) }}</td>
                    <td>{{ bytesText(graphComparison.optimized_bytes) }}</td>
                  </tr>
                  <tr
                    v-for="op in [
                      ...new Set([
                        ...Object.keys(graphComparison.original_operators),
                        ...Object.keys(graphComparison.optimized_operators),
                      ]),
                    ]"
                    :key="op"
                  >
                    <td>{{ op }}</td>
                    <td>{{ graphComparison.original_operators[op] ?? 0 }}</td>
                    <td>{{ graphComparison.optimized_operators[op] ?? 0 }}</td>
                  </tr>
                </tbody>
              </table>
              <p class="muted small">{{ graphComparison.scope }}</p>
              <p class="muted small">
                {{ result.optimization?.note }}
              </p></template
            ></template
          >
          <template v-else
            ><details v-if="result.workflow_execution" open>
              <summary>
                Olive workflow · requested, executed and observed
              </summary>
              <p class="muted small">
                Executed passes can leave the graph unchanged. Candidate
                signatures record observed graph changes; they do not establish
                a speedup.
              </p>
              <pre>{{
                JSON.stringify(result.workflow_execution, null, 2)
              }}</pre>
            </details>
            <details open>
              <summary>Saved configuration</summary>
              <pre>{{ JSON.stringify(result.config, null, 2) }}</pre>
            </details>
            <details>
              <summary>Environment and artifact target</summary>
              <pre>{{
                JSON.stringify(
                  {
                    environment: result.environment,
                    artifact: result.optimization?.artifact,
                  },
                  null,
                  2,
                )
              }}</pre>
            </details>
            <details>
              <summary>Input sample snapshots</summary>
              <pre>{{ JSON.stringify(result.input_samples, null, 2) }}</pre>
            </details>
            <details open>
              <summary>Phase states and execution log</summary>
              <pre
                >{{ JSON.stringify(result.phases, null, 2) }}
{{ result.logs.join("\n") }}</pre
              >
            </details></template
          >
          <div class="lab-scope-note">
            <AlertTriangle :size="14" /><span
              >Numerical consistency covers the saved samples. Task quality is
              reported only for explicitly configured evaluators. GPU
              performance and memory usage are not measured.</span
            >
          </div>
        </template>
        <div v-else-if="!lab.running" class="lab-awaiting">
          <GitCompareArrows :size="34" />
          <h2>A reproducible experiment starts here.</h2>
          <p>
            Run a baseline, then create and verify an ORT optimized model. Every
            latency is sampled from real inference.
          </p>
          <div>
            <span>Baseline</span><span>Optimize</span><span>Validate</span
            ><span>Compare</span>
          </div>
        </div>
      </main>
    </div>
  </section>
</template>
<style scoped>
.lab-mode-tabs {
  display: flex;
  gap: 6px;
  padding: 7px 20px;
  border-bottom: 1px solid #293747;
  background: #141d2b;
}
.lab-mode-tabs button {
  font-size: 11px;
  color: #91a6bd;
}
.lab-mode-tabs button.active {
  background: #29374d;
  color: #c3dbe9;
  border-color: #547189;
}
.quant-quality-summary {
  margin: 12px 0;
  padding: 15px 18px;
  border: 1px solid #35645d;
  border-radius: 10px;
  background: linear-gradient(120deg, #193235, #1c2639);
}
.quant-quality-summary p {
  font-size: 12px;
  color: #b9c9d5;
  margin: 8px 0;
}
.quant-quality-badges {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 12px;
}
.quant-quality-badges span {
  border: 1px solid #42675f;
  border-radius: 5px;
  padding: 4px 8px;
  font-size: 11px;
  background: #203c36;
  color: #a7ddcd;
}
.quant-quality-badges .quality-failed {
  background: #453327;
  color: #ffd0a3;
  border-color: #73523b;
}
.quant-quality-summary pre {
  font-size: 11px;
  max-height: 360px;
  overflow: auto;
}
</style>
