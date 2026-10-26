<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from "vue";
import {
  Activity,
  Play,
  Square,
  Download,
  RefreshCw,
  ArrowUpRight,
  Layers,
} from "lucide-vue-next";
import { rpc } from "../api";
import { useEditor } from "../stores/editor";
import { useModels } from "../stores/models";
import { useLab } from "../stores/lab";
import { useDiagnostics } from "../stores/diagnostics";
import {
  errorLevel,
  errorThresholds,
  rankDiagnosticRows,
  type DiagnosticRecord,
  type DiagnosticReport,
} from "../diagnostic-types";
import type { OnnxReport } from "../model-types";
import DiagnosticDetails from "./DiagnosticDetails.vue";
const props = defineProps<{ desktop?: boolean }>();
const emit = defineEmits<{
  viewNode: [
    report: DiagnosticReport,
    row: DiagnosticRecord,
    side: "fp32" | "quantized",
    low: number | null,
    high: number | null,
  ];
}>();
const editor = useEditor(),
  models = useModels(),
  lab = useLab(),
  diag = useDiagnostics();
const datasets = ref<any[]>([]),
  original = ref<OnnxReport | null>(null),
  busy = ref(false),
  exportDirectory = ref("");
const kind = ref("activation"),
  metric = ref("mae"),
  status = ref(""),
  operator = ref(""),
  search = ref(""),
  topN = ref(20),
  selectedId = ref("");
const lowOverride = ref<string>(""),
  highOverride = ref<string>("");
const choices = computed(() =>
  models.completed.filter((v) => v.artifact?.format === "onnx"),
);
const auto = computed(() => errorThresholds(diag.report?.records ?? []));
const low = computed(() =>
  lowOverride.value === "" ? auto.value.low : Number(lowOverride.value),
);
const high = computed(() =>
  highOverride.value === "" ? auto.value.high : Number(highOverride.value),
);
const thresholdError = computed(() =>
  low.value != null &&
  high.value != null &&
  (!Number.isFinite(low.value) ||
    !Number.isFinite(high.value) ||
    low.value < 0 ||
    high.value < low.value)
    ? "Thresholds must be finite, nonnegative, and low ≤ high. Graph colors are unavailable until corrected."
    : "",
);
const rows = computed(() =>
  rankDiagnosticRows(diag.report?.records ?? [], {
    kind: kind.value,
    metric: metric.value,
    status: status.value,
    operator: operator.value,
    search: search.value,
    topN: topN.value,
  }),
);
const selected = computed(
  () => rows.value.find((v) => v.id === selectedId.value) ?? rows.value[0],
);
const types = computed(() =>
  [...new Set((diag.report?.records ?? []).map((v) => v.operator_type))].sort(),
);
const operatorText = computed({
  get: () => diag.config.operator_types.join(", "),
  set: (s: string) =>
    (diag.config.operator_types = [
      ...new Set(
        s
          .split(",")
          .map((v) => v.trim())
          .filter(Boolean),
      ),
    ]),
});
const disabled = computed(() => busy.value || diag.running);
const canStart = computed(
  () =>
    !disabled.value &&
    !lab.running &&
    !models.running &&
    !editor.running &&
    !!diag.config.fp32_model_id &&
    !!diag.config.quantized_model_id &&
    !!diag.config.dataset_id,
);
const fmt = (v: unknown) =>
  typeof v === "number" && Number.isFinite(v)
    ? Number(v.toPrecision(6)).toString()
    : "—";
async function action(fn: () => Promise<unknown>) {
  busy.value = true;
  try {
    await fn();
  } catch (e) {
    diag.error = String(e);
  } finally {
    busy.value = false;
  }
}
let sourceRevision = 0;
async function sourceChanged() {
  const revision = ++sourceRevision,
    id = diag.config.fp32_model_id;
  original.value = null;
  if (!id) return;
  const report = await rpc<OnnxReport>("model.load", {
    project: editor.projectPath,
    id,
  });
  if (revision === sourceRevision && id === diag.config.fp32_model_id)
    original.value = report;
}
watch(
  () => diag.config.fp32_model_id,
  () => void action(sourceChanged),
);
watch(
  () => diag.report?.id,
  () => {
    selectedId.value = "";
    lowOverride.value = "";
    highOverride.value = "";
  },
);
async function refresh() {
  await Promise.all([models.refresh(), diag.refresh()]);
  datasets.value = await rpc<any[]>("lab.dataset.list", {
    project: editor.projectPath,
  });
  if (!diag.config.fp32_model_id)
    diag.config.fp32_model_id =
      lab.source?.artifact.id ??
      choices.value.find((v) => !v.artifact?.parent_model_id)?.id ??
      "";
  if (!diag.config.quantized_model_id)
    diag.config.quantized_model_id =
      lab.experiment?.optimization?.artifact.id ??
      choices.value.find(
        (v) => v.artifact?.parent_model_id === diag.config.fp32_model_id,
      )?.id ??
      "";
  if (!diag.config.dataset_id)
    diag.config.dataset_id =
      datasets.value.find(
        (v) => v.role === "diagnostic" && v.availability?.available,
      )?.id ?? "";
  await sourceChanged();
}
function focus(row: DiagnosticRecord, side: "fp32" | "quantized") {
  if (!diag.report) return;
  emit("viewNode", diag.report, row, side, low.value, high.value);
}
async function exportReport() {
  const path = props.desktop
    ? await rpc<string>("dialog.directory")
    : exportDirectory.value;
  if (path) await diag.exportReport(path);
}
let polling = false;
const timer = setInterval(async () => {
  if (polling || !diag.running) return;
  polling = true;
  try {
    await diag.poll();
  } catch (e) {
    diag.error = String(e);
  } finally {
    polling = false;
  }
}, 700);
onMounted(() => void action(refresh));
onUnmounted(() => clearInterval(timer));
</script>
<template>
  <section class="diagnostics-panel" aria-label="Quantization diagnostics">
    <div class="diagnostic-intro">
      <div>
        <span class="eyebrow">MEASURE THE DIVERGENCE</span>
        <h2><Layers :size="22" /> Quantization Diagnostics</h2>
        <p>
          Compare real intermediate activations and constant weights. High error
          requires investigation; upstream error can accumulate.
        </p>
      </div>
      <button
        :disabled="disabled"
        @click="action(refresh)"
        title="Refresh diagnostic resources"
      >
        <RefreshCw :size="14" />Refresh
      </button>
    </div>
    <div class="diagnostic-layout">
      <aside class="diagnostic-config">
        <fieldset :disabled="disabled">
          <h3>Models & diagnostic data</h3>
          <label
            >Original FP32 artifact<select
              aria-label="Diagnostic FP32 model"
              v-model="diag.config.fp32_model_id"
            >
              <option disabled value="">Choose original</option>
              <option v-for="m in choices" :key="m.id" :value="m.id">
                {{ m.artifact?.name }} · {{ m.id.slice(0, 8) }}
              </option>
            </select></label
          >
          <label
            >Quantized artifact<select
              aria-label="Diagnostic quantized model"
              v-model="diag.config.quantized_model_id"
            >
              <option disabled value="">Choose QDQ model</option>
              <option
                v-for="m in choices"
                :key="m.id"
                :value="m.id"
                :disabled="m.id === diag.config.fp32_model_id"
              >
                {{ m.artifact?.name }} · {{ m.id.slice(0, 8) }}
              </option>
            </select></label
          >
          <label
            >Diagnostic dataset<select
              aria-label="Diagnostic dataset"
              v-model="diag.config.dataset_id"
            >
              <option disabled value="">Choose diagnostic split</option>
              <option
                v-for="d in datasets.filter((v) => v.role === 'diagnostic')"
                :key="d.id"
                :value="d.id"
                :disabled="!d.availability?.available"
              >
                {{ d.name }} · {{ d.batch_count }} batches ·
                {{ d.id.slice(0, 8) }}
              </option>
            </select></label
          >
          <p class="note">
            Create diagnostic role NPZ/image snapshots in the Lab dataset panel.
            Final evaluation data is never selected here.
          </p>
          <h3>Capture scope</h3>
          <div class="diagnostic-fields">
            <label
              >Max samples · batches<input
                aria-label="Diagnostic max samples"
                type="number"
                min="1"
                max="16"
                v-model.number="diag.config.max_samples" /></label
            ><label
              >Max capture tensors<input
                aria-label="Diagnostic max tensors"
                type="number"
                min="2"
                max="256"
                v-model.number="diag.config.max_tensors"
            /></label>
          </div>
          <label
            >Selected original nodes · optional<select
              aria-label="Diagnostic selected nodes"
              multiple
              size="5"
              v-model="diag.config.selected_nodes"
            >
              <option
                v-for="n in original?.analysis.nodes.filter(
                  (n) => n.kind === 'onnx_node',
                )"
                :key="n.id"
                :value="n.id"
              >
                {{ n.id }} · {{ n.label }} · {{ n.operator }}
              </option>
            </select></label
          >
          <button @click="diag.config.selected_nodes = []">
            Clear node filter
          </button>
          <label
            >Operator types · optional<input
              aria-label="Diagnostic operator types"
              v-model="operatorText"
              placeholder="Conv, Gemm"
          /></label>
          <p class="note">
            An empty filter selects the first 64 original nodes. Max Tensors
            counts exposed outputs across both models, including QDQ pre/post
            values. One sample is one saved batch; example counts are separate.
          </p>
          <h3>Resources</h3>
          <div class="diagnostic-fields">
            <label
              >Tensor working MiB<input
                aria-label="Diagnostic memory budget"
                type="number"
                min="1"
                max="512"
                v-model.number="diag.config.memory_budget_mb" /></label
            ><label
              >Temporary disk MiB<input
                aria-label="Diagnostic disk budget"
                type="number"
                min="1"
                max="1024"
                v-model.number="diag.config.disk_budget_mb" /></label
            ><label
              >Timeout · seconds<input
                aria-label="Diagnostic timeout"
                type="number"
                min="3"
                max="900"
                v-model.number="diag.config.timeout"
            /></label>
          </div>
          <label class="check"
            ><input
              type="checkbox"
              aria-label="Diagnostic weight errors"
              v-model="diag.config.include_weights"
            />Compare constant weights</label
          >
          <p class="note">
            CPUExecutionProvider · graph optimization disabled · one thread.
            Budget covers capture/statistics tensors, not total process RAM.
            Models ≤128 MiB each. Temporary models are deleted on completion,
            cancel or timeout; raw activations are not retained.
          </p>
          <button
            class="primary run-diagnostic"
            :disabled="!canStart"
            @click="action(() => diag.start(diag.config))"
          >
            <Play :size="14" />Run layer diagnostic
          </button>
        </fieldset>
        <button v-if="diag.running" class="cancel" @click="diag.stop()">
          <Square :size="13" />Cancel diagnostic
        </button>
      </aside>
      <main class="diagnostic-results">
        <div class="diagnostic-history">
          <label
            >Diagnostic history<select
              aria-label="Diagnostic history"
              :disabled="diag.running"
              :value="diag.report?.id ?? ''"
              @change="
                action(() =>
                  diag.open(($event.target as HTMLSelectElement).value),
                )
              "
            >
              <option disabled value="">Choose saved diagnostic</option>
              <option v-for="r in diag.records" :key="r.id" :value="r.id">
                {{ new Date(r.created_at * 1000).toLocaleString() }} ·
                {{ r.status }} · {{ r.id.slice(0, 8) }}
              </option>
            </select></label
          ><button
            :disabled="!diag.report || diag.running"
            @click="action(exportReport)"
          >
            <Download :size="13" />Export diagnostic report
          </button>
        </div>
        <input
          v-if="!desktop"
          aria-label="Diagnostic export directory"
          v-model="exportDirectory"
          placeholder="Existing parent directory for JSON/Markdown"
        />
        <p v-if="diag.exported" class="diagnostic-exported" role="status">
          Exported: {{ diag.exported }}
        </p>
        <p v-if="diag.error" class="diagnostic-error" role="alert">
          {{ diag.error }}
        </p>
        <section
          v-if="diag.running"
          class="diagnostic-progress"
          aria-label="Diagnostic capture progress"
        >
          <strong
            >{{ diag.job?.phase ?? "Starting diagnostic process" }} ·
            {{ Math.round((diag.job?.progress ?? 0) * 100) }}%</strong
          ><progress :value="diag.job?.progress ?? 0" max="1" />
          <pre>{{ diag.job?.logs }}</pre>
        </section>
        <template v-if="diag.report">
          <div class="diagnostic-result-title">
            <strong>DIAGNOSTIC {{ diag.report.id.slice(0, 8) }}</strong
            ><span class="diagnostic-status">{{ diag.report.status }}</span
            ><span
              >ORT {{ diag.report.environment.onnxruntime }} ·
              {{ diag.report.config.provider }}</span
            >
          </div>
          <p
            v-for="(v, k) in diag.report.availability"
            :key="k"
            v-show="!v.available"
            class="diagnostic-error"
          >
            {{ k }} resource unavailable: {{ v.reason }}. Saved measurements are
            retained.
          </p>
          <div class="diagnostic-summary" aria-label="Tensor matching coverage">
            <span
              ><small>Comparable activations</small
              ><b
                >{{ diag.report.matching_summary.comparable_tensors ?? 0 }} /
                {{ diag.report.matching_summary.scope_tensors ?? 0 }}</b
              ></span
            ><span
              ><small>Matching coverage</small
              ><b>{{
                diag.report.matching_summary.coverage == null
                  ? "—"
                  : `${fmt(diag.report.matching_summary.coverage * 100)}%`
              }}</b></span
            ><span
              ><small>Captured batches / examples</small
              ><b
                >{{ diag.report.captures.length }} /
                {{
                  diag.report.captures.reduce((n, v) => n + v.example_count, 0)
                }}</b
              ></span
            >
          </div>
          <p class="note">
            Coverage counts reference output tensors with ≥1 valid feed.
            Partial/unmatched/unsupported positions remain visible; coverage
            does not establish model quality.
          </p>
          <div class="diagnostic-filters">
            <label
              >Kind<select aria-label="Diagnostic error kind" v-model="kind">
                <option value="activation">Activation Error</option>
                <option value="weight">Weight Error</option>
              </select></label
            ><label
              >Rank by<select
                aria-label="Diagnostic ranking metric"
                v-model="metric"
              >
                <option value="mae">MAE · largest first</option>
                <option value="max_abs_error">Max Error · largest first</option>
                <option value="rmse">RMSE · largest first</option>
                <option value="cosine_similarity">Cosine · lowest first</option>
              </select></label
            ><label
              >Status<select
                aria-label="Diagnostic match filter"
                v-model="status"
              >
                <option value="">All statuses</option>
                <option>Matched</option>
                <option>Partially Matched</option>
                <option>Unmatched</option>
                <option>Unsupported</option>
              </select></label
            ><label
              >Operator<select
                aria-label="Diagnostic operator filter"
                v-model="operator"
              >
                <option value="">All operators</option>
                <option v-for="t in types" :key="t">{{ t }}</option>
              </select></label
            ><label
              >Top N<input
                aria-label="Diagnostic top N"
                type="number"
                min="1"
                max="256"
                v-model.number="topN"
            /></label>
          </div>
          <input
            aria-label="Search diagnostic tensors"
            v-model="search"
            placeholder="Search tensor, node name or graph ID"
          />
          <details class="heat-settings">
            <summary>Graph MAE colors · relative to this diagnostic</summary>
            <p>
              Default low/medium boundary is p50={{ fmt(auto.low) }},
              medium/high boundary p90={{ fmt(auto.high) }}. These are
              distribution ranks, not universal quality gates.
            </p>
            <div class="diagnostic-fields">
              <label
                >Low/medium override<input
                  aria-label="Diagnostic low threshold"
                  type="number"
                  min="0"
                  step="any"
                  v-model="lowOverride"
                  placeholder="Auto p50" /></label
              ><label
                >Medium/high override<input
                  aria-label="Diagnostic high threshold"
                  type="number"
                  min="0"
                  step="any"
                  v-model="highOverride"
                  placeholder="Auto p90"
              /></label>
            </div>
            <p v-if="thresholdError" class="diagnostic-error" role="alert">
              {{ thresholdError }}
            </p>
          </details>
          <div class="diagnostic-table-scroll">
            <table class="diagnostic-table" aria-label="Layer error ranking">
              <thead>
                <tr>
                  <th>Original node / Tensor</th>
                  <th>Match</th>
                  <th>MAE</th>
                  <th>RMSE</th>
                  <th>Max</th>
                  <th>Cosine</th>
                  <th>Valid feeds</th>
                  <th>Graph location</th>
                </tr>
              </thead>
              <tbody>
                <tr
                  v-for="r in rows"
                  :key="r.id"
                  :data-record-id="r.id"
                  :class="{ selected: selected?.id === r.id }"
                  @click="selectedId = r.id"
                >
                  <td>
                    <b>{{ r.node_name || r.fp32_node_id }}</b
                    ><code>{{ r.fp32_tensor }}</code
                    ><small>{{ r.operator_type }} · {{ r.fp32_node_id }}</small>
                  </td>
                  <td>
                    <span :class="['error-badge', errorLevel(r, low, high)]">{{
                      r.match_status
                    }}</span
                    ><small
                      v-if="
                        r.kind === 'activation' &&
                        errorLevel(r, low, high) === 'high'
                      "
                      >High Error · investigate</small
                    >
                  </td>
                  <td>{{ fmt(r.aggregate?.mae) }}</td>
                  <td>{{ fmt(r.aggregate?.rmse) }}</td>
                  <td>{{ fmt(r.aggregate?.max_abs_error) }}</td>
                  <td>{{ fmt(r.aggregate?.cosine_similarity) }}</td>
                  <td>
                    {{ r.aggregate?.valid_samples ?? 0 }} /
                    {{ r.aggregate?.samples ?? 0 }}
                  </td>
                  <td>
                    <button
                      :disabled="
                        diag.report.availability?.fp32?.available === false
                      "
                      :aria-label="`Locate FP32 ${r.id}`"
                      @click.stop="focus(r, 'fp32')"
                    >
                      FP32 <ArrowUpRight :size="11" /></button
                    ><button
                      :disabled="
                        !r.quantized_node_id ||
                        diag.report.availability?.quantized?.available === false
                      "
                      :aria-label="`Locate INT8 ${r.id}`"
                      @click.stop="focus(r, 'quantized')"
                    >
                      INT8 <ArrowUpRight :size="11" />
                    </button>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
          <DiagnosticDetails
            v-if="selected"
            :rows="[selected]"
            :diagnostic-id="diag.report.id"
          />
          <details>
            <summary>Configuration, resource limits & capture metadata</summary>
            <pre>{{
              JSON.stringify(
                {
                  config: diag.report.config,
                  resources: diag.report.resources,
                  captures: diag.report.captures,
                  lineage: diag.report.lineage,
                  matching: diag.report.matching_summary,
                },
                null,
                2,
              )
            }}</pre>
          </details>
          <details>
            <summary>Matching limitations, warnings & diagnostic logs</summary>
            <p
              v-for="warning in [
                ...diag.report.limitations,
                ...diag.report.warnings,
              ]"
              :key="warning"
            >
              {{ warning }}
            </p>
            <pre>{{ diag.report.logs.join("\n") }}</pre>
          </details>
          <p class="note">
            Diagnostic elapsed {{ fmt(diag.report.elapsed_seconds) }} s includes
            instrumentation and session creation. Use Benchmark results to
            assess inference performance.
          </p>
        </template>
        <div v-else-if="!diag.running" class="diagnostic-empty">
          <Activity :size="36" />
          <h3>Inspect the actual numerical path.</h3>
          <p>
            Select the original model, a QDQ artifact and diagnostic split to
            collect real tensors. Missing matches will remain explicit.
          </p>
        </div>
      </main>
    </div>
  </section>
</template>
<style scoped>
.diagnostics-panel {
  min-height: 0;
  flex: 1;
  overflow: auto;
  background: #111923;
}
.diagnostic-intro {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 22px 26px;
  border-bottom: 1px solid #2a3b4d;
  background: linear-gradient(120deg, #1c2b3e, #16202c);
}
h2 {
  display: flex;
  gap: 10px;
  align-items: center;
  margin: 9px 0;
  font-size: 21px;
}
h3 {
  font-size: 12px;
  margin: 18px 0 10px;
  color: #c5d6e8;
}
.diagnostic-intro p {
  font-size: 12px;
  color: #9dafc3;
  margin: 7px 0;
}
.diagnostic-layout {
  display: grid;
  grid-template-columns: 290px minmax(0, 1fr);
  min-height: 600px;
}
.diagnostic-config {
  padding: 18px;
  border-right: 1px solid #293847;
  background: #151e2a;
}
.diagnostic-config fieldset {
  border: 0;
  padding: 0;
  margin: 0;
  min-width: 0;
}
.diagnostics-panel label {
  display: flex;
  flex-direction: column;
  gap: 7px;
  font-size: 10px;
  color: #99b0c6;
  margin: 10px 0;
}
.diagnostics-panel input,
.diagnostics-panel select {
  width: 100%;
  min-width: 0;
  font-size: 11px;
  padding: 8px;
  background: #0e1721;
  border: 1px solid #354456;
  border-radius: 6px;
  color: #d2dcea;
}
.diagnostic-fields {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}
.diagnostic-config .check {
  flex-direction: row;
  align-items: center;
  gap: 8px;
}
.check input {
  width: 14px;
}
.note {
  font-size: 10px;
  line-height: 1.65;
  color: #899caf;
}
.run-diagnostic {
  width: 100%;
  justify-content: center;
  margin-top: 12px;
}
.cancel {
  margin-top: 8px;
  color: #efa2a6;
}
.diagnostic-results {
  padding: 20px;
  min-width: 0;
}
.diagnostic-history {
  display: flex;
  gap: 12px;
  align-items: center;
}
.diagnostic-history label {
  flex: 1;
  min-width: 0;
}
.diagnostic-result-title {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
  font-size: 11px;
  margin: 20px 0;
}
.diagnostic-result-title > span:last-child {
  color: #91a6b9;
}
.diagnostic-status {
  padding: 4px 8px;
  border: 1px solid #527877;
  border-radius: 5px;
  color: #a5ddce;
}
.diagnostic-summary {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
}
.diagnostic-summary span {
  padding: 15px;
  background: linear-gradient(135deg, #1c3440, #1c2638);
  border: 1px solid #345667;
  border-radius: 8px;
  display: flex;
  flex-direction: column;
  gap: 9px;
}
.diagnostic-summary small {
  font-size: 10px;
  color: #9cb9c8;
}
.diagnostic-summary b {
  font-size: 21px;
  color: #bce5e6;
}
.diagnostic-filters {
  display: grid;
  grid-template-columns: 1fr 1.4fr 1fr 1fr 70px;
  gap: 9px;
  margin-top: 18px;
}
.diagnostic-table-scroll {
  overflow: auto;
  margin-top: 14px;
}
.diagnostic-table {
  border-collapse: collapse;
  width: 100%;
  font-size: 11px;
  white-space: nowrap;
}
.diagnostic-table th {
  padding: 10px;
  background: #213044;
  text-align: left;
  color: #afc6da;
  font-size: 10px;
}
.diagnostic-table td {
  padding: 11px 9px;
  border-bottom: 1px solid #2b3949;
  vertical-align: top;
}
.diagnostic-table td:first-child {
  min-width: 150px;
}
.diagnostic-table code,
.diagnostic-table small {
  display: block;
  max-width: 250px;
  overflow: hidden;
  text-overflow: ellipsis;
  color: #89a2b8;
  font-size: 10px;
  margin-top: 4px;
}
.diagnostic-table tr {
  cursor: pointer;
}
.diagnostic-table tr.selected {
  background: #1b2d3c;
}
.diagnostic-table button {
  padding: 5px;
  font-size: 10px;
  margin-right: 4px;
}
.error-badge {
  display: inline-block;
  padding: 4px 7px;
  border-radius: 4px;
  background: #203b36;
  color: #9dd5c1;
}
.error-badge.medium {
  background: #3a3528;
  color: #dfc28e;
}
.error-badge.high {
  background: #432d37;
  color: #e9a4b6;
}
.error-badge.unavailable {
  background: #303645;
  color: #b5bdd0;
}
.diagnostic-progress {
  padding: 15px;
  background: #233043;
  border: 1px solid #425a77;
  border-radius: 8px;
}
.diagnostic-progress progress {
  width: 100%;
  margin-top: 10px;
}
.diagnostic-progress pre {
  max-height: 180px;
}
.diagnostics-panel pre {
  white-space: pre-wrap;
  overflow: auto;
  font-size: 10px;
  max-height: 320px;
}
.diagnostics-panel details {
  margin-top: 16px;
  font-size: 11px;
  line-height: 1.6;
  color: #91a9c0;
}
.diagnostic-error {
  padding: 12px;
  background: #3c2830;
  border: 1px solid #76505b;
  color: #f0a8b0;
  font-size: 11px;
  border-radius: 6px;
  overflow-wrap: anywhere;
}
.diagnostic-exported {
  padding: 10px;
  color: #9bdac0;
  background: #1b3530;
  border-radius: 6px;
  font-size: 11px;
  overflow-wrap: anywhere;
}
.diagnostic-empty {
  text-align: center;
  padding: 80px 20px;
  color: #93a9bc;
}
.diagnostic-empty p {
  font-size: 12px;
}
.heat-settings p {
  font-size: 10px;
}
@media (max-width: 1100px) {
  .diagnostic-layout {
    grid-template-columns: 255px minmax(0, 1fr);
  }
  .diagnostic-results {
    padding: 14px;
  }
  .diagnostic-filters {
    grid-template-columns: 1fr 1fr 1fr;
  }
  .diagnostic-summary b {
    font-size: 17px;
  }
  .diagnostic-history {
    flex-wrap: wrap;
  }
}
</style>
