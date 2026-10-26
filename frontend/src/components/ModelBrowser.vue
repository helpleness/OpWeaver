<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch, nextTick } from "vue";
import {
  Box,
  Upload,
  Search,
  RotateCcw,
  Square,
  FolderOpen,
  X,
  Layers,
  FileCode2,
} from "lucide-vue-next";
import GraphCanvas from "./GraphCanvas.vue";
import { useModels } from "../stores/models";
import { useEditor } from "../stores/editor";
import { groupAnalysis, relatedNodes } from "../analysis-view";
import { bytesText } from "../model-types";
import { rpc } from "../api";
const props = defineProps<{ desktop?: boolean }>();
const emit = defineEmits<{ openPython: [] }>();
const models = useModels(),
  editor = useEditor();
const dialog = ref(false),
  source = ref(""),
  busy = ref(false),
  probe = ref<any>(null),
  importers = ref<any[]>([]);
const entry = ref(""),
  constructor = ref("{}"),
  inputs = ref('[{"shape":[4,8],"dtype":"float32"}]');
const search = ref(""),
  grouping = ref<"none" | "operator" | "module">("none"),
  direction = ref<"none" | "upstream" | "downstream" | "both">("both");
const subgraph = computed({
  get: () => models.subgraphKey,
  set: (value: string) => (models.subgraphKey = value),
});
const canvas = ref<InstanceType<typeof GraphCanvas>>();
const original = computed(() =>
  subgraph.value
    ? (models.report?.subgraphs[subgraph.value] ?? null)
    : (models.report?.analysis ?? null),
);
const projected = computed(() =>
  original.value ? groupAnalysis(original.value, grouping.value) : null,
);
const matches = computed(() =>
  search.value
    ? (original.value?.nodes
        .filter((n) =>
          `${n.label} ${n.operator} ${n.id}`
            .toLowerCase()
            .includes(search.value.toLowerCase()),
        )
        .slice(0, 40) ?? [])
    : [],
);
const selectedGroup = computed(() =>
  projected.value?.nodes.find((n) => n.id === models.selectedNode),
);
const groupMembers = computed(
  () => selectedGroup.value?.details.group_members as string[] | undefined,
);
const highlighted = computed(() =>
  projected.value
    ? relatedNodes(projected.value, models.selectedNode, direction.value)
    : [],
);
async function action(fn: () => Promise<unknown>) {
  busy.value = true;
  try {
    await fn();
  } catch (e) {
    models.error = String(e);
  } finally {
    busy.value = false;
  }
}
async function inspect() {
  probe.value = await rpc<any>("model.probe", { source: source.value });
  entry.value =
    probe.value.entries?.find((v: any) => v.kind === "class")?.name ??
    probe.value.entries?.[0]?.name ??
    "";
}
async function start() {
  if (!probe.value)
    throw new Error("Detect the model format before importing.");
  if (probe.value.requires_execution && !entry.value)
    throw new Error("Select a model class or factory.");
  await models.start(
    source.value,
    probe.value.requires_execution
      ? {
          entry: entry.value,
          constructor: JSON.parse(constructor.value),
          inputs: JSON.parse(inputs.value),
          execute_confirmed: true,
        }
      : {},
  );
  dialog.value = false;
}
async function select(id: string) {
  if (grouping.value !== "none") grouping.value = "none";
  models.selectedNode = id;
  search.value = "";
  await nextTick();
  canvas.value?.focus(id);
}
async function focusDiagnosticNode() {
  const request = models.focusRequest;
  if (!request) return;
  if (
    request?.modelId === models.report?.artifact.id &&
    original.value?.nodes.some((n) => n.id === request.nodeId)
  )
    await select(request.nodeId);
}
watch(
  () => [models.focusRequest?.revision, original.value?.id],
  focusDiagnosticNode,
  { flush: "post" },
);
onMounted(() => void nextTick(focusDiagnosticNode));
watch(
  () => models.report?.artifact.id,
  () => {
    subgraph.value = "";
    search.value = "";
    grouping.value =
      (models.report?.analysis.nodes.length ?? 0) > 250 ? "operator" : "none";
  },
  { immediate: true },
);
watch(subgraph, () => {
  models.selectedNode = null;
  search.value = "";
  grouping.value =
    (original.value?.nodes.length ?? 0) > 250 ? "operator" : "none";
});
watch(source, () => (probe.value = null));
const polling = setInterval(
  () => void models.poll().catch((e) => (models.error = String(e))),
  700,
);
onUnmounted(() => clearInterval(polling));
onMounted(
  () =>
    void action(async () => {
      importers.value = await rpc<any[]>("model.importers");
      await models.refresh();
    }),
);
defineExpose({
  showImport: () => {
    source.value = "";
    probe.value = null;
    dialog.value = true;
  },
});
</script>

<template>
  <section class="model-browser">
    <header class="import-toolbar">
      <div class="import-title">
        <Box :size="16" /><strong>Model analysis</strong
        ><span class="badge">{{
          models.report?.artifact.format.toUpperCase() ?? "LOCAL"
        }}</span>
      </div>
      <select
        aria-label="Imported model"
        :value="models.report?.artifact.id ?? ''"
        :disabled="models.running || models.loading"
        @change="models.open(($event.target as HTMLSelectElement).value)"
      >
        <option disabled value="">Select an imported model</option>
        <option
          v-for="item in models.completed"
          :key="item.id"
          :value="item.id"
        >
          {{ item.artifact?.name }} · {{ item.id.slice(0, 6) }}
        </option>
      </select>
      <button
        title="Re-analyze saved model"
        :disabled="!models.report || models.running"
        @click="
          action(() =>
            models.start(
              `${editor.projectPath}/${models.report!.artifact.file_path}`,
              models.report?.pytorch
                ? {
                    entry: models.report.model.entry,
                    constructor:
                      (models.report.pytorch.capture as any).constructor ?? {},
                    inputs: (models.report.pytorch.capture as any).input_specs,
                    execute_confirmed: true,
                  }
                : {},
            ),
          )
        "
      >
        <RotateCcw :size="14" />
      </button>
      <button
        class="primary"
        :disabled="!editor.projectPath || models.running || busy"
        @click="
          source = '';
          probe = null;
          dialog = true;
        "
      >
        <Upload :size="13" />Import model
      </button>
    </header>
    <div class="import-summary-strip" v-if="models.report">
      <strong>{{ models.report.artifact.name }}</strong
      ><span>{{ bytesText(models.report.artifact.total_file_bytes) }}</span
      ><span>{{ models.report.statistics.nodes }} operators</span
      ><span
        >{{ models.report.inputs.length }} in /
        {{ models.report.outputs.length }} out</span
      ><span class="import-readonly">Read-only · original preserved</span>
    </div>
    <div class="analysis-controls" v-if="models.report">
      <div class="model-node-search">
        <Search :size="13" /><input
          aria-label="Search model nodes"
          v-model="search"
          placeholder="Search name, operator or ID…"
        />
        <div class="model-search-results" v-if="search">
          <button
            v-for="node in matches"
            :key="node.id"
            @click="select(node.id)"
          >
            <strong>{{ node.label }}</strong
            ><code>{{ node.operator }}</code></button
          ><span v-if="!matches.length">No matching nodes</span>
        </div>
      </div>
      <label class="inline-control"
        >Group<select aria-label="Group model nodes" v-model="grouping">
          <option value="none">Expanded</option>
          <option value="operator">By operator</option>
          <option value="module">By module</option>
        </select></label
      >
      <label class="inline-control"
        >Trace<select aria-label="Highlight related nodes" v-model="direction">
          <option value="both">Both directions</option>
          <option value="upstream">Upstream</option>
          <option value="downstream">Downstream</option>
          <option value="none">None</option>
        </select></label
      >
      <select
        v-if="Object.keys(models.report.subgraphs ?? {}).length"
        aria-label="Model subgraph"
        v-model="subgraph"
      >
        <option value="">Main graph</option>
        <option
          v-for="(graph, id) in models.report.subgraphs"
          :value="id"
          :key="id"
        >
          {{ graph.name }} · {{ id }}
        </option>
      </select>
      <button v-if="models.report.pytorch" @click="emit('openPython')">
        <FileCode2 :size="13" />Open Python
      </button>
      <button @click="models.selectedNode = null">Overview</button>
    </div>
    <div v-if="models.running" class="import-progress" role="status">
      <span class="live-dot"></span>Importing and analyzing
      {{ models.job?.file_path.split(/[\\/]/).at(-1) }}…<span
        >No inference is run for ONNX.</span
      ><button @click="action(models.stop)">
        <Square :size="11" />Cancel import
      </button>
    </div>
    <div v-if="models.loading" class="import-progress" role="status">
      Loading saved analysis…
    </div>
    <div v-if="models.error" class="import-error" role="alert">
      <strong>Import unavailable</strong><span>{{ models.error }}</span
      ><small v-if="models.report"
        >The previous valid analysis is retained.</small
      ><button title="Dismiss import error" @click="models.error = ''">
        <X :size="13" />
      </button>
    </div>
    <div class="collapsed-group" v-if="groupMembers?.length">
      <Layers :size="14" /><span
        >{{ groupMembers.length }} nodes in this group</span
      ><button
        @click="
          grouping = 'none';
          models.selectedNode = null;
        "
      >
        Expand all nodes</button
      ><button
        v-for="id in groupMembers.slice(0, 15)"
        :key="id"
        @click="select(id)"
      >
        {{ original?.nodes.find((n) => n.id === id)?.label }}
      </button>
    </div>
    <p v-if="models.diagnosticContext" class="diagnostic-graph-legend">
      Diagnostic {{ models.diagnosticContext.report.id.slice(0, 8) }} ·
      activation MAE colors: low ≤ {{ models.diagnosticContext.low ?? "—" }},
      medium ≤ {{ models.diagnosticContext.high ?? "—" }}, high above. Gray:
      unavailable. Relative distribution, not quality or root-cause attribution.
      Expand groups for individual node badges.
    </p>
    <div class="import-graph-area" v-if="models.report">
      <GraphCanvas
        ref="canvas"
        :analysis="projected"
        :selected-node="models.selectedNode"
        :highlighted="highlighted"
        :diagnostic-badges="models.diagnosticBadges"
        @select="models.selectedNode = $event"
        @error="models.error = $event"
      />
    </div>
    <div v-else class="import-empty">
      <div class="info-icon"><Box :size="32" /></div>
      <span class="eyebrow">MODEL IMPORTERS</span>
      <h1>Understand a real model.</h1>
      <p>
        Bring an ONNX model or a PyTorch definition into the same analysis
        workspace. Inspect real structure, tensors, weights and diagnostics.
      </p>
      <div class="import-format-list">
        <span v-for="item in importers" :key="item.id"
          >{{ item.format.toUpperCase()
          }}<small
            >{{ item.extensions.join(" · ") }} ·
            {{ item.available ? "Available" : "Missing runtime" }}</small
          ></span
        >
      </div>
      <button
        class="primary"
        :disabled="!editor.projectPath || models.running"
        @click="dialog = true"
      >
        <Upload :size="14" />Import model
      </button>
      <p class="muted small">
        {{
          editor.projectPath
            ? "Models stay local. ONNX imports do not execute inference."
            : "Create or open a project to register model artifacts."
        }}
      </p>
    </div>
    <footer class="import-history" v-if="models.records.length">
      <span>Import history</span
      ><span
        v-for="record in models.records.slice(0, 6)"
        :key="record.id"
        :title="record.error ?? record.file_path"
        >{{
          record.artifact?.source_name ?? record.file_path.split(/[\\/]/).at(-1)
        }}
        · {{ record.state }}</span
      >
    </footer>
    <div v-if="dialog" class="modal-backdrop">
      <section
        class="modal import-model-dialog"
        role="dialog"
        aria-modal="true"
        aria-label="Import model"
      >
        <button
          class="modal-close"
          title="Close model import"
          @click="dialog = false"
        >
          <X :size="17" />
        </button>
        <h2>Import model</h2>
        <p>
          Keep the original intact. A versioned local artifact and analysis
          report will be saved in this project.
        </p>
        <label
          >Model source path
          <div class="input-with-button">
            <input
              v-model="source"
              aria-label="Model source path"
              placeholder="C:\models\model.onnx"
            /><button
              v-if="desktop"
              title="Browse model file"
              @click="
                action(async () => {
                  const file = await rpc<string>('dialog.model');
                  if (file) source = file;
                })
              "
            >
              <FolderOpen :size="14" />
            </button></div
        ></label>
        <button
          class="subtle full-width"
          :disabled="busy || !source"
          @click="action(inspect)"
        >
          Detect format / inspect entry points
        </button>
        <p v-if="probe" class="import-detected">
          {{ probe.format.toUpperCase() }} · {{ probe.id }} ·
          {{
            probe.requires_execution
              ? "Explicit local Python execution"
              : "Static model analysis"
          }}
        </p>
        <template v-if="probe?.requires_execution"
          ><label
            >Model class or factory<select
              aria-label="Imported Python entry"
              v-model="entry"
            >
              <option v-for="item in probe.entries" :key="item.name">
                {{ item.name }}
              </option>
            </select></label
          ><label
            >Constructor JSON<textarea
              aria-label="Imported Python constructor"
              v-model="constructor"
              rows="2"
            /></label
          ><label
            >Example inputs JSON<textarea
              aria-label="Imported Python inputs"
              v-model="inputs"
              rows="3"
            />
          </label>
          <p class="python-execution-note">
            Import executes local Python in a disposable process. It has your
            user permissions and is not a sandbox.
          </p></template
        >
        <button
          class="primary full-width"
          :disabled="busy || !source || !probe || models.running"
          @click="action(start)"
        >
          {{
            probe?.requires_execution
              ? "Import and capture Python"
              : "Import and analyze"
          }}
        </button>
        <p class="modal-error" v-if="models.error">{{ models.error }}</p>
      </section>
    </div>
  </section>
</template>
<style scoped>
.diagnostic-graph-legend {
  margin: 0;
  padding: 9px 16px;
  color: #9ebccd;
  font-size: 10px;
  background: #1a2b37;
  border-bottom: 1px solid #375364;
  line-height: 1.5;
}
</style>
