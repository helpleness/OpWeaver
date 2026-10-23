<script setup lang="ts">
import { computed, ref, toRaw, watch } from "vue";
import { useLab } from "../stores/lab";
import { useEditor } from "../stores/editor";
import { rpc } from "../api";
import { defaultWorkflow, type ExperimentConfig } from "../lab-types";

const props = defineProps<{ prepareInputs: () => void }>();

const lab = useLab(),
  editor = useEditor();
const capability = computed(() =>
  lab.backends.find((b) => b.id === "microsoft.olive"),
);
const environment = computed(() => capability.value?.environment);
const manifest = computed(() => capability.value?.passes ?? []);
const python = ref("");
const adding = ref("OnnxPeepholeOptimizer");
const busy = ref(false),
  message = ref("");
const advanced = ref("");
const preview = ref<any>(null);
const saved = ref<{ id: string; config: ExperimentConfig }[]>([]);
const workflow = computed(() => lab.config.optimization.workflow!);
watch(
  workflow,
  () => {
    preview.value = null;
    message.value = "";
  },
  { deep: true },
);
if (!lab.config.optimization.workflow)
  lab.config.optimization.workflow = defaultWorkflow();
watch(
  () => environment.value?.python_path,
  (value) => {
    python.value = value ?? "";
  },
  { immediate: true },
);
async function action(fn: () => Promise<unknown>) {
  busy.value = true;
  message.value = "";
  lab.error = "";
  try {
    await fn();
  } catch (e) {
    lab.error = String(e);
  } finally {
    busy.value = false;
  }
}
async function configure() {
  await rpc("lab.environment.configure", {
    project: editor.projectPath,
    environment: { python_path: python.value },
  });
  lab.config.optimization.environment = null;
  await lab.probe();
  message.value = environment.value?.reason ?? "Environment checked";
}
function add() {
  const spec = manifest.value.find(
    (p) => p.type === adding.value && p.available,
  );
  if (!spec) return;
  workflow.value.passes.push({
    schema_version: "1.0",
    id: `pass_${Date.now().toString(36)}`,
    type: spec.type,
    enabled: true,
    parameters: Object.fromEntries(
      spec.parameters.map((p) => [p.name, p.default]),
    ),
  });
  if (spec.type === "OnnxOpVersionConversion")
    workflow.value.passes.at(-1)!.parameters.target_opset =
      (lab.source?.model.opsets as Record<string, number>)?.[""] ?? 18;
  preview.value = null;
}
function move(index: number, delta: number) {
  const items = workflow.value.passes;
  const other = index + delta;
  if (other < 0 || other >= items.length) return;
  [items[index], items[other]] = [items[other], items[index]];
}
function allowed(pass: any, name: string, value: string) {
  if (pass.type !== "OnnxStaticQuantization") return true;
  const p = pass.parameters;
  if (name === "quant_format" && value === "QOperator")
    return p.activation_type === "uint8";
  if (name === "precision" && value === "uint8")
    return p.activation_type === "uint8" && !p.per_channel;
  if (name === "activation_type" && value === "int8")
    return p.precision !== "uint8" && p.quant_format !== "QOperator";
  return true;
}
async function validate() {
  props.prepareInputs();
  preview.value = await rpc("lab.workflow.preview", {
    project: editor.projectPath,
    config: lab.config,
  });
  message.value = "Configuration validated by the installed Olive API";
}
async function save() {
  props.prepareInputs();
  await rpc("lab.workflow.save", {
    project: editor.projectPath,
    config: lab.config,
  });
  await list();
  message.value = "Workflow configuration saved";
}
async function list() {
  saved.value = await rpc("lab.workflow.list", { project: editor.projectPath });
}
async function load(id: string) {
  const next = saved.value.find((v) => v.id === id);
  if (!next) return;
  await lab.selectSource(next.config.source_model_id);
  lab.config = structuredClone(toRaw(next.config));
  preview.value = null;
}
async function applyAdvanced() {
  props.prepareInputs();
  const next = structuredClone(toRaw(lab.config));
  next.optimization.workflow = JSON.parse(advanced.value);
  const checked = await rpc<{ config: ExperimentConfig }>(
    "lab.workflow.preview",
    { project: editor.projectPath, config: next },
  );
  lab.config.optimization.workflow = checked.config.optimization.workflow;
  await validate();
}
void list().catch((e) => (lab.error = String(e)));
</script>
<template>
  <section class="olive-workflow" aria-label="Olive workflow editor">
    <div class="olive-environment">
      <strong>Independent environment</strong
      ><span :class="{ available: environment?.available }">{{
        environment?.status ?? "Not Installed"
      }}</span>
    </div>
    <label
      >Python interpreter or virtual environment<input
        aria-label="Olive Python environment"
        v-model="python"
        placeholder="C:\envs\olive\Scripts\python.exe"
    /></label>
    <button :disabled="busy" @click="action(configure)">
      {{ busy ? "Checking…" : "Save & detect environment" }}
    </button>
    <p class="muted small">
      {{ environment?.reason }} · Verified API: 0.13.x. Install explicitly with
      scripts/setup_olive.ps1.
    </p>
    <details v-if="environment?.versions">
      <summary>Versions & installation</summary>
      <pre>{{ JSON.stringify(environment, null, 2) }}</pre>
    </details>
    <label
      >Workflow ID<input aria-label="Olive workflow ID" v-model="workflow.id"
    /></label>
    <label
      >Available pass<select aria-label="Olive pass type" v-model="adding">
        <option
          v-for="item in manifest"
          :key="item.type"
          :value="item.type"
          :disabled="!item.available"
        >
          {{ item.type }}{{ item.available ? "" : " · unavailable" }}
        </option>
      </select></label
    >
    <p
      class="muted small"
      v-for="item in manifest.filter((p) => !p.available)"
      :key="item.type"
    >
      {{ item.type }}: {{ item.reason }}
    </p>
    <button
      :disabled="
        !environment?.available ||
        workflow.passes.length >= 16 ||
        !manifest.find((p) => p.type === adding)?.available
      "
      @click="add"
    >
      Add pass
    </button>
    <article
      class="olive-pass"
      v-for="(pass, index) in workflow.passes"
      :key="pass.id"
    >
      <div class="olive-pass-heading">
        <label
          ><input
            type="checkbox"
            v-model="pass.enabled"
            :aria-label="`Enable pass ${index + 1}`"
          />{{ index + 1 }} · {{ pass.type }}</label
        >
      </div>
      <div class="olive-pass-actions">
        <button
          :disabled="index === 0"
          :aria-label="`Move pass ${index + 1} up`"
          @click="move(index, -1)"
        >
          ↑</button
        ><button
          :disabled="index === workflow.passes.length - 1"
          :aria-label="`Move pass ${index + 1} down`"
          @click="move(index, 1)"
        >
          ↓</button
        ><button
          :aria-label="`Delete pass ${index + 1}`"
          @click="workflow.passes.splice(index, 1)"
        >
          Remove
        </button>
      </div>
      <label
        v-for="param in manifest.find((p) => p.type === pass.type)
          ?.parameters ?? []"
        :key="param.name"
        :title="param.description"
        class="olive-parameter"
      >
        <input
          v-if="param.type === 'boolean'"
          type="checkbox"
          v-model="pass.parameters[param.name]"
          :disabled="
            param.available === false ||
            (param.name === 'per_channel' &&
              pass.parameters.precision === 'uint8')
          "
          :aria-label="`${index + 1} ${param.name}`"
        />
        <span>{{ param.name }}</span>
        <input
          v-if="param.type === 'integer'"
          type="number"
          v-model.number="pass.parameters[param.name]"
          :min="param.minimum"
          :max="param.maximum"
          :aria-label="`${index + 1} ${param.name}`"
        />
        <select
          v-if="param.type === 'enum'"
          v-model="pass.parameters[param.name]"
          :aria-label="`${index + 1} ${param.name}`"
        >
          <option
            v-for="value in param.choices"
            :key="value"
            :value="value"
            :disabled="!allowed(pass, param.name, value)"
          >
            {{ value }}
          </option>
        </select>
        <input
          v-if="param.type === 'strings'"
          :value="(pass.parameters[param.name] ?? []).join(', ')"
          :aria-label="`${index + 1} ${param.name}`"
          placeholder="Comma separated; empty uses runtime defaults"
          @change="
            pass.parameters[param.name] = (
              $event.target as HTMLInputElement
            ).value
              .split(',')
              .map((v) => v.trim())
              .filter(Boolean)
          "
        />
      </label>
      <p class="muted small">
        {{ manifest.find((p) => p.type === pass.type)?.model_constraint }}
      </p>
      <p v-if="pass.type === 'OnnxStaticQuantization'" class="muted small">
        CPU static quantization · explicit calibration dataset below. Hidden
        preprocessing and automatic QDQ operator expansion are disabled. Add
        OnnxQuantizationPreprocess before this pass when needed. INT8 errors are
        evaluated separately from execution validity.
      </p>
    </article>
    <label
      >Output model name<input
        aria-label="Olive output model name"
        v-model="workflow.output.model_name"
    /></label>
    <label class="olive-parameter"
      ><input
        type="checkbox"
        v-model="workflow.output.save_external_data"
      />Separate large weights</label
    >
    <label
      >External tensor threshold (bytes)<input
        type="number"
        min="0"
        v-model.number="workflow.output.external_data_threshold"
    /></label>
    <div class="olive-pass-actions">
      <button
        :disabled="busy || !lab.source || !environment?.available"
        @click="action(validate)"
      >
        Validate & preview</button
      ><button
        :disabled="busy || !lab.source || !environment?.available"
        @click="action(save)"
      >
        Save workflow
      </button>
    </div>
    <label v-if="saved.length"
      >Saved workflows<select
        aria-label="Saved Olive workflows"
        value=""
        @change="action(() => load(($event.target as HTMLSelectElement).value))"
      >
        <option value="" disabled>Choose configuration</option>
        <option v-for="item in saved" :key="item.id">{{ item.id }}</option>
      </select></label
    >
    <details @toggle="advanced = JSON.stringify(workflow, null, 2)">
      <summary>Advanced generic workflow JSON</summary>
      <textarea
        aria-label="Advanced Olive workflow"
        v-model="advanced"
        rows="14"
        spellcheck="false"
      /><button :disabled="busy" @click="action(applyAdvanced)">
        Validate & apply JSON
      </button>
    </details>
    <details v-if="preview" open>
      <summary>Generated Olive native configuration</summary>
      <p class="muted small">{{ preview.note }}</p>
      <pre>{{ JSON.stringify(preview.native_workflow, null, 2) }}</pre>
    </details>
    <p v-if="message" role="status" class="olive-message">{{ message }}</p>
  </section>
</template>
<style scoped>
.olive-workflow {
  margin: 12px 0;
  padding: 12px;
  border: 1px solid #394661;
  border-radius: 9px;
  background: linear-gradient(140deg, #1b2638, #151c29);
}
.olive-environment {
  display: flex;
  gap: 7px;
  align-items: center;
  flex-wrap: wrap;
  margin-bottom: 9px;
}
.olive-environment span {
  font-size: 10px;
  padding: 3px 6px;
  background: #3a2e27;
  color: #e2b68c;
  border-radius: 4px;
}
.olive-environment .available {
  background: #203e37;
  color: #91d4bb;
}
.olive-pass {
  margin: 10px 0;
  border: 1px solid #344153;
  padding: 9px;
  border-radius: 6px;
  background: #111a27;
}
.olive-pass-heading label,
.olive-parameter {
  display: flex;
  flex-direction: row;
  align-items: center;
  gap: 6px;
  overflow-wrap: anywhere;
}
.olive-workflow input[type="checkbox"] {
  width: 14px;
  margin: 0;
  flex-shrink: 0;
}
.olive-pass-actions {
  display: flex;
  gap: 5px;
  margin: 7px 0;
}
.olive-pass-actions button {
  flex: 1;
  justify-content: center;
}
.olive-workflow details {
  margin: 10px 0;
}
.olive-workflow pre {
  font-size: 10px;
  overflow: auto;
  max-height: 280px;
  white-space: pre;
}
.olive-workflow textarea {
  width: 100%;
  box-sizing: border-box;
  margin: 8px 0;
  font-family: Consolas, monospace;
}
.olive-message {
  color: #91d4bb;
}
</style>
