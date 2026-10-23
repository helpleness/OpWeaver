<script setup lang="ts">
import { onMounted, ref } from "vue";
import { rpc } from "../api";
import { useLab } from "../stores/lab";
import { useEditor } from "../stores/editor";
import ImageDatasetEditor from "./ImageDatasetEditor.vue";
const props = defineProps<{ prepareInputs: () => void; desktop?: boolean }>();
const lab = useLab(),
  editor = useEditor();
const records = ref<any[]>([]),
  busy = ref(false),
  preview = ref<any>(null);
const role = ref("calibration"),
  mode = ref("npz");
const paths = ref(""),
  mapping = ref("{}"),
  name = ref("Calibration dataset"),
  count = ref(8),
  seed = ref(42),
  labelKey = ref("");
const imageEditor = ref<InstanceType<typeof ImageDatasetEditor>>();
function roleChanged() {
  const names: Record<string, string> = {
    calibration: "Calibration dataset",
    evaluation: "Evaluation dataset",
    validation: "Validation dataset",
    diagnostic: "Diagnostic dataset",
    final_evaluation: "Final evaluation dataset",
  };
  name.value = names[role.value] ?? "Dataset";
  seed.value =
    role.value === "calibration"
      ? 42
      : role.value === "diagnostic"
        ? 2042
        : role.value === "final_evaluation"
          ? 3042
          : 1042;
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
async function refresh() {
  records.value = await rpc<any[]>("lab.dataset.list", {
    project: editor.projectPath,
  });
}
async function create() {
  if (mode.value !== "images") props.prepareInputs();
  const files = paths.value
    .split(/\r?\n/)
    .map((s) => s.trim())
    .filter(Boolean);
  const record = await rpc<any>("lab.dataset.create", {
    project: editor.projectPath,
    config: {
      source_model_id: lab.config.source_model_id,
      name: name.value,
      role: role.value,
      mode: mode.value,
      files: mode.value === "npz" ? files : [],
      count: mode.value === "npz" ? files.length : count.value,
      seed: seed.value,
      shapes: mode.value === "random" ? lab.config.inputs.shapes : {},
      mapping: JSON.parse(mapping.value),
      label_key: labelKey.value || null,
      image: mode.value === "images" ? imageEditor.value?.config() : null,
    },
  });
  if (role.value === "calibration")
    lab.config.optimization.calibration_dataset_id = record.id;
  else if (role.value === "final_evaluation")
    lab.config.evaluation.final_dataset_id = record.id;
  else if (role.value === "validation" || role.value === "evaluation")
    lab.config.evaluation.dataset_id = record.id;
  preview.value = record;
  await refresh();
}
async function copyInputs() {
  props.prepareInputs();
  mode.value = lab.config.inputs.mode;
  paths.value = lab.config.inputs.files.join("\n");
  seed.value =
    lab.config.inputs.seed +
    (role.value === "calibration"
      ? 0
      : role.value === "diagnostic"
        ? 2000
        : role.value === "final_evaluation"
          ? 3000
          : 1000);
  mapping.value = JSON.stringify(
    Object.fromEntries((lab.source?.inputs ?? []).map((v) => [v.name, v.name])),
  );
}
async function choose() {
  const files = await rpc<string[]>("dialog.inputs");
  if (files?.length) paths.value = files.join("\n");
}
async function inspect(id: string | null) {
  if (id)
    preview.value = await rpc("lab.dataset.load", {
      project: editor.projectPath,
      id,
    });
}
onMounted(() => void action(refresh));
</script>
<template>
  <section
    class="dataset-panel"
    aria-label="Calibration and evaluation datasets"
  >
    <strong>Calibration & evaluation datasets</strong>
    <p class="muted small">
      Local images or NPZ · immutable snapshots, input mapping and SHA256
      checked. Validation uses independent data by default; final evaluation
      stays held out.
    </p>
    <label
      >Calibration dataset<select
        aria-label="Calibration dataset"
        v-model="lab.config.optimization.calibration_dataset_id"
        @change="
          action(() => inspect(lab.config.optimization.calibration_dataset_id))
        "
      >
        <option :value="null">Choose dataset</option>
        <option
          v-for="item in records"
          :key="item.id"
          :value="item.id"
          :disabled="
            !item.availability?.available || item.role === 'final_evaluation'
          "
        >
          {{ item.name }} · {{ item.batch_count }} batches ·
          {{ item.id.slice(0, 6)
          }}{{ item.availability?.available ? "" : " · missing/changed" }}
        </option>
      </select></label
    >
    <label
      >Evaluation dataset<select
        aria-label="Evaluation dataset"
        v-model="lab.config.evaluation.dataset_id"
        @change="action(() => inspect(lab.config.evaluation.dataset_id))"
      >
        <option :value="null">Choose independent dataset</option>
        <option
          v-for="item in records"
          :key="item.id"
          :value="item.id"
          :disabled="
            !item.availability?.available || item.role === 'final_evaluation'
          "
        >
          {{ item.name }} · {{ item.batch_count }} batches ·
          {{ item.id.slice(0, 6) }}
        </option>
      </select></label
    >
    <label
      >Held-out final evaluation (optional)<select
        aria-label="Final evaluation dataset"
        v-model="lab.config.evaluation.final_dataset_id"
        @change="action(() => inspect(lab.config.evaluation.final_dataset_id))"
      >
        <option :value="null">
          Not selected · never used to tune quantization
        </option>
        <option
          v-for="item in records.filter((r) => r.role === 'final_evaluation')"
          :key="item.id"
          :value="item.id"
          :disabled="!item.availability?.available"
        >
          {{ item.name }} · {{ item.example_count }} examples
        </option>
      </select></label
    >
    <label class="reuse"
      ><input
        type="checkbox"
        aria-label="Allow calibration reuse"
        v-model="lab.config.evaluation.allow_calibration_reuse"
      />Explicitly allow calibration reuse</label
    >
    <p
      v-if="lab.config.evaluation.allow_calibration_reuse"
      class="dataset-warning"
    >
      Reused calibration data is not independent quality evaluation.
    </p>
    <details open>
      <summary>Create a dataset snapshot</summary>
      <label
        >Role<select
          aria-label="Dataset role"
          v-model="role"
          @change="roleChanged"
        >
          <option value="calibration">Calibration</option>
          <option value="evaluation">Evaluation</option>
          <option value="validation">Validation · tune on this split</option>
          <option value="diagnostic">Diagnostic · intermediate tensors</option>
          <option value="final_evaluation">Final evaluation · held out</option>
        </select></label
      >
      <label>Name<input aria-label="Dataset name" v-model="name" /></label>
      <label
        >Source<select aria-label="Dataset source" v-model="mode">
          <option value="npz">NPZ files</option>
          <option value="random">Synthetic seeded inputs</option>
          <option value="images">Local image classification CSV</option>
        </select></label
      >
      <button v-if="mode !== 'images'" @click="action(copyInputs)">
        Copy current InputConfig
      </button>
      <ImageDatasetEditor v-if="mode === 'images'" ref="imageEditor" />
      <template v-if="mode === 'npz'">
        <label
          >NPZ paths · one batch per line<textarea
            aria-label="Dataset NPZ paths"
            v-model="paths"
            rows="4"
          />
        </label>
        <button v-if="desktop" @click="action(choose)">
          Choose dataset NPZ files
        </button>
        <label
          >Input → NPZ key mapping (JSON)<input
            aria-label="Dataset input mapping"
            v-model="mapping"
            placeholder='{"image":"pixels"}'
        /></label>
        <label
          >Label key (optional)<input
            aria-label="Dataset label key"
            v-model="labelKey"
            placeholder="labels · integer [batch]"
        /></label>
      </template>
      <template v-else-if="mode === 'random'">
        <label
          >Batches<input
            aria-label="Dataset batch count"
            type="number"
            min="1"
            max="512"
            v-model.number="count"
        /></label>
        <label
          >Seed<input
            aria-label="Dataset seed"
            type="number"
            min="0"
            v-model.number="seed"
        /></label>
        <p class="dataset-warning">
          Synthetic data is for testing. Concrete shapes come from InputConfig
          above; production calibration needs representative data.
        </p>
      </template>
      <button :disabled="busy || !lab.source" @click="action(create)">
        {{ busy ? "Checking data…" : "Validate & save dataset" }}
      </button>
    </details>
    <details v-if="preview">
      <summary>
        Dataset preview · {{ preview.batch_count }} batches /
        {{ preview.example_count }} examples
      </summary>
      <p
        v-for="warning in preview.warnings"
        :key="warning"
        class="dataset-warning"
      >
        {{ warning }}
      </p>
      <code>{{ preview.fingerprint }}</code>
      <table class="lab-table">
        <thead>
          <tr>
            <th>Input</th>
            <th>Batch shape</th>
            <th>DType</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="tensor in preview.samples?.[0]?.tensors"
            :key="tensor.name"
          >
            <td>{{ tensor.name }}</td>
            <td>{{ tensor.shape }}</td>
            <td>{{ tensor.dtype }}</td>
          </tr>
        </tbody>
      </table>
      <p class="muted small">
        Role {{ preview.role }} · Adapter
        {{ preview.source_adapter?.type ?? "npz" }} · Version
        {{ preview.dataset_version ?? "1.0" }} · Classes
        {{ preview.observed_class_count ?? "not provided" }}
      </p>
      <pre v-if="preview.preprocessing">{{
        JSON.stringify(
          {
            class_mapping: preview.class_mapping,
            label_distribution: preview.label_distribution,
            preprocessing: preview.preprocessing,
          },
          null,
          2,
        )
      }}</pre>
      <details>
        <summary>Sample hashes & source adapter</summary>
        <pre>{{
          JSON.stringify(
            {
              adapter: preview.source_adapter,
              samples: preview.samples?.slice(0, 3),
            },
            null,
            2,
          )
        }}</pre>
      </details>
    </details>
    <details>
      <summary>Task evaluator & optional quality gates</summary>
      <label
        >Task evaluator<select
          aria-label="Task evaluator"
          v-model="lab.config.evaluation.task"
        >
          <option value="none">
            Numerical errors only · task not evaluated
          </option>
          <option value="classification_top1">
            Classification Top1 / Top5 · labels required
          </option>
        </select></label
      >
      <label v-if="lab.config.evaluation.task === 'classification_top1'"
        >Scores output name<input
          aria-label="Classification output name"
          v-model="lab.config.evaluation.output_name"
          placeholder="scores · [batch, classes]"
      /></label>
      <template v-for="(value, key) in lab.config.evaluation.gates" :key="key"
        ><label v-if="key !== 'schema_version'"
          >{{ key
          }}<input
            type="number"
            min="0"
            step="any"
            :aria-label="`Quality gate ${key}`"
            :value="value ?? ''"
            placeholder="Unset"
            @input="
              (lab.config.evaluation.gates as any)[key] =
                ($event.target as HTMLInputElement).value === ''
                  ? null
                  : Number(($event.target as HTMLInputElement).value)
            " /></label
      ></template>
      <p class="muted small">
        Gate failure retains executable artifacts and Benchmark results.
        Accuracy drop requires labeled Top1 evaluation.
      </p>
    </details>
  </section>
</template>
<style scoped>
.dataset-panel {
  margin: 12px 0;
  padding: 12px;
  border: 1px solid #35605a;
  border-radius: 9px;
  background: linear-gradient(140deg, #192c30, #151e29);
}
.dataset-panel details {
  margin: 12px 0;
}
.dataset-panel pre {
  max-height: 240px;
  overflow: auto;
  font-size: 10px;
}
.dataset-panel code {
  overflow-wrap: anywhere;
  font-size: 10px;
}
.dataset-warning {
  color: #dfba88;
  font-size: 11px;
  line-height: 1.5;
}
.reuse {
  display: flex;
  flex-direction: row;
  align-items: center;
  gap: 6px;
}
.reuse input {
  width: 14px;
  flex-shrink: 0;
}
</style>
