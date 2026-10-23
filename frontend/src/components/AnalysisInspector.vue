<script setup lang="ts">
import { computed, ref, watch } from "vue";
import {
  Braces,
  Code2,
  LockKeyhole,
  SlidersHorizontal,
  ExternalLink,
} from "lucide-vue-next";
import { useCode } from "../stores/code";
const code = useCode();
const node = computed(() =>
  code.activeAnalysis?.nodes.find((n) => n.id === code.selectedAnalysis),
);
const mapping = computed(() =>
  code.sourceMap.find((m) => m.graph_node_id === node.value?.id),
);
const key = ref("");
const value = ref("");
const error = ref("");
const editableKeys = computed(() =>
  Object.keys(node.value?.parameters ?? {}).filter(
    (k) => !["module_type", "representation", "args", "kwargs"].includes(k),
  ),
);
watch(node, () => {
  key.value = editableKeys.value[0] ?? "";
  value.value = key.value
    ? JSON.stringify(node.value?.parameters[key.value])
    : "";
  error.value = "";
});
watch(
  key,
  () => (value.value = JSON.stringify(node.value?.parameters[key.value])),
);
async function preview() {
  try {
    await code.previewParameter(
      node.value!.id,
      key.value,
      JSON.parse(value.value),
    );
  } catch (e) {
    error.value = String(e);
  }
}
</script>
<template>
  <aside class="inspector analysis-inspector">
    <div class="panel-heading">
      <SlidersHorizontal :size="15" /><strong>Analysis inspector</strong
      ><span class="badge">FX</span>
    </div>
    <template v-if="node"
      ><div class="inspector-intro">
        <div class="large-node-icon"><Braces :size="22" /></div>
        <div>
          <h2>{{ node.label }}</h2>
          <code>{{ node.module_path || node.kind }}</code>
        </div>
      </div>
      <div class="inspector-section">
        <span class="pill"
          ><LockKeyhole :size="11" />Code-First · read-only graph</span
        >
        <p class="description">{{ node.operator }}</p>
      </div>
      <div class="section-label">PARAMETERS</div>
      <div class="inspector-section">
        <pre>{{ JSON.stringify(node.parameters, null, 2) }}</pre>
      </div>
      <div class="section-label">TENSOR / EXAMPLE INPUT</div>
      <div class="inspector-section">
        <pre>{{ JSON.stringify(node.tensors, null, 2) }}</pre>
      </div>
      <div class="section-label">SOURCE MAPPING</div>
      <div class="inspector-section">
        <div class="tensor-card">
          <header>
            <span>{{ mapping?.status ?? "unmapped" }}</span
            ><Code2 :size="12" />
          </header>
          <code>{{ mapping?.module_path || "Call site" }}</code>
          <p>
            {{
              mapping?.start_line
                ? `Lines ${mapping.start_line}–${mapping.end_line ?? mapping.start_line}`
                : "Exact location unavailable"
            }}
          </p>
          <p class="muted small">{{ mapping?.reason }}</p>
        </div>
        <button
          class="subtle full-width"
          :disabled="!mapping?.start_line"
          @click="code.locateNode(node.id)"
        >
          <ExternalLink :size="12" />Locate source
        </button>
      </div>
      <template
        v-if="
          mapping?.status === 'exact' && node.module_path && editableKeys.length
        "
        ><div class="section-label">SAFE SOURCE EDIT</div>
        <div class="inspector-section parameter-list">
          <p class="muted small">
            Prepares a local literal edit. Computed parameters and ambiguous
            constructors are rejected.
          </p>
          <label
            >Parameter<select aria-label="Source patch parameter" v-model="key">
              <option v-for="name in editableKeys" :key="name">
                {{ name }}
              </option>
            </select></label
          ><label
            >New value (JSON)<input
              aria-label="Source patch value"
              v-model="value" /></label
          ><button
            class="subtle full-width"
            :disabled="code.running"
            @click="preview"
          >
            Review source patch
          </button>
          <p class="modal-error" v-if="error">{{ error }}</p>
        </div></template
      >
    </template>
    <div v-else class="inspector-section model-analysis-summary">
      <h2>Model analysis</h2>
      <template v-if="code.lastResult"
        ><p>
          {{ code.activeAnalysis?.name }} ·
          {{ code.lastResult.capture.backend }}
        </p>
        <div class="info-facts">
          <div>
            <span>Nodes</span
            ><code>{{ code.lastResult.statistics.nodes }}</code>
          </div>
          <div>
            <span>Parameters</span
            ><code>{{
              code.lastResult.statistics.parameters.toLocaleString()
            }}</code>
          </div>
          <div>
            <span>Source size</span
            ><code>{{ code.lastResult.statistics.file_bytes }} B</code>
          </div>
          <div><span>FLOPs</span><code>Not measured</code></div>
        </div>
        <p class="muted small">{{ code.lastResult.capture.specialized }}</p>
        <p class="muted small">
          Select a node to inspect parameters and its source mapping.
        </p></template
      >
      <p class="muted small" v-else>
        Parse a Python model to inspect real structure and tensor metadata.
      </p>
    </div>
  </aside>
</template>
