<script setup lang="ts">
import { computed, ref } from "vue";
import {
  SlidersHorizontal,
  Box,
  LockKeyhole,
  ArrowLeft,
  FileCode2,
  AlertTriangle,
} from "lucide-vue-next";
import { useModels } from "../stores/models";
import { bytesText, shapeText, type TensorInfo } from "../model-types";
const models = useModels();
const emit = defineEmits<{ openPython: [] }>();
const report = computed(() => models.report);
const graph = computed(() =>
  models.subgraphKey
    ? report.value?.subgraphs[models.subgraphKey]
    : report.value?.analysis,
);
const node = computed(() =>
  graph.value?.nodes.find((n) => n.id === models.selectedNode),
);
const detail = computed(
  () => (node.value?.details as Record<string, any>) ?? {},
);
const nodeInputs = computed(() =>
  detail.value.input_tensors
    ? (Object.values(detail.value.input_tensors) as TensorInfo[])
    : ((graph.value?.edges
        .filter((e) => e.target === node.value?.id)
        .map((e) => ({
          name: `${e.source}:${e.target_port}`,
          shape: null,
          dtype: "unknown",
          ...graph.value?.nodes.find((n) => n.id === e.source)?.tensors,
        })) ?? []) as TensorInfo[]),
);
const nodeOutputs = computed(() =>
  detail.value.output_tensors
    ? (Object.values(detail.value.output_tensors) as TensorInfo[])
    : node.value?.outputs.length
      ? ([
          {
            name: node.value.label,
            shape: null,
            dtype: "unknown",
            ...node.value.tensors,
          },
        ] as TensorInfo[])
      : [],
);
const mapping = computed(() =>
  report.value?.pytorch?.source_map.find(
    (m) => m.graph_node_id === node.value?.id,
  ),
);
const nodeWeights = computed(
  () =>
    report.value?.initializers.filter((w) =>
      nodeInputs.value.some((v) => v.name === w.name),
    ) ?? [],
);
const weightSearch = ref("");
const weights = computed(
  () =>
    report.value?.initializers.filter((w) =>
      `${w.graph_path}${w.name} ${w.dtype}`
        .toLowerCase()
        .includes(weightSearch.value.toLowerCase()),
    ) ?? [],
);
const operators = computed(() =>
  Object.entries(report.value?.statistics.operators ?? {}).sort(
    (a, b) => b[1] - a[1],
  ),
);
const dynamic = (t: TensorInfo) =>
  t.shape === null
    ? "Unknown rank"
    : (t.shape ?? [])
        .map((d, i) =>
          typeof d === "number" ? null : `axis ${i}: ${d ?? "unknown"}`,
        )
        .filter(Boolean)
        .join(" · ") || "Static shape";
const relatedWarnings = computed(
  () =>
    report.value?.diagnostics.filter((d) => d.node === node.value?.id) ?? [],
);
</script>
<template>
  <aside class="inspector model-inspector" aria-label="Model Inspector">
    <div class="panel-heading">
      <SlidersHorizontal :size="15" /><strong>Model Inspector</strong
      ><span class="badge">{{
        report?.artifact.format.toUpperCase() ?? "MODEL"
      }}</span>
    </div>
    <div v-if="!report" class="inspector-empty">
      <Box :size="28" />
      <h2>Inspect real structure.</h2>
      <p>
        Import a model to see its tensors, operator distribution and
        diagnostics.
      </p>
    </div>
    <template v-else>
      <details
        v-if="report.quantization?.present"
        class="model-detail-section"
        open
      >
        <summary>
          Quantization · {{ report.quantization.representation }}
        </summary>
        <p class="muted small">
          Q {{ report.quantization.quantize_linear }} · DQ
          {{ report.quantization.dequantize_linear }} · QOperator
          {{ report.quantization.qoperator_count }} · Integer weights
          {{ (report.quantization.integer_weights as any[])?.length }}
        </p>
        <p class="muted small">{{ report.quantization.scope }}</p>
        <details>
          <summary>
            Integer tensors, scale / zero point & remaining float weights
          </summary>
          <pre>{{ JSON.stringify(report.quantization, null, 2) }}</pre>
        </details>
      </details>
      <template v-if="node">
        <button class="inspector-back" @click="models.selectedNode = null">
          <ArrowLeft :size="13" />Model overview
        </button>
        <div class="inspector-intro">
          <div class="large-node-icon"><Box :size="22" /></div>
          <div>
            <h2>{{ node.label }}</h2>
            <code>{{ node.id }}</code>
          </div>
        </div>
        <div class="inspector-section">
          <span class="pill"><LockKeyhole :size="11" />Read-only analysis</span>
          <dl class="model-detail-list">
            <dt>Operator</dt>
            <dd>{{ detail.op_type ?? node.operator }}</dd>
            <dt>Domain</dt>
            <dd>{{ detail.domain ?? "PyTorch" }}</dd>
            <dt>Opset</dt>
            <dd>{{ detail.opset ?? "Not applicable" }}</dd>
            <dt>Schema</dt>
            <dd>{{ detail.schema_status ?? "FX capture" }}</dd>
            <dt>Ports</dt>
            <dd>{{ node.inputs.length }} in / {{ node.outputs.length }} out</dd>
          </dl>
          <span v-if="detail.known === false" class="model-warning-badge"
            >Unknown / custom operator</span
          >
        </div>
        <details class="model-detail-section" open>
          <summary>Attributes / parameters</summary>
          <pre>{{ JSON.stringify(node.parameters, null, 2) }}</pre>
        </details>
        <details class="model-detail-section" open>
          <summary>Input tensors · {{ nodeInputs.length }}</summary>
          <div
            v-for="tensor in nodeInputs"
            :key="tensor.name"
            class="model-tensor-card"
          >
            <strong>{{ tensor.name }}</strong
            ><code>{{ shapeText(tensor) }}</code
            ><small>{{ tensor.dtype }} · {{ dynamic(tensor) }}</small
            ><small v-if="tensor.implicit_capture"
              >Implicit outer-scope capture</small
            >
          </div>
          <p class="muted small" v-if="!nodeInputs.length">
            {{
              node.kind === "placeholder"
                ? "Model input"
                : JSON.stringify(node.tensors)
            }}
          </p>
        </details>
        <details class="model-detail-section" open>
          <summary>Output tensors · {{ nodeOutputs.length }}</summary>
          <div
            v-for="tensor in nodeOutputs"
            :key="tensor.name"
            class="model-tensor-card"
          >
            <strong>{{ tensor.name }}</strong
            ><code>{{ shapeText(tensor) }}</code
            ><small>{{ tensor.dtype }} · {{ dynamic(tensor) }}</small>
          </div>
          <pre v-if="!nodeOutputs.length">{{
            JSON.stringify(node.tensors, null, 2)
          }}</pre>
        </details>
        <details class="model-detail-section" v-if="nodeWeights.length" open>
          <summary>Related initializers · {{ nodeWeights.length }}</summary>
          <div v-for="w in nodeWeights" :key="w.name" class="model-tensor-card">
            <strong>{{ w.name }}</strong
            ><code>{{ shapeText(w) }} · {{ w.dtype }}</code
            ><small
              >{{ w.elements.toLocaleString() }} elements ·
              {{ bytesText(w.logical_bytes) }} logical ·
              {{ w.external ? "external" : "embedded" }}</small
            >
          </div>
        </details>
        <details
          class="model-detail-section"
          v-if="detail.subgraphs?.length"
          open
        >
          <summary>Control-flow subgraphs</summary>
          <button
            class="subtle full-width"
            v-for="child in detail.subgraphs"
            :key="child.id"
            @click="models.subgraphKey = child.id"
          >
            Open {{ child.name }}
          </button>
        </details>
        <details
          class="model-detail-section"
          v-if="relatedWarnings.length"
          open
        >
          <summary>Node diagnostics</summary>
          <p
            class="model-diagnostic warning"
            v-for="(item, i) in relatedWarnings"
            :key="i"
          >
            {{ item.message }}
          </p>
        </details>
        <div v-if="mapping" class="inspector-section">
          <strong>Source mapping · {{ mapping.status }}</strong>
          <p class="muted small">
            {{ mapping.module_path }} ·
            {{
              mapping.start_line
                ? `Lines ${mapping.start_line}–${mapping.end_line ?? mapping.start_line}`
                : "Location unavailable"
            }}
          </p>
          <p class="muted small">{{ mapping.reason }}</p>
        </div>
        <button
          v-if="report.pytorch"
          class="subtle inspector-python"
          @click="emit('openPython')"
        >
          <FileCode2 :size="13" />Open mapped Python source
        </button>
      </template>
      <template v-else>
        <div class="inspector-intro">
          <div class="large-node-icon"><Box :size="22" /></div>
          <div>
            <h2>{{ report.artifact.name }}</h2>
            <code>{{ report.artifact.source_name }}</code>
          </div>
        </div>
        <div class="model-overview-metrics">
          <div>
            <span>Operators</span
            ><strong>{{ report.statistics.nodes.toLocaleString() }}</strong>
          </div>
          <div>
            <span>Operator types</span><strong>{{ operators.length }}</strong>
          </div>
          <div>
            <span>Inputs / outputs</span
            ><strong
              >{{ report.inputs.length }} / {{ report.outputs.length }}</strong
            >
          </div>
          <div>
            <span>Artifact size</span
            ><strong>{{ bytesText(report.artifact.total_file_bytes) }}</strong>
          </div>
        </div>
        <details class="model-detail-section" open>
          <summary>Model metadata</summary>
          <dl class="model-detail-list">
            <dt>Format</dt>
            <dd>{{ report.artifact.format.toUpperCase() }}</dd>
            <dt>IR</dt>
            <dd>{{ report.model.ir_version }}</dd>
            <dt>Opsets</dt>
            <dd>
              {{
                Object.entries(report.model.opsets)
                  .map(([key, v]) => `${key || "ai.onnx"}: ${v}`)
                  .join(", ") || "Not applicable"
              }}
            </dd>
            <dt>Producer</dt>
            <dd>
              {{ report.model.producer || "Unknown" }}
              {{ report.model.producer_version }}
            </dd>
            <dt>Parser</dt>
            <dd>
              {{ report.artifact.importer_id }}
              {{ report.model.parser_version }}
            </dd>
            <dt>Imported</dt>
            <dd>
              {{
                new Date(report.artifact.imported_at * 1000).toLocaleString()
              }}
            </dd>
            <dt>Checker</dt>
            <dd>{{ report.validation.checker }}</dd>
            <dt>Shape inference</dt>
            <dd>{{ report.validation.shape_inference }}</dd>
            <dt>Execution</dt>
            <dd>{{ report.validation.runtime }}</dd>
            <dt>FLOPs</dt>
            <dd>Not measured</dd>
          </dl>
          <p class="muted small">{{ report.model.scope }}</p>
          <details>
            <summary>Artifact identity / paths</summary>
            <code class="artifact-identity"
              >{{ report.artifact.id }}
              {{ report.artifact.sha256 }}
              {{ report.artifact.file_path }}
              Source: {{ report.artifact.source_path }}</code
            >
          </details>
          <details v-if="report.artifact.experiment_id">
            <summary>Optimization lineage / target</summary>
            <code class="artifact-identity"
              >Parent: {{ report.artifact.parent_model_id }} Experiment:
              {{ report.artifact.experiment_id }}
              {{
                JSON.stringify(report.artifact.target_environment, null, 2)
              }}</code
            >
          </details>
        </details>
        <details class="model-detail-section" open>
          <summary>Model inputs · {{ report.inputs.length }}</summary>
          <div
            v-for="tensor in report.inputs"
            :key="tensor.name"
            class="model-tensor-card"
          >
            <strong>{{ tensor.name }}</strong
            ><code>{{ shapeText(tensor) }}</code
            ><small>{{ tensor.dtype }} · {{ dynamic(tensor) }}</small>
          </div>
        </details>
        <details class="model-detail-section" open>
          <summary>Model outputs · {{ report.outputs.length }}</summary>
          <div
            v-for="tensor in report.outputs"
            :key="tensor.name"
            class="model-tensor-card"
          >
            <strong>{{ tensor.name }}</strong
            ><code>{{ shapeText(tensor) }}</code
            ><small>{{ tensor.dtype }} · {{ dynamic(tensor) }}</small>
          </div>
        </details>
        <details class="model-detail-section" open>
          <summary>Weights / parameters</summary>
          <dl class="model-detail-list">
            <dt>Initializers</dt>
            <dd>{{ report.statistics.initializer_count }}</dd>
            <dt>Initializer elements</dt>
            <dd>
              {{
                report.statistics.initializer_elements?.toLocaleString() ??
                "Not applicable"
              }}
            </dd>
            <dt v-if="report.statistics.learnable_parameters !== undefined">
              PyTorch parameters
            </dt>
            <dd v-if="report.statistics.learnable_parameters !== undefined">
              {{ report.statistics.learnable_parameters.toLocaleString() }}
            </dd>
            <dt>Logical tensor bytes</dt>
            <dd>{{ bytesText(report.statistics.logical_tensor_bytes) }}</dd>
            <dt
              v-for="(count, dtype) in report.statistics.weight_dtypes"
              :key="dtype"
            >
              {{ dtype }} elements <span>{{ count.toLocaleString() }}</span>
            </dt>
          </dl>
          <p class="muted small">{{ report.statistics.parameters_note }}</p>
          <div
            class="model-tensor-card"
            v-for="file in report.artifact.external_files"
            :key="file.path"
          >
            <strong>{{ file.path }}</strong
            ><small>{{ bytesText(file.file_bytes) }} · external file</small>
          </div>
          <input
            v-if="report.initializers.length"
            aria-label="Search initializers"
            v-model="weightSearch"
            placeholder="Find an initializer…"
          />
          <div
            class="initializer-preview"
            v-for="w in weights.slice(0, 100)"
            :key="`${w.graph_path}${w.name}`"
          >
            <strong>{{ w.graph_path }}{{ w.name }}</strong
            ><code>{{ shapeText(w) }} · {{ w.dtype }}</code
            ><small>{{ w.elements.toLocaleString() }} elements</small>
          </div>
          <p class="muted small" v-if="weights.length > 100">
            Showing first 100 of {{ weights.length }} matches. Narrow the
            search.
          </p>
        </details>
        <details class="model-detail-section" open>
          <summary>Operator distribution</summary>
          <div
            v-for="[op, count] in operators"
            :key="op"
            class="operator-distribution-row"
          >
            <span>{{ op }}</span
            ><strong>{{ count }}</strong
            ><i
              :style="{ width: `${(count / (operators[0]?.[1] ?? 1)) * 100}%` }"
            />
          </div>
        </details>
        <details class="model-detail-section" open>
          <summary>Diagnostics · {{ report.diagnostics.length }}</summary>
          <p class="muted small" v-if="!report.diagnostics.length">
            No structural diagnostics. Runtime support has not been certified.
          </p>
          <div
            v-for="(item, i) in report.diagnostics"
            :key="i"
            class="model-diagnostic"
            :class="item.severity"
          >
            <strong
              ><AlertTriangle :size="12" />{{ item.severity.toUpperCase() }} ·
              {{ (item as any).code ?? "ANALYSIS" }}</strong
            >
            <p>{{ item.message }}</p>
            <small>{{ (item as any).source }}</small>
            <p class="muted small" v-if="(item as any).suggested_action">
              {{ (item as any).suggested_action }}
            </p>
          </div>
        </details>
      </template>
    </template>
  </aside>
</template>
