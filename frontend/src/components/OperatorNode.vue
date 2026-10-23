<script setup lang="ts">
import { computed } from "vue";
import { Handle, Position } from "@vue-flow/core";
import {
  Box,
  ArrowDownToLine,
  ArrowUpFromLine,
  Plus,
  Activity,
  Layers,
} from "lucide-vue-next";
import { useEditor } from "../stores/editor";
import type { AnalysisNode } from "../code-types";

const props = defineProps<{
  id: string;
  selected?: boolean;
  analysis?: AnalysisNode;
}>();
const editor = useEditor();
const node = computed(() =>
  props.analysis
    ? {
        id: props.id,
        label: props.analysis.label,
        operator: { id: props.analysis.operator },
        params: props.analysis.parameters,
        inputs: props.analysis.inputs.map((name) => ({ name })),
        outputs: props.analysis.outputs.map((name) => ({ name })),
      }
    : editor.graph.nodes?.find((n) => n.id === props.id),
);
const op = computed(() =>
  editor.registry.find((o) => o.id === node.value?.operator.id),
);
const tensor = computed(() =>
  props.analysis?.tensors.shape
    ? {
        shape: props.analysis.tensors.shape,
        dtype: props.analysis.tensors.dtype,
        device: props.analysis.tensors.device,
      }
    : editor.validation.tensors[
        `${props.id}:${node.value?.outputs?.[0]?.name ?? "in"}`
      ],
);
const errors = computed(() =>
  props.analysis
    ? []
    : editor.validation.issues.filter((i) => i.node === props.id),
);
const icon = computed(
  () =>
    ({
      Structure:
        node.value?.operator.id === "core.input"
          ? ArrowUpFromLine
          : ArrowDownToLine,
      Activation: Activity,
      Tensor: Plus,
      Layers,
    })[op.value?.category ?? ""] ?? Box,
);
const summary = computed(() => {
  const p = node.value?.params ?? {};
  if (props.analysis) return props.analysis.module_path || props.analysis.kind;
  if (node.value?.operator.id === "core.linear")
    return `${p.in_features} → ${p.out_features} · ${p.bias ? "bias" : "no bias"}`;
  if (node.value?.operator.id === "core.dropout") return `probability = ${p.p}`;
  return op.value?.id.replace("core.", "torch · ") ?? "Unknown operator";
});
</script>

<template>
  <div
    class="operator-node"
    :class="[
      { selected, invalid: errors.length > 0 },
      op?.category?.toLowerCase(),
    ]"
    :title="errors.map((e) => e.message).join('\n')"
  >
    <div class="node-heading">
      <span class="node-icon"><component :is="icon" :size="15" /></span
      ><strong>{{ node?.label || op?.label }}</strong
      ><span class="node-type">{{ op?.version }}</span>
    </div>
    <div class="node-description">{{ summary }}</div>
    <div class="node-tensor">
      <span class="tensor-dot"></span
      >{{ tensor ? `[${tensor.shape.join(", ")}]` : "shape pending"
      }}<span>{{ tensor?.dtype ?? "—" }}</span>
    </div>
    <template v-for="(port, index) in node?.inputs" :key="`in-${port.name}`">
      <Handle
        :id="port.name"
        type="target"
        :connectable="!analysis"
        :position="Position.Left"
        :style="{ top: `${48 + index * 24}px` }"
        :aria-label="`${node?.label} input ${port.name}`"
      />
      <span
        v-if="(node?.inputs?.length ?? 0) > 1"
        class="port-label"
        :style="{ top: `${42 + index * 24}px` }"
        >{{ port.name }}</span
      >
    </template>
    <Handle
      v-for="(port, index) in node?.outputs"
      :id="port.name"
      :key="`out-${port.name}`"
      type="source"
      :connectable="!analysis"
      :position="Position.Right"
      :style="{ top: `${48 + index * 24}px` }"
      :aria-label="`${node?.label} output ${port.name}`"
    />
    <div v-if="errors.length" class="node-error">
      {{ errors.length }} issue{{ errors.length > 1 ? "s" : "" }}
    </div>
  </div>
</template>
