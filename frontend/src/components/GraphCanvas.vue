<script setup lang="ts">
import { computed, nextTick, ref, shallowRef, watch } from "vue";
import {
  VueFlow,
  useVueFlow,
  type Connection,
  type NodeChange,
  type EdgeChange,
  type NodeDragEvent,
  type Node,
} from "@vue-flow/core";
import { Background } from "@vue-flow/background";
import { Controls } from "@vue-flow/controls";
import { MiniMap } from "@vue-flow/minimap";
import {
  Maximize,
  LayoutGrid,
  Undo2,
  Redo2,
  MousePointer2,
  LockKeyhole,
} from "lucide-vue-next";
import OperatorNode from "./OperatorNode.vue";
import { useEditor } from "../stores/editor";
import { useCode } from "../stores/code";

const editor = useEditor();
const code = useCode();
const emit = defineEmits<{ error: [message: string] }>();
const { screenToFlowCoordinate, fitView, getNodes } = useVueFlow({
  id: "model",
});
const dragging = ref(false);
const nodes = shallowRef<Node[]>([]);
const analysisPositions = ref<Record<string, { x: number; y: number }>>({});
const analysis = computed(() => code.activeAnalysis);
const readonly = computed(() => code.readOnly);

// Transient positions belong to Vue Flow until release. Feeding saved coordinates
// back during pointer selection was resetting the node beneath the cursor.
function syncNodes() {
  if (readonly.value && !analysis.value) {
    nodes.value = [];
    return;
  }
  const live = new Map(getNodes.value.map((n) => [n.id, n]));
  if (analysis.value) {
    nodes.value = analysis.value.nodes.map((node, i) => ({
      id: node.id,
      type: "operator",
      position:
        dragging.value && live.get(node.id)
          ? { ...live.get(node.id)!.position }
          : (analysisPositions.value[node.id] ?? {
              x: 60 + (i % 4) * 245,
              y: 80 + Math.floor(i / 4) * 190,
            }),
      data: { analysis: node },
      selected:
        code.analysisSelection.includes(node.id) ||
        code.selectedAnalysis === node.id,
    }));
  } else {
    nodes.value = (editor.graph.nodes ?? []).map((node, i) => ({
      id: node.id,
      type: "operator",
      position:
        dragging.value && live.get(node.id)
          ? { ...live.get(node.id)!.position }
          : (editor.layout.positions[node.id] ?? { x: 60 + i * 235, y: 150 }),
      data: {},
      selected: editor.selection.includes(node.id),
    }));
  }
}
watch(
  () => [
    editor.graph.nodes?.map((n) => n.id).join("|"),
    editor.layout.positions,
    editor.selection,
    analysis.value,
    code.selectedAnalysis,
    code.analysisSelection,
  ],
  syncNodes,
  { deep: true, immediate: true },
);
const edges = computed(() =>
  (readonly.value
    ? (analysis.value?.edges ?? [])
    : (editor.graph.edges ?? [])
  ).map((edge) => ({
    id: edge.id,
    source: edge.source,
    target: edge.target,
    sourceHandle: edge.source_port,
    targetHandle: edge.target_port,
    type: "smoothstep",
    animated: !analysis.value && editor.running,
    style: {
      stroke:
        !analysis.value &&
        editor.validation.issues.some((i) => i.edge === edge.id)
          ? "#ee7b85"
          : "#65758b",
      strokeWidth: 1.5,
    },
  })),
);
function connect(connection: Connection) {
  if (readonly.value) return;
  try {
    editor.connect(
      connection.source,
      connection.target,
      connection.sourceHandle ?? "out",
      connection.targetHandle ?? "in",
    );
  } catch (error) {
    emit("error", String(error));
  }
}
function drop(event: DragEvent) {
  event.preventDefault();
  if (readonly.value) {
    emit(
      "error",
      "Code-First graphs are read-only. Edit the source and parse again.",
    );
    return;
  }
  const id = event.dataTransfer?.getData("application/opweaver-operator");
  if (id)
    editor.addNode(
      id,
      screenToFlowCoordinate({ x: event.clientX - 90, y: event.clientY - 25 }),
    );
}
function nodesChange(changes: NodeChange[]) {
  if (readonly.value) return;
  editor.removeNodes(
    changes.filter((c) => c.type === "remove").map((c) => c.id),
  );
  const selection = new Set(editor.selection);
  for (const change of changes)
    if (change.type === "select") {
      if (change.selected) selection.add(change.id);
      else selection.delete(change.id);
    }
  if (JSON.stringify([...selection]) !== JSON.stringify(editor.selection))
    editor.selection = [...selection];
}
function edgesChange(changes: EdgeChange[]) {
  if (!readonly.value)
    editor.removeEdges(
      changes.filter((c) => c.type === "remove").map((c) => c.id),
    );
}
function dragStart() {
  dragging.value = true;
}
function dragStop(event: NodeDragEvent) {
  const positions = Object.fromEntries(
    event.nodes.map((n) => [n.id, { ...n.position }]),
  );
  if (analysis.value) Object.assign(analysisPositions.value, positions);
  else editor.moveNodes(positions);
  dragging.value = false;
  syncNodes();
}
function select(id: string) {
  if (analysis.value) {
    code.selectedAnalysis = id;
    code.locateNode(id);
  } else editor.selected = id;
}
function arrange() {
  const order =
    analysis.value?.nodes.map((n) => n.id) ??
    (editor.validation.order.length === editor.graph.nodes?.length
      ? editor.validation.order
      : (editor.graph.nodes?.map((n) => n.id) ?? []));
  const positions = Object.fromEntries(
    order.map((id, i) => [
      id,
      { x: 60 + (i % 4) * 245, y: 80 + Math.floor(i / 4) * 210 },
    ]),
  );
  if (analysis.value) {
    analysisPositions.value = positions;
    syncNodes();
  } else editor.moveNodes(positions);
  void nextTick(() => fitView({ padding: 0.2, duration: 150 }));
}
watch(
  () => analysis.value?.source_hash ?? editor.graph.id,
  () => {
    analysisPositions.value = {};
    setTimeout(() => fitView({ padding: 0.2, duration: 150 }), 100);
  },
);
defineExpose({ fit: () => fitView({ padding: 0.2 }) });
</script>

<template>
  <div
    class="canvas-wrap"
    :class="{ 'is-dragging': dragging, 'analysis-canvas': readonly }"
    @dragover.prevent
    @drop="drop"
  >
    <div class="canvas-toolbar">
      <span
        ><LockKeyhole v-if="readonly" :size="13" /><MousePointer2
          v-else
          :size="13"
        />{{ readonly ? "Analyze" : "Select" }}</span
      ><i></i>
      <button
        title="Undo (Ctrl+Z)"
        :disabled="readonly || !editor.past.length"
        @click="editor.undo"
      >
        <Undo2 :size="15" />
      </button>
      <button
        title="Redo (Ctrl+Shift+Z)"
        :disabled="readonly || !editor.future.length"
        @click="editor.redo"
      >
        <Redo2 :size="15" /></button
      ><i></i>
      <button title="Arrange nodes" @click="arrange">
        <LayoutGrid :size="15" /></button
      ><button
        title="Fit graph"
        @click="fitView({ padding: 0.2, duration: 150 })"
      >
        <Maximize :size="15" />
      </button>
    </div>
    <div class="canvas-caption">
      <span
        class="live-dot"
        :class="{
          warning: analysis
            ? code.status === 'stale' || code.status === 'parse_failed'
            : !editor.validation.valid,
        }"
      ></span
      >{{
        analysis
          ? `${analysis.nodes.length} nodes · FX analysis`
          : editor.validating
            ? "Checking graph…"
            : editor.validation.valid
              ? "Graph is valid"
              : `${editor.validation.issues.length} graph issues`
      }}<span>{{ analysis ? "CPU eval · read-only" : "DAG · Eager" }}</span>
    </div>
    <VueFlow
      id="model"
      :nodes="nodes"
      :edges="edges"
      :snap-to-grid="false"
      :min-zoom="0.15"
      :max-zoom="2"
      :default-viewport="{ x: 20, y: 60, zoom: 0.8 }"
      fit-view-on-init
      :delete-key-code="readonly ? null : ['Backspace', 'Delete']"
      :nodes-connectable="!readonly"
      :edges-updatable="!readonly"
      :multi-selection-key-code="'Shift'"
      selection-key-code="Shift"
      :node-drag-threshold="3"
      @connect="connect"
      @node-click="({ node }) => select(node.id)"
      @pane-click="editor.selected = null"
      @nodes-change="nodesChange"
      @edges-change="edgesChange"
      @node-drag-start="dragStart"
      @node-drag-stop="dragStop"
    >
      <template #node-operator="props"
        ><OperatorNode
          :id="props.id"
          :selected="props.selected"
          :analysis="props.data.analysis"
      /></template>
      <Background :gap="20" :size="1" pattern-color="#29313f" /><Controls
        position="bottom-left"
        :show-interactive="false"
      />
      <MiniMap
        position="bottom-right"
        :pannable="true"
        :zoomable="true"
        node-color="#536279"
        mask-color="rgba(11, 14, 19, 0.65)"
      />
      <div v-if="!nodes.length" class="canvas-empty">
        <div class="empty-graph-icon">⌘</div>
        <h2>
          {{
            readonly ? "Capture your Python model." : "Your model starts here."
          }}
        </h2>
        <p>
          {{
            readonly
              ? "Open source, choose a model entry and parse explicitly."
              : "Drag an Input from the library, then connect layers."
          }}
        </p>
        <span>Drag to pan · Scroll to zoom · Shift + drag to select</span>
      </div>
    </VueFlow>
  </div>
</template>
