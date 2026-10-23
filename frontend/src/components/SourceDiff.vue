<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from "vue";
import * as monaco from "monaco-editor/editor/editor.api.js";
import "monaco-editor/languages/definitions/python/register.js";
import EditorWorker from "monaco-editor/editor/editor.worker.js?worker";
const props = defineProps<{ before: string; after: string }>();
const emit = defineEmits<{ ready: [] }>();
const host = ref<HTMLDivElement>();
let editor: monaco.editor.IStandaloneDiffEditor;
let beforeModel: monaco.editor.ITextModel;
let afterModel: monaco.editor.ITextModel;
(self as any).MonacoEnvironment = { getWorker: () => new EditorWorker() };
onMounted(() => {
  beforeModel = monaco.editor.createModel(props.before, "python");
  afterModel = monaco.editor.createModel(props.after, "python");
  editor = monaco.editor.createDiffEditor(host.value!, {
    automaticLayout: true,
    theme: "vs-dark",
    readOnly: true,
    originalEditable: false,
    renderSideBySide: true,
    minimap: { enabled: false },
    fontSize: 12,
    scrollBeyondLastLine: false,
    renderOverviewRuler: false,
  });
  editor.onDidUpdateDiff(() => emit("ready"));
  editor.setModel({ original: beforeModel, modified: afterModel });
});
onBeforeUnmount(() => {
  editor?.dispose();
  beforeModel?.dispose();
  afterModel?.dispose();
});
</script>
<template>
  <div class="source-diff" ref="host" aria-label="Source change preview"></div>
</template>
