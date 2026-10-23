<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref, watch } from "vue";
import * as monaco from "monaco-editor/editor/editor.api.js";
import "monaco-editor/languages/definitions/python/register.js";
import "monaco-editor/editor/contrib/find/browser/findController.js";
import EditorWorker from "monaco-editor/editor/editor.worker.js?worker";
import type { Diagnostic } from "../code-types";

const props = defineProps<{
  path: string;
  project: string;
  text: string;
  diagnostics: Diagnostic[];
  jump: { line: number; endLine?: number; token: number } | null;
}>();
const emit = defineEmits<{
  edit: [text: string];
  line: [line: number];
  save: [];
}>();
const host = ref<HTMLDivElement>();
const models = new Map<string, monaco.editor.ITextModel>();
const views = new Map<string, monaco.editor.ICodeEditorViewState | null>();
let editor: monaco.editor.IStandaloneCodeEditor;
let active = "";
let navigating = false;
(self as any).MonacoEnvironment = { getWorker: () => new EditorWorker() };

function activate() {
  if (!editor || !props.path) return;
  if (active) views.set(active, editor.saveViewState());
  const key = `${props.project}/${props.path}`;
  let model = models.get(key);
  if (!model) {
    model = monaco.editor.createModel(
      props.text,
      "python",
      monaco.Uri.parse(`opweaver-source:///${encodeURIComponent(key)}`),
    );
    models.set(key, model);
  } else if (model.getValue() !== props.text) model.setValue(props.text);
  active = key;
  navigating = true;
  editor.setModel(model);
  editor.restoreViewState(views.get(key) ?? null);
  navigating = false;
  mark();
}
function mark() {
  const model = editor?.getModel();
  if (!model) return;
  monaco.editor.setModelMarkers(
    model,
    "opweaver-python",
    props.diagnostics
      .filter((d) => d.line)
      .map((d) => ({
        severity:
          d.severity === "error"
            ? monaco.MarkerSeverity.Error
            : monaco.MarkerSeverity.Warning,
        message: d.message,
        startLineNumber: Math.min(d.line!, model.getLineCount()),
        endLineNumber: Math.min(d.line!, model.getLineCount()),
        startColumn: d.column ?? 1,
        endColumn: (d.column ?? 1) + 1,
      })),
  );
}
onMounted(() => {
  monaco.editor.defineTheme("opweaver-dark", {
    base: "vs-dark",
    inherit: true,
    rules: [
      { token: "keyword", foreground: "B9A0F5" },
      { token: "string", foreground: "ABD3B2" },
      { token: "comment", foreground: "627791" },
    ],
    colors: {
      "editor.background": "#101620",
      "editorLineNumber.foreground": "#4e617c",
      "editorLineNumber.activeForeground": "#b7a1e4",
      "editor.lineHighlightBackground": "#192131",
      "editor.selectionBackground": "#44375f80",
    },
  });
  editor = monaco.editor.create(host.value!, {
    theme: "opweaver-dark",
    automaticLayout: true,
    editContext: false,
    fontFamily: "Cascadia Code, Consolas, monospace",
    fontSize: 12,
    lineHeight: 20,
    minimap: { enabled: false },
    padding: { top: 14 },
    scrollBeyondLastLine: false,
    smoothScrolling: false,
    cursorSmoothCaretAnimation: "off",
    cursorBlinking: "blink",
    wordWrap: "off",
    renderWhitespace: "selection",
    tabSize: 4,
    ariaLabel: "Python source editor",
  });
  editor.onDidChangeModelContent(() => {
    if (!navigating) emit("edit", editor.getValue());
  });
  editor.onDidChangeCursorPosition((event) => {
    if (!navigating) emit("line", event.position.lineNumber);
  });
  editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS, () =>
    emit("save"),
  );
  activate();
});
watch(() => [props.path, props.project], activate);
watch(
  () => props.text,
  (content) => {
    const model = editor?.getModel();
    if (model && content !== model.getValue()) {
      navigating = true;
      model.pushEditOperations(
        [],
        [{ range: model.getFullModelRange(), text: content }],
        () => null,
      );
      navigating = false;
    }
  },
);
watch(() => props.diagnostics, mark, { deep: true });
watch(
  () => props.jump?.token,
  () => {
    if (!editor || !props.jump) return;
    navigating = true;
    editor.setSelection(
      new monaco.Range(
        props.jump.line,
        1,
        props.jump.endLine ?? props.jump.line,
        1000,
      ),
    );
    editor.revealLineInCenterIfOutsideViewport(props.jump.line);
    navigating = false;
  },
);
onBeforeUnmount(() => {
  editor?.dispose();
  models.forEach((model) => model.dispose());
});
</script>
<template>
  <div ref="host" class="python-editor" data-testid="python-editor"></div>
</template>
