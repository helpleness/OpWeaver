<script setup lang="ts">
import {
  computed,
  defineAsyncComponent,
  onBeforeUnmount,
  ref,
  watch,
} from "vue";
import {
  Columns2,
  Braces,
  Workflow,
  FilePlus2,
  FileCode2 as FileCodeIcon,
  FolderOpen,
  Save,
  Play,
  Square,
  ArrowRight,
  X,
  ShieldCheck,
  PanelRightClose,
  PanelRightOpen,
  RotateCcw,
} from "lucide-vue-next";
import GraphCanvas from "./GraphCanvas.vue";
import { useEditor } from "../stores/editor";
import { useCode } from "../stores/code";
import { rpc } from "../api";

const PythonEditor = defineAsyncComponent(() => import("./PythonEditor.vue"));
const SourceDiff = defineAsyncComponent(() => import("./SourceDiff.vue"));
const props = defineProps<{ desktop?: boolean; inspectorVisible: boolean }>();
const emit = defineEmits<{ error: [message: string]; toggleInspector: [] }>();
const editor = useEditor();
const code = useCode();
const split = ref(48);
const container = ref<HTMLDivElement>();
const resizing = ref(false);
const modal = ref<"import" | "new" | "generate" | "">("");
const path = ref("");
const settings = ref(false);
const busy = ref(false);
const reloadConfirm = ref(false);
const diffReady = ref(false);
watch(
  () => code.proposal,
  () => (diffReady.value = false),
);
const activeDirty = computed(
  () => code.document && code.document.text !== code.document.saved,
);
let poll = setInterval(() => {
  void code.pollCapture().catch((error) => {
    code.error = String(error);
    code.status = "parse_failed";
  });
}, 700);
onBeforeUnmount(() => clearInterval(poll));
async function action(task: () => Promise<unknown>) {
  busy.value = true;
  code.error = "";
  try {
    await task();
  } catch (e) {
    code.error = String(e);
    emit("error", String(e));
  } finally {
    busy.value = false;
  }
}
function resize(event: PointerEvent) {
  if (!resizing.value || !container.value) return;
  const bounds = container.value.getBoundingClientRect();
  split.value = Math.min(
    75,
    Math.max(25, ((event.clientX - bounds.left) / bounds.width) * 100),
  );
}
function startResize(event: PointerEvent) {
  resizing.value = true;
  (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
  resize(event);
}
function showDialog(kind: "import" | "new" | "generate") {
  path.value =
    kind === "import"
      ? ""
      : kind === "new"
        ? "src/new_model.py"
        : "src/model.py";
  modal.value = kind;
}
async function submit() {
  await action(async () => {
    if (modal.value === "import") await code.importFile(path.value);
    else if (modal.value === "new") await code.newFile(path.value);
    else await code.previewGeneration(path.value);
    modal.value = "";
  });
}
async function apply() {
  await action(async () => {
    await code.applyProposal();
    code.view = "split";
  });
}
function setMode(value: string) {
  code.mode = value as "visual" | "code-first";
  code.status = code.lastResult ? "stale" : "pending";
  void code.persist();
}
</script>

<template>
  <section class="model-workbench" :class="[code.view, { resizing }]">
    <div class="model-viewbar">
      <div class="view-switch">
        <button
          title="Full graph view"
          :class="{ active: code.view === 'graph' }"
          @click="code.view = 'graph'"
        >
          <Workflow :size="13" />Graph</button
        ><button
          title="Graph and Python split view"
          :class="{ active: code.view === 'split' }"
          @click="code.view = 'split'"
        >
          <Columns2 :size="13" />Split</button
        ><button
          title="Full code view"
          :class="{ active: code.view === 'code' }"
          @click="code.view = 'code'"
        >
          <Braces :size="13" />Code
        </button>
      </div>
      <select
        aria-label="Model editing mode"
        :value="code.mode"
        @change="setMode(($event.target as HTMLSelectElement).value)"
      >
        <option value="visual">Visual Authoring</option>
        <option value="code-first">Code-First · analysis</option>
      </select>
      <span
        class="sync-status"
        :class="code.status"
        :title="`Code revision ${code.codeRevision} · Graph revision ${code.graphRevision} · Captured ${code.parsedHash.slice(0, 12) || 'none'}`"
        ><span
          class="live-dot"
          :class="{ warning: !['synced', 'read_only'].includes(code.status) }"
        ></span
        >{{ code.syncLabel }}</span
      >
      <button
        :title="inspectorVisible ? 'Hide inspector' : 'Show inspector'"
        @click="emit('toggleInspector')"
      >
        <PanelRightClose v-if="inspectorVisible" :size="14" /><PanelRightOpen
          v-else
          :size="14"
        />
      </button>
    </div>
    <div
      class="model-panels"
      :style="{ '--graph-share': `${split}%` }"
      ref="container"
    >
      <div class="graph-pane" v-show="code.view !== 'code'">
        <GraphCanvas @error="emit('error', $event)" />
      </div>
      <div
        v-show="code.view === 'split'"
        class="model-resize-handle"
        role="separator"
        aria-label="Resize graph and code panels"
        aria-orientation="vertical"
        tabindex="0"
        @pointerdown="startResize"
        @pointermove="resize"
        @pointerup="resizing = false"
        @pointercancel="resizing = false"
        @keydown.left.prevent="split = Math.max(25, split - 5)"
        @keydown.right.prevent="split = Math.min(75, split + 5)"
      ></div>
      <div class="code-pane" v-show="code.view !== 'graph'">
        <div class="source-filebar">
          <FileCodeIcon /><select
            aria-label="Python project file"
            :value="code.activeFile"
            :disabled="code.running"
            @change="
              action(() =>
                code.open(($event.target as HTMLSelectElement).value),
              )
            "
          >
            <option v-if="!code.files.length" value="">No Python source</option>
            <option v-for="file in code.files" :key="file" :value="file">
              {{ file
              }}{{
                code.documents[file] &&
                code.documents[file].text !== code.documents[file].saved
                  ? " •"
                  : ""
              }}
            </option></select
          ><button
            title="New Python file"
            :disabled="!editor.projectPath"
            @click="showDialog('new')"
          >
            <FilePlus2 :size="13" /></button
          ><button
            title="Import Python file"
            :disabled="!editor.projectPath"
            @click="showDialog('import')"
          >
            <FolderOpen :size="13" /></button
          ><button
            title="Save Python file"
            :disabled="!code.activeFile || busy"
            @click="action(code.saveActive)"
          >
            <Save :size="13" />
          </button>
          <button
            title="Reload Python file from disk"
            :disabled="!code.activeFile || code.running"
            @click="reloadConfirm = true"
          >
            <RotateCcw :size="13" />
          </button>
        </div>
        <div class="capture-toolbar">
          <select
            aria-label="Model class or factory"
            v-model="code.entry"
            :disabled="code.running"
          >
            <option v-if="!code.entries.length">{{ code.entry }}</option>
            <option
              v-for="entry in code.entries"
              :key="entry.name"
              :value="entry.name"
            >
              {{ entry.name }}
            </option></select
          ><button
            class="subtle"
            :class="{ active: settings }"
            title="Constructor and example inputs"
            @click="settings = !settings"
          >
            Inputs</button
          ><button
            v-if="code.running"
            class="stop-button"
            @click="action(code.stopCapture)"
          >
            <Square :size="11" />Cancel</button
          ><button
            v-else
            class="primary"
            :disabled="!code.activeFile || busy"
            @click="action(code.startCapture)"
          >
            <Play :size="11" />Parse graph
          </button>
        </div>
        <div class="capture-settings" v-if="settings">
          <label
            >Constructor kwargs (JSON)<textarea
              aria-label="Constructor kwargs"
              v-model="code.constructor"
              rows="2"
            ></textarea></label
          ><label
            >Example input specs (JSON)<textarea
              aria-label="Example input specs"
              v-model="code.inputConfig"
              rows="3"
            ></textarea>
          </label>
          <p>
            Concrete shape executes on CPU. Optional design_shape may use B for
            the batch axis. Capture uses eval mode.
          </p>
        </div>
        <div class="code-sync-note" v-if="!editor.projectPath">
          Create a project to edit and save Python files.
        </div>
        <PythonEditor
          v-else-if="code.activeFile"
          :key="editor.projectPath"
          :project="editor.projectPath"
          :path="code.activeFile"
          :text="code.text"
          :diagnostics="code.diagnostics"
          :jump="code.jump"
          @edit="code.edit"
          @line="code.locateLine"
          @save="action(code.saveActive)"
        />
        <div v-else class="source-empty">
          <Braces :size="32" />
          <h3>Code is a first-class citizen.</h3>
          <p>
            Generate a real PyTorch model from your graph,<br />or import a
            Python file to inspect and edit it.
          </p>
          <button
            class="primary"
            :disabled="!editor.projectPath || code.mode === 'code-first'"
            @click="showDialog('generate')"
          >
            Generate from graph <ArrowRight :size="13" /></button
          ><button
            class="subtle"
            :disabled="!editor.projectPath"
            @click="showDialog('import')"
          >
            Import Python file
          </button>
        </div>
        <div class="source-actionbar">
          <button
            class="subtle"
            :disabled="
              code.mode === 'code-first' || !editor.projectPath || code.running
            "
            @click="showDialog('generate')"
          >
            Graph → Code · review diff</button
          ><span :class="{ unsaved: activeDirty }">{{
            activeDirty
              ? "Unsaved source"
              : code.activeFile
                ? "Source saved"
                : "No source"
          }}</span>
        </div>
        <div class="execution-boundary">
          <ShieldCheck :size="11" />Open/save never execute. Parse runs local
          Python in a disposable process; this is not a sandbox.
        </div>
      </div>
    </div>
    <div v-if="code.error" class="code-error" role="alert">
      {{ code.error
      }}<button title="Dismiss code error" @click="code.error = ''">
        <X :size="12" />
      </button>
    </div>
    <div class="source-diagnostics" v-if="code.diagnostics.length">
      <button
        v-for="(item, i) in code.diagnostics"
        :key="i"
        @click="
          item.line && (code.jump = { line: item.line, token: Date.now() })
        "
      >
        <span>{{ item.severity.toUpperCase() }}</span
        >{{ item.message }}<small v-if="item.line">line {{ item.line }}</small>
      </button>
    </div>
  </section>
  <div v-if="modal" class="modal-backdrop" @click.self="modal = ''">
    <section class="modal" role="dialog" aria-modal="true">
      <button
        class="modal-close"
        title="Close source dialog"
        @click="modal = ''"
      >
        <X :size="17" />
      </button>
      <h2>
        {{
          modal === "import"
            ? "Import Python source"
            : modal === "new"
              ? "New Python file"
              : "Generate PyTorch code"
        }}
      </h2>
      <p>
        {{
          modal === "import"
            ? "The file is copied into this project. Opening it does not execute Python."
            : modal === "generate"
              ? "Review the complete diff before writing. Hand-edited files require a new output filename."
              : "Create an editable file inside the project src directory."
        }}
      </p>
      <label
        >{{ modal === "import" ? "Python source path" : "Project output path" }}
        <div class="input-with-button">
          <input v-model="path" autofocus /><button
            v-if="modal === 'import' && desktop"
            title="Browse Python source"
            @click="
              action(async () => {
                const selected = await rpc('dialog.python');
                if (selected) path = selected;
              })
            "
          >
            <FolderOpen :size="15" />
          </button></div></label
      ><button
        class="primary full-width"
        :disabled="busy || !path"
        @click="submit"
      >
        {{
          modal === "generate"
            ? "Preview source diff"
            : modal === "import"
              ? "Import source"
              : "Create file"
        }}
      </button>
      <p class="modal-error" v-if="code.error">{{ code.error }}</p>
    </section>
  </div>
  <div v-if="code.proposal" class="modal-backdrop diff-backdrop">
    <section
      class="diff-dialog"
      role="dialog"
      aria-modal="true"
      aria-label="Review generated source"
    >
      <header>
        <div>
          <h2>
            {{
              code.proposal.kind === "patch"
                ? "Review local source patch"
                : "Review generated Python"
            }}
          </h2>
          <p>
            {{ code.proposal.path }} · existing source is protected by a content
            hash
          </p>
        </div>
        <button
          title="Cancel source diff"
          :disabled="!diffReady"
          @click="code.proposal = null"
        >
          <X :size="17" />
        </button>
      </header>
      <SourceDiff
        :before="code.proposal.before"
        :after="code.proposal.text"
        @ready="diffReady = true"
      />
      <footer>
        <span>Left: current source · Right: proposed source</span
        ><button :disabled="!diffReady" @click="code.proposal = null">
          Cancel</button
        ><button class="primary" :disabled="busy || !diffReady" @click="apply">
          {{ diffReady ? "Apply reviewed change" : "Computing diff…" }}
        </button>
      </footer>
      <p v-if="code.error" class="modal-error">{{ code.error }}</p>
    </section>
  </div>
  <div v-if="reloadConfirm" class="modal-backdrop">
    <section class="modal" role="dialog" aria-modal="true">
      <h2>Reload saved source?</h2>
      <p>
        The editor buffer will be replaced by the file on disk. Keep editing to
        preserve your current unsaved buffer.
      </p>
      <div class="modal-actions">
        <button @click="reloadConfirm = false">Keep editing</button
        ><button
          class="primary"
          @click="
            action(async () => {
              await code.reloadActive();
              reloadConfirm = false;
            })
          "
        >
          Reload saved source
        </button>
      </div>
    </section>
  </div>
</template>
