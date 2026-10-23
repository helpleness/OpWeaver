<script setup lang="ts">
import { computed } from "vue";
import { SlidersHorizontal, Box, Trash2, Check, X } from "lucide-vue-next";
import { useEditor } from "../stores/editor";

const editor = useEditor();
const emit = defineEmits<{ error: [message: string] }>();
const node = computed(() => editor.selectedNode);
const op = computed(() => editor.selectedOperator);
function change(key: string, event: Event, type?: string) {
  const target = event.target as HTMLInputElement;
  try {
    const value =
      type === "boolean"
        ? target.checked
        : type === "integer" || type === "number"
          ? Number(target.value)
          : type === "array"
            ? JSON.parse(target.value)
            : target.value;
    if (
      (type === "number" || type === "integer") &&
      (!target.value || !Number.isFinite(value))
    )
      throw new Error(`${key} must be a number`);
    if (node.value) editor.updateParams(node.value.id, key, value);
  } catch (error) {
    emit("error", `Invalid ${key}: ${String(error)}`);
  }
}
</script>

<template>
  <aside class="inspector">
    <div class="panel-heading">
      <SlidersHorizontal :size="15" /><strong>Inspector</strong
      ><span v-if="node" class="badge">NODE</span>
    </div>
    <template v-if="node && op">
      <div class="inspector-intro">
        <div class="large-node-icon"><Box :size="23" /></div>
        <div>
          <h2>{{ op.label }}</h2>
          <code>{{ op.id }}@{{ op.version }}</code>
        </div>
      </div>
      <div class="inspector-section">
        <label
          >Display name<input
            :value="node.label"
            @change="
              editor.renameNode(
                node.id,
                ($event.target as HTMLInputElement).value,
              )
            "
        /></label>
        <p class="description">{{ op.description }}</p>
      </div>
      <div class="section-label">
        PARAMETERS
        <span>{{ Object.keys(op.parameters.properties).length }}</span>
      </div>
      <div class="inspector-section parameter-list">
        <p v-if="!Object.keys(op.parameters.properties).length" class="muted">
          This operator has no parameters.
        </p>
        <label
          v-for="(schema, key) in op.parameters.properties"
          :key="key"
          :class="{ 'checkbox-label': schema.type === 'boolean' }"
        >
          <span>{{ key }}</span>
          <input
            v-if="schema.type === 'boolean'"
            type="checkbox"
            :checked="Boolean(node.params?.[key])"
            @change="change(String(key), $event, schema.type)"
          />
          <select
            v-else-if="key === 'dtype'"
            :value="node.params?.[key]"
            @change="change(String(key), $event, schema.type)"
          >
            <option>float32</option>
          </select>
          <select
            v-else-if="key === 'device'"
            :value="node.params?.[key]"
            @change="change(String(key), $event, schema.type)"
          >
            <option>any</option>
            <option>cpu</option>
            <option>cuda</option>
          </select>
          <input
            v-else
            :type="
              ['integer', 'number'].includes(schema.type ?? '')
                ? 'number'
                : 'text'
            "
            :value="
              schema.type === 'array'
                ? JSON.stringify(node.params?.[key])
                : node.params?.[key]
            "
            :min="schema.minimum"
            :max="schema.maximum"
            :step="schema.type === 'number' ? 0.01 : 1"
            @change="change(String(key), $event, schema.type)"
          />
        </label>
      </div>
      <div class="section-label">TENSOR CONTRACTS</div>
      <div class="inspector-section">
        <div
          v-for="port in [...(node.inputs ?? []), ...(node.outputs ?? [])]"
          :key="port.name"
          class="tensor-card"
        >
          <header>
            <span>{{ port.name }}</span
            ><small>{{
              node.inputs?.includes(port) ? "INPUT" : "OUTPUT"
            }}</small>
          </header>
          <template v-if="editor.validation.tensors[`${node.id}:${port.name}`]"
            ><code
              >[{{
                editor.validation.tensors[`${node.id}:${port.name}`].shape.join(
                  ", ",
                )
              }}]</code
            >
            <div>
              {{ editor.validation.tensors[`${node.id}:${port.name}`].dtype }}
              <span>·</span>
              {{ editor.validation.tensors[`${node.id}:${port.name}`].device }}
            </div></template
          >
          <span v-else class="muted">Waiting for valid connections</span>
        </div>
      </div>
      <div class="section-label">CAPABILITIES</div>
      <div class="inspector-section capabilities">
        <div><span>PyTorch Eager</span><Check :size="14" /></div>
        <div>
          <span>Autograd</span
          ><Check v-if="op.capabilities.autograd" :size="14" />
        </div>
        <div>
          <span>torch.export</span><span class="muted">Not verified</span>
        </div>
        <div>
          <span>ONNX / custom kernel</span><X :size="14" class="muted" />
        </div>
        <div>
          <span>Devices</span
          ><code>{{ op.capabilities.devices.join(" / ") }}</code>
        </div>
      </div>
      <div class="inspector-section">
        <button
          class="danger subtle full-width"
          @click="editor.removeNodes([node.id])"
        >
          <Trash2 :size="14" /> Delete node</button
        ><code class="node-id">{{ node.id }}</code>
      </div>
    </template>
    <div v-else class="inspector-empty">
      <SlidersHorizontal :size="28" />
      <h3>A closer look</h3>
      <p>
        Select a node to edit parameters,<br />inspect tensors and check
        capabilities.
      </p>
      <div class="keyboard-hint"><kbd>Shift</kbd> Multi-select</div>
      <div class="keyboard-hint"><kbd>Del</kbd> Remove selected</div>
      <div class="keyboard-hint"><kbd>Ctrl Z</kbd> Undo</div>
    </div>
  </aside>
</template>
