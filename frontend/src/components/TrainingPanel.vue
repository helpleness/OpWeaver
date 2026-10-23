<script setup lang="ts">
import { computed } from "vue";
import {
  Play,
  RotateCcw,
  Database,
  Cpu,
  FileBox,
  SlidersHorizontal,
} from "lucide-vue-next";
import { useEditor } from "../stores/editor";
import LossChart from "./LossChart.vue";

defineProps<{ cudaAvailable: boolean }>();
const emit = defineEmits<{ train: []; resume: [path: string] }>();
const editor = useEditor();
const metric = computed(() => editor.job?.metrics.at(-1));
</script>

<template>
  <div class="training-workspace">
    <div class="workspace-heading">
      <div>
        <span class="eyebrow">EXPERIMENT WORKBENCH</span>
        <h1>Train what you build.</h1>
        <p>Real PyTorch training. Reproducible data. Resumable state.</p>
      </div>
      <span class="pill"
        ><Cpu :size="13" /> {{ editor.config.device.toUpperCase() }} /
        EAGER</span
      >
    </div>
    <div class="training-columns">
      <section class="training-card">
        <h3><SlidersHorizontal :size="16" /> Run configuration</h3>
        <div class="form-grid">
          <label
            >Optimizer<select
              v-model="editor.config.optimizer"
              :disabled="editor.running"
            >
              <option>AdamW</option>
              <option>SGD</option>
            </select></label
          >
          <label
            >Loss function<select
              v-model="editor.config.loss"
              :disabled="editor.running"
            >
              <option>CrossEntropyLoss</option>
              <option>MSELoss</option>
            </select></label
          >
          <label
            >Total epochs<input
              v-model.number="editor.config.epochs"
              type="number"
              min="1"
              max="10000"
              :disabled="editor.running"
          /></label>
          <label
            >Batch size<input
              v-model.number="editor.config.batch_size"
              type="number"
              min="1"
              :disabled="editor.running"
          /></label>
          <label
            >Learning rate<input
              v-model.number="editor.config.learning_rate"
              type="number"
              min="0.000001"
              step="0.001"
              :disabled="editor.running"
          /></label>
          <label
            >Random seed<input
              v-model.number="editor.config.seed"
              type="number"
              min="0"
              :disabled="editor.running"
          /></label>
          <label
            >Samples<input
              v-model.number="editor.config.samples"
              type="number"
              min="16"
              :disabled="editor.running"
          /></label>
          <label
            >Device<select
              v-model="editor.config.device"
              :disabled="editor.running"
            >
              <option value="cpu">CPU</option>
              <option value="cuda" :disabled="!cudaAvailable">
                {{ cudaAvailable ? "CUDA GPU" : "CUDA unavailable" }}
              </option>
            </select></label
          >
        </div>
        <div class="dataset-note">
          <Database :size="17" />
          <p>
            <strong>Offline synthetic dataset</strong>Fixed-seed random inputs
            with a linear teacher. Input [B, features] → output [B, classes].
            MSE uses continuous targets.
          </p>
        </div>
        <button
          class="primary full-width"
          :disabled="editor.running"
          @click="emit('train')"
        >
          <Play :size="14" /> Start experiment
        </button>
      </section>
      <section class="training-card run-overview">
        <div class="card-title">
          <h3>Live learning curve</h3>
          <span
            class="pill"
            :class="{ positive: editor.job?.state === 'COMPLETED' }"
            >{{ editor.job?.state ?? "IDLE" }}</span
          >
        </div>
        <div class="run-stats">
          <div>
            <span>LOSS</span
            ><strong>{{ metric?.loss.toFixed(4) ?? "—" }}</strong>
          </div>
          <div>
            <span>ACCURACY</span
            ><strong>{{
              metric?.accuracy != null
                ? `${(metric.accuracy * 100).toFixed(1)}%`
                : "—"
            }}</strong>
          </div>
          <div>
            <span>STEP</span><strong>{{ metric?.step ?? "—" }}</strong>
          </div>
        </div>
        <LossChart :metrics="editor.job?.metrics ?? []" />
        <p class="muted">
          Per-batch training metrics · no validation split in the synthetic
          adapter.
        </p>
      </section>
    </div>
    <section class="training-card experiments">
      <div class="card-title">
        <h3><FileBox :size="16" /> Experiments & checkpoints</h3>
        <button class="subtle" @click="editor.loadHistory">
          <RotateCcw :size="13" /> Refresh
        </button>
      </div>
      <p v-if="!editor.history.length" class="muted">
        Experiments are saved to this project's SQLite database. Checkpoints are
        written at each epoch and when stopped.
      </p>
      <div v-for="run in editor.history" :key="run.id" class="experiment-row">
        <div>
          <code>{{ run.id.slice(0, 8) }}</code
          ><small>{{ new Date(run.created_at * 1000).toLocaleString() }}</small>
        </div>
        <span>{{ run.config.optimizer }} · {{ run.config.epochs }} epochs</span
        ><span class="pill">{{ run.state }}</span
        ><button
          class="subtle"
          :disabled="!run.checkpoint || editor.running"
          @click="run.checkpoint && emit('resume', run.checkpoint)"
        >
          <RotateCcw :size="13" /> Resume
        </button>
      </div>
    </section>
  </div>
</template>
