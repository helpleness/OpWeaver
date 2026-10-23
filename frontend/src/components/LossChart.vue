<script setup lang="ts">
import { computed } from "vue";
import type { Metric } from "../types";
const props = defineProps<{ metrics: Metric[] }>();
const values = computed(() =>
  props.metrics.filter((m) => Number.isFinite(m.loss)),
);
const max = computed(
  () => Math.max(...values.value.map((m) => m.loss), 0.001) * 1.08,
);
const minStep = computed(() => values.value[0]?.step ?? 0);
const maxStep = computed(() => values.value.at(-1)?.step ?? 1);
const points = computed(() => {
  const stride = Math.max(1, Math.floor(values.value.length / 500));
  return values.value
    .filter((_, i) => i % stride === 0 || i === values.value.length - 1)
    .map(
      (m) =>
        `${52 + ((m.step - minStep.value) / Math.max(maxStep.value - minStep.value, 1)) * 700},${154 - (m.loss / max.value) * 126}`,
    )
    .join(" ");
});
</script>

<template>
  <div v-if="values.length" class="loss-chart">
    <svg
      viewBox="0 0 800 184"
      preserveAspectRatio="none"
      role="img"
      aria-label="Actual training loss by optimizer step"
    >
      <g v-for="fraction in [0, 0.25, 0.5, 0.75, 1]" :key="fraction">
        <line
          x1="52"
          :y1="154 - fraction * 126"
          x2="752"
          :y2="154 - fraction * 126"
          stroke="#242d39"
          stroke-dasharray="3 5"
        />
        <text x="40" :y="158 - fraction * 126" text-anchor="end">
          {{ (max * fraction).toFixed(2) }}
        </text>
      </g>
      <polyline
        :points="points"
        fill="none"
        stroke="#a89aff"
        stroke-width="2"
        vector-effect="non-scaling-stroke"
        stroke-linejoin="round"
      />
      <text x="52" y="178">{{ minStep }}</text>
      <text x="752" y="178" text-anchor="end">{{ maxStep }} steps</text>
    </svg>
  </div>
  <div v-else class="metrics-empty">
    <span class="chart-placeholder">↗</span>
    <div>
      <strong>Ready for your first experiment</strong>
      <p>Start training to see actual loss, accuracy and learning rate.</p>
    </div>
  </div>
</template>
