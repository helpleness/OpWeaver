<script setup lang="ts">
import { computed } from "vue";
import { metric } from "../lab-types";
const props = defineProps<{ baseline: number[]; optimized?: number[] }>();
const maximum = computed(() =>
  Math.max(...props.baseline, ...(props.optimized ?? []), 0.000001),
);
function points(samples: number[]) {
  // Bound SVG point count; full unmodified samples remain in the report.
  const stride = Math.max(1, Math.ceil(samples.length / 500));
  return samples
    .filter((_, i) => i % stride === 0)
    .map(
      (v, i) =>
        `${36 + ((i * stride) / Math.max(1, samples.length - 1)) * 600},${110 - (v / maximum.value) * 90}`,
    )
    .join(" ");
}
</script>
<template>
  <div class="latency-chart">
    <div class="lab-card-heading">
      <strong>Measured inference samples</strong
      ><span
        ><i class="baseline-dot" />Baseline
        <i class="optimized-dot" />Optimized</span
      >
    </div>
    <svg
      viewBox="0 0 660 137"
      role="img"
      aria-label="Raw inference latency samples in milliseconds"
    >
      <path d="M36 16V110H640 M36 65H640" fill="none" stroke="#293341" />
      <text x="35" y="12">{{ metric(maximum) }} ms</text>
      <text x="8" y="113">0</text>
      <text x="36" y="130">Sample 1</text>
      <text x="550" y="130">{{ baseline.length }} samples</text>
      <polyline
        :points="points(baseline)"
        fill="none"
        stroke="#93a8da"
        stroke-width="1.3"
      />
      <polyline
        v-if="optimized?.length"
        :points="points(optimized)"
        fill="none"
        stroke="#7dceb0"
        stroke-width="1.3"
      />
    </svg>
    <p class="muted small">
      Session initialization and input preparation excluded. Sequential CPU
      measurements; no concurrent throughput claim.
    </p>
  </div>
</template>
