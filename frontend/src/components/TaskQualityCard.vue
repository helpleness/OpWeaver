<script setup lang="ts">
import { computed } from "vue";
import { metric } from "../lab-types";
const props = defineProps<{ single?: any; paired?: any; final?: any }>();
const left = computed(() => props.paired?.baseline_metrics ?? props.single);
const right = computed(() => props.paired?.optimized_metrics);
const accuracy = (value: unknown) =>
  typeof value === "number" ? `${metric(value * 100, 3)}%` : "—";
</script>
<template>
  <section
    v-if="left?.status === 'evaluated'"
    class="task-quality-card"
    aria-label="Classification task quality"
  >
    <div class="quality-title">
      <strong>Classification quality</strong
      ><span
        >{{ left.samples }} labeled examples · {{ left.classes }} score
        columns</span
      >
    </div>
    <table class="lab-table">
      <thead>
        <tr>
          <th>Measured metric</th>
          <th>Original baseline</th>
          <th v-if="right">Candidate</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Top1 accuracy</td>
          <td>{{ accuracy(left.top1_accuracy) }}</td>
          <td v-if="right">{{ accuracy(right.top1_accuracy) }}</td>
        </tr>
        <tr>
          <td>Top5 accuracy</td>
          <td>{{ accuracy(left.top5_accuracy) }}</td>
          <td v-if="right">{{ accuracy(right.top5_accuracy) }}</td>
        </tr>
        <tr>
          <td>Correct / incorrect</td>
          <td>{{ left.correct }} / {{ left.incorrect }}</td>
          <td v-if="right">{{ right.correct }} / {{ right.incorrect }}</td>
        </tr>
      </tbody>
    </table>
    <p v-if="paired">
      Top1 drop
      {{ metric(paired.accuracy_drop_percentage_points, 3) }} percentage points
      · relative change {{ metric(paired.accuracy_change_percent, 3)
      }}{{ paired.accuracy_change_percent == null ? "" : "%" }}
    </p>
    <p v-else>
      Original model evaluation only. Candidate quality has not been measured.
    </p>
    <p v-if="left.top5_status !== 'evaluated'">
      Top5 is not applicable when the model has fewer than 5 classes.
    </p>
    <p class="muted small">{{ left.postprocessing }}. {{ left.scope }}.</p>
    <details>
      <summary>Labels, preprocessing reference & batch predictions</summary>
      <pre>{{
        JSON.stringify({ baseline: left, candidate: right }, null, 2)
      }}</pre>
    </details>
    <details v-if="final">
      <summary>Held-out final evaluation · {{ final.status }}</summary>
      <p>Final results are recorded separately from tuning/validation.</p>
      <pre>{{ JSON.stringify(final, null, 2) }}</pre>
    </details>
  </section>
</template>
<style scoped>
.task-quality-card {
  margin: 14px 0;
  border: 1px solid #3b5e71;
  background: linear-gradient(120deg, #182e38, #1c2639);
  padding: 16px;
  border-radius: 10px;
}
.quality-title {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 10px;
}
.quality-title span,
.task-quality-card p {
  font-size: 12px;
  color: #b1cbd6;
}
.task-quality-card pre {
  font-size: 10px;
  overflow: auto;
  max-height: 300px;
}
</style>
