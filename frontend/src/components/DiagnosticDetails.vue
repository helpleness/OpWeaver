<script setup lang="ts">
import { computed, ref, watch } from "vue";
import type { DiagnosticRecord } from "../diagnostic-types";
const props = defineProps<{ rows: DiagnosticRecord[]; diagnosticId: string }>();
const chosen = ref("");
watch(
  () => props.rows,
  () => {
    if (!props.rows.some((v) => v.id === chosen.value))
      chosen.value = props.rows[0]?.id ?? "";
  },
  { immediate: true },
);
const row = computed(() => props.rows.find((v) => v.id === chosen.value));
const value = (x: unknown) =>
  typeof x === "number" && Number.isFinite(x)
    ? Number(x.toPrecision(6)).toString()
    : "—";
</script>
<template>
  <section
    v-if="rows.length"
    class="diagnostic-detail"
    aria-label="Diagnostic Tensor details"
  >
    <strong>Quantization diagnostic · {{ diagnosticId.slice(0, 8) }}</strong>
    <select
      v-if="rows.length > 1"
      aria-label="Diagnostic Tensor"
      v-model="chosen"
    >
      <option v-for="item in rows" :key="item.id" :value="item.id">
        {{ item.kind }} · {{ item.fp32_tensor }}
      </option>
    </select>
    <template v-if="row">
      <p>
        <span class="status">{{ row.match_status }}</span> ·
        {{ row.kind === "weight" ? "Weight Error" : "Activation Error" }}
      </p>
      <dl>
        <dt>FP32 Tensor</dt>
        <dd>{{ row.fp32_tensor }}</dd>
        <dt>INT8/DQ Tensor</dt>
        <dd>{{ row.quantized_tensor ?? "No reliable counterpart" }}</dd>
        <dt>Original inferred shape</dt>
        <dd>{{ row.fp32_shape ?? "unknown" }}</dd>
        <dt>Candidate inferred shape</dt>
        <dd>{{ row.quantized_shape ?? "unknown" }}</dd>
        <dt>Diagnostic Status</dt>
        <dd>{{ row.message }}</dd>
        <dt>Match Method</dt>
        <dd>{{ row.match_method }}</dd>
        <dt>Sample Count · valid / total feeds</dt>
        <dd>
          {{ row.aggregate?.valid_samples ?? 0 }} /
          {{ row.aggregate?.samples ?? 0 }}
        </dd>
      </dl>
      <div class="diagnostic-metrics">
        <span
          v-for="key in [
            'mae',
            'mse',
            'rmse',
            'max_abs_error',
            'mean_relative_error',
            'cosine_similarity',
          ]"
          :key="key"
          ><small>{{ key }}</small
          ><b>{{ value(row.aggregate?.[key]) }}</b></span
        >
      </div>
      <p v-if="!row.aggregate?.valid_samples">
        No numerical comparison is available. Missing metrics are not zero.
      </p>
      <p class="scope">
        {{ row.aggregate?.aggregation }}. High error may accumulate upstream;
        further verification is required.
      </p>
      <details v-if="row.quantization">
        <summary>Weight dtype, scale / zero point and axis</summary>
        <pre>{{ JSON.stringify(row.quantization, null, 2) }}</pre>
      </details>
      <details>
        <summary>Actual capture shapes & per-feed metrics</summary>
        <pre>{{ JSON.stringify(row.samples, null, 2) }}</pre>
      </details>
    </template>
  </section>
</template>
<style scoped>
.diagnostic-detail {
  margin: 12px 0;
  padding: 14px;
  border: 1px solid #37576a;
  border-radius: 9px;
  background: linear-gradient(130deg, #192c36, #192131);
  font-size: 11px;
  overflow-wrap: anywhere;
}
.diagnostic-detail strong {
  color: #b7d9e6;
}
.diagnostic-detail select {
  width: 100%;
  margin-top: 9px;
}
.status {
  color: #93d8c1;
}
.diagnostic-detail dl {
  display: grid;
  grid-template-columns: 110px 1fr;
  gap: 8px;
  margin: 10px 0;
}
.diagnostic-detail dt,
.scope {
  color: #899aaf;
}
.diagnostic-detail dd {
  margin: 0;
  color: #c9d6e5;
}
.diagnostic-metrics {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 6px;
}
.diagnostic-metrics span {
  padding: 8px;
  background: #101b26;
  border-radius: 5px;
  display: flex;
  flex-direction: column;
  gap: 5px;
}
.diagnostic-metrics small {
  color: #8ca8bb;
}
.diagnostic-detail pre {
  max-height: 260px;
  overflow: auto;
  font-size: 10px;
  white-space: pre-wrap;
}
.diagnostic-detail details {
  margin-top: 12px;
}
.scope {
  line-height: 1.6;
}
</style>
