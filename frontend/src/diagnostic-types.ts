import type {
  DiagnosticConfig as ConfigSchema,
  DiagnosticResult as ResultSchema,
  LayerErrorRecord as RecordSchema,
} from "./generated/quantization-diagnostic";

export type DiagnosticConfig = Required<ConfigSchema>;
export type DiagnosticRecord = Omit<
  Required<RecordSchema>,
  "aggregate" | "samples" | "quantization"
> & {
  aggregate: Record<string, any> | null;
  samples: Record<string, any>[];
  quantization: Record<string, any> | null;
};
export type DiagnosticReport = Omit<
  Required<ResultSchema>,
  "records" | "models" | "dataset" | "matching_summary" | "resources" | "config"
> & {
  records: DiagnosticRecord[];
  config: DiagnosticConfig;
  models: Record<string, any>;
  dataset: Record<string, any> | null;
  matching_summary: Record<string, any>;
  resources: Record<string, any>;
  availability?: Record<string, { available: boolean; reason?: string }>;
};
export interface DiagnosticJob {
  id: string;
  project: string;
  state: string;
  status: string;
  created_at: number;
  phase?: string;
  progress?: number;
  logs?: string;
  error?: string | null;
  result?: DiagnosticReport | null;
  matching_summary?: Record<string, any>;
}
export const diagnosticDefaults = (): DiagnosticConfig => ({
  schema_version: "1.0",
  fp32_model_id: "",
  quantized_model_id: "",
  dataset_id: "",
  max_samples: 3,
  max_tensors: 32,
  selected_nodes: [],
  operator_types: [],
  memory_budget_mb: 128,
  disk_budget_mb: 512,
  timeout: 180,
  provider: "CPUExecutionProvider",
  include_weights: true,
  temporary_policy: "delete_on_terminal",
});

export function rankDiagnosticRows(
  rows: DiagnosticRecord[],
  options: {
    metric: string;
    kind: string;
    status: string;
    operator: string;
    search: string;
    topN: number;
  },
) {
  const search = options.search.toLowerCase();
  return rows
    .filter(
      (r) =>
        r.kind === options.kind &&
        (!options.status || r.match_status === options.status) &&
        (!options.operator || r.operator_type === options.operator) &&
        (!search ||
          `${r.node_name} ${r.fp32_node_id} ${r.quantized_node_id} ${r.fp32_tensor} ${r.quantized_tensor}`
            .toLowerCase()
            .includes(search)),
    )
    .sort((a, b) => {
      const x = a.aggregate?.[options.metric],
        y = b.aggregate?.[options.metric];
      const validX = typeof x === "number" && Number.isFinite(x),
        validY = typeof y === "number" && Number.isFinite(y);
      if (!validX || !validY)
        return validX ? -1 : validY ? 1 : a.id.localeCompare(b.id);
      return (
        (options.metric === "cosine_similarity" ? x - y : y - x) ||
        a.id.localeCompare(b.id)
      );
    })
    .slice(0, Math.max(1, Math.min(256, options.topN || 20)));
}

export function errorThresholds(rows: DiagnosticRecord[]) {
  const values = rows
    .filter((r) => r.kind === "activation")
    .map((r) => r.aggregate?.mae)
    .filter((v) => typeof v === "number" && Number.isFinite(v))
    .sort((a, b) => a - b);
  const quantile = (p: number) => {
    if (!values.length) return null;
    const i = (values.length - 1) * p,
      lo = Math.floor(i),
      hi = Math.ceil(i);
    return values[lo] + (values[hi] - values[lo]) * (i - lo);
  };
  return { low: quantile(0.5), high: quantile(0.9) };
}
export function errorLevel(
  row: DiagnosticRecord,
  low: number | null,
  high: number | null,
) {
  const value = row.aggregate?.mae;
  if (
    typeof value !== "number" ||
    !Number.isFinite(value) ||
    !row.aggregate?.valid_samples ||
    low == null ||
    high == null ||
    !Number.isFinite(low) ||
    !Number.isFinite(high) ||
    low < 0 ||
    high < low
  )
    return "unavailable";
  return high != null && value > high
    ? "high"
    : low != null && value > low
      ? "medium"
      : "low";
}
