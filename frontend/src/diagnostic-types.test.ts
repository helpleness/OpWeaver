import { expect, it } from "vitest";
import {
  errorLevel,
  errorThresholds,
  rankDiagnosticRows,
  type DiagnosticRecord,
} from "./diagnostic-types";

const row = (
  id: string,
  mae: number | null,
  cosine: number | null,
  operator = "Conv",
  status: DiagnosticRecord["match_status"] = "Matched",
): DiagnosticRecord => ({
  schema_version: "1.0",
  id,
  kind: "activation",
  fp32_tensor: `tensor_${id}`,
  fp32_node_id: `node_${id}`,
  quantized_tensor: null,
  quantized_node_id: null,
  quantized_tensor_producer_id: null,
  node_name: id,
  node_name_unique: true,
  fp32_shape: [2],
  quantized_shape: [2],
  match_method: "unit ranking fixture",
  message: "Known statistics for deterministic ranking tests",
  samples: [],
  quantization: null,
  operator_type: operator,
  match_status: status,
  aggregate: {
    mae,
    rmse: mae,
    max_abs_error: mae == null ? null : mae * 2,
    cosine_similarity: cosine,
    valid_samples: mae == null ? 0 : 2,
  },
});
const rows = [
  row("a", 0.1, 0.8),
  row("b", 0.9, 0.4, "Gemm"),
  row("c", 0.3, null, "Gemm"),
  row("d", null, null, "Relu", "Unmatched"),
];
const options = {
  kind: "activation",
  metric: "mae",
  status: "",
  operator: "",
  search: "",
  topN: 2,
};
it("sorts known errors, lowest cosine, and keeps undefined metrics at the end", () => {
  expect(rankDiagnosticRows(rows, options).map((r) => r.id)).toEqual([
    "b",
    "c",
  ]);
  expect(
    rankDiagnosticRows(rows, {
      ...options,
      metric: "cosine_similarity",
      topN: 4,
    }).map((r) => r.id),
  ).toEqual(["b", "a", "c", "d"]);
  expect(
    rankDiagnosticRows(rows, { ...options, metric: "max_abs_error" }).map(
      (r) => r.id,
    ),
  ).toEqual(["b", "c"]);
  expect(
    rankDiagnosticRows(rows, { ...options, metric: "rmse" }).map((r) => r.id),
  ).toEqual(["b", "c"]);
});
it("filters by type, match status and tensor/node searches independently", () => {
  expect(
    rankDiagnosticRows(rows, {
      ...options,
      operator: "Gemm",
      search: "tensor_c",
    }).map((r) => r.id),
  ).toEqual(["c"]);
  expect(
    rankDiagnosticRows(rows, { ...options, status: "Unmatched", topN: 4 }).map(
      (r) => r.id,
    ),
  ).toEqual(["d"]);
  expect(
    rankDiagnosticRows(rows, { ...options, search: "node_a" }).map((r) => r.id),
  ).toEqual(["a"]);
});
it("uses explicit report-relative thresholds and marks unavailable metrics separately", () => {
  const thresholds = errorThresholds(rows);
  expect(thresholds.low).toBe(0.3);
  expect(thresholds.high).toBeCloseTo(0.78);
  expect(errorLevel(rows[1], thresholds.low, thresholds.high)).toBe("high");
  expect(errorLevel(rows[3], thresholds.low, thresholds.high)).toBe(
    "unavailable",
  );
  expect(errorLevel(rows[1], 1, 2)).toBe("low");
  expect(errorLevel(rows[1], 2, 1)).toBe("unavailable");
  expect(errorLevel(rows[1], Number.NaN, 2)).toBe("unavailable");
});
