import type {
  GraphImportResult as ReportSchema,
  ModelArtifact as ArtifactSchema,
} from "./generated/onnx-report";
import type { AnalysisGraph, Diagnostic } from "./code-types";

export interface TensorInfo {
  name: string;
  shape: (number | string | null)[] | null;
  dtype: string;
  type?: string;
  implicit_capture?: boolean;
}
export interface WeightInfo extends TensorInfo {
  elements: number;
  logical_bytes: number | null;
  external: boolean;
  sparse: boolean;
  graph_path: string;
}
export type ModelArtifact = Omit<Required<ArtifactSchema>, "external_files"> & {
  external_files: { path: string; sha256: string; file_bytes: number }[];
};
export type OnnxReport = Omit<
  Required<ReportSchema>,
  | "analysis"
  | "artifact"
  | "model"
  | "inputs"
  | "outputs"
  | "initializers"
  | "statistics"
  | "validation"
  | "diagnostics"
> & {
  artifact: ModelArtifact;
  analysis: AnalysisGraph;
  model: {
    ir_version: number | string;
    opsets: Record<string, number>;
    producer: string;
    producer_version: string;
    model_version: number;
    domain: string;
    doc_string: string;
    parser_version: string;
    scope: string;
    metadata: { key: string; value: string }[];
    entry: string;
  };
  inputs: TensorInfo[];
  outputs: TensorInfo[];
  initializers: WeightInfo[];
  statistics: {
    nodes: number;
    graph_nodes: number;
    edges: number;
    operators: Record<string, number>;
    initializer_count: number;
    initializer_elements: number | null;
    learnable_parameters?: number;
    parameters_note: string;
    logical_tensor_bytes: number | null;
    unknown_operators: Record<string, number>;
    flops: null;
    weight_dtypes: Record<string, number>;
    subgraphs?: number;
  };
  validation: { checker: string; shape_inference: string; runtime: string };
  diagnostics: Diagnostic[];
  subgraphs: Record<string, AnalysisGraph>;
  pytorch: import("./code-types").CaptureResult | null;
};
export interface ModelJob {
  id: string;
  project: string;
  state: string;
  file_path: string;
  created_at: number;
  error: string | null;
  result?: OnnxReport | null;
  artifact?: ModelArtifact | null;
}
export const shapeText = (tensor: TensorInfo) =>
  tensor.shape === null || tensor.shape === undefined
    ? "Unknown rank"
    : `[${tensor.shape.map((d) => d ?? "?").join(", ")}]`;
export const bytesText = (value: number | null | undefined) => {
  if (value == null) return "Unknown";
  if (value < 1024) return `${value} B`;
  const index = Math.min(Math.floor(Math.log(value) / Math.log(1024)), 4);
  return `${(value / 1024 ** index).toFixed(2)} ${["B", "KiB", "MiB", "GiB", "TiB"][index]}`;
};
