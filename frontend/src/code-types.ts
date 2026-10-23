import type { Graph } from "./types";
import type {
  AnalysisGraph as SchemaAnalysisGraph,
  AnalysisNode as SchemaAnalysisNode,
  Edge as SchemaEdge,
} from "./generated/analysis-graph";
import type { SourceMapping as SchemaSourceMapping } from "./generated/source-mapping";

export interface SourceFile {
  path: string;
  text: string;
  hash: string;
}
export interface Diagnostic {
  severity: string;
  message: string;
  line?: number;
  column?: number;
  node?: string;
  hint?: string;
}
export type SourceMapping = Required<SchemaSourceMapping>;
export type AnalysisNode = Omit<Required<SchemaAnalysisNode>, "tensors"> & {
  tensors: { shape?: (string | number)[]; dtype?: string; device?: string };
};
export type AnalysisGraph = Omit<
  Required<SchemaAnalysisGraph>,
  "nodes" | "edges" | "diagnostics"
> & {
  nodes: AnalysisNode[];
  edges: Required<SchemaEdge>[];
  diagnostics: Diagnostic[];
};
export interface CaptureResult {
  success: boolean;
  analysis: AnalysisGraph;
  design_graph: Graph | null;
  source_hash: string;
  graph_hash: string;
  source_map: SourceMapping[];
  diagnostics: Diagnostic[];
  forward: {
    outputs: {
      shape: number[];
      dtype: string;
      mean: number;
      min: number;
      max: number;
    }[];
    parameters: number;
    mode: string;
    device: string;
  };
  statistics: { parameters: number; file_bytes: number; nodes: number };
  capture: { specialized: string; backend: string };
}
export interface CaptureJob {
  id: string;
  state: string;
  result: CaptureResult | null;
  error: string | null;
  logs: string;
  source_hash: string;
  file_path: string;
  artifact_dir: string;
  project: string;
}
export type SyncState =
  | "synced"
  | "pending"
  | "parsing"
  | "parse_failed"
  | "stale"
  | "conflict"
  | "read_only";
export interface CodeProposal {
  path: string;
  before: string;
  text: string;
  expected_hash: string;
  diff: string;
  kind: "generate" | "patch";
  source_map?: SourceMapping[];
  class_name?: string;
  graph_snapshot?: string;
}
