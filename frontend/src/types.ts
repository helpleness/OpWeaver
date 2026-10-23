export type {
  Graph,
  Node as IRNode,
  Edge,
  TensorSpec,
} from "./generated/graph-ir";

export interface Operator {
  id: string;
  version: string;
  label: string;
  category: string;
  inputs: string[];
  outputs: string[];
  defaults: Record<string, unknown>;
  description: string;
  parameters: {
    properties: Record<
      string,
      {
        type?: string;
        default?: unknown;
        minimum?: number;
        maximum?: number;
        exclusiveMinimum?: number;
        exclusiveMaximum?: number;
        title?: string;
      }
    >;
  };
  capabilities: {
    dtypes: string[];
    devices: string[];
    autograd: boolean;
    torch_export: boolean;
    onnx: boolean;
    custom_kernel: boolean;
    backends: string[];
  };
}
export interface Layout {
  version: 1;
  positions: Record<string, { x: number; y: number }>;
}
export interface Validation {
  valid: boolean;
  issues: { severity: string; message: string; node?: string; edge?: string }[];
  order: string[];
  tensors: Record<
    string,
    { shape: (number | string)[]; dtype: string; device: string }
  >;
}
import type { TrainingConfig as GeneratedTrainingConfig } from "./generated/training-config";
export type TrainingConfig = Required<GeneratedTrainingConfig>;
export interface Metric {
  step: number;
  epoch: number;
  batch: number;
  loss: number;
  accuracy: number | null;
  learning_rate: number;
  elapsed_seconds: number;
}
export interface Job {
  id: string;
  state: string;
  config: TrainingConfig;
  metrics: Metric[];
  logs: { time: number; message: string }[];
  checkpoint: string | null;
  error: string | null;
  graph_digest: string;
  created_at: number;
}
