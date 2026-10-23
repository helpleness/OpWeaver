import type {
  Experiment as ExperimentSchema,
  ExperimentConfig as ConfigSchema,
} from "./generated/optimization-experiment";

type Complete<T> = [T] extends [never]
  ? never
  : T extends readonly (infer U)[]
    ? Complete<U>[]
    : T extends object
      ? { [K in keyof T]-?: Complete<T[K]> }
      : T;
export type Experiment = Complete<ExperimentSchema>;
export type Workflow = Omit<
  Complete<NonNullable<ConfigSchema["optimization"]>["workflow"]> & {},
  "passes"
> & {
  passes: {
    schema_version: "1.0";
    id: string;
    type: string;
    enabled: boolean;
    parameters: Record<string, any>;
  }[];
};
export type ExperimentConfig = Omit<Complete<ConfigSchema>, "optimization"> & {
  optimization: Omit<
    Complete<NonNullable<ConfigSchema["optimization"]>>,
    "workflow"
  > & { workflow: Workflow | null };
};
export interface PassCapability {
  type: string;
  available: boolean;
  reason?: string;
  model_constraint?: string;
  parameters: {
    name: string;
    type: string;
    default: boolean | number | string | string[];
    choices?: string[];
    description: string;
    available?: boolean;
    reason?: string;
    minimum?: number;
    maximum?: number;
  }[];
}
export interface BackendCapability {
  id: string;
  name: string;
  available: boolean;
  implemented: boolean;
  version: string | null;
  formats: string[];
  devices: string[];
  precisions: string[];
  levels: string[];
  dependencies: string[];
  reason: string;
  note: string;
  environment?: Record<string, any> | null;
  passes?: PassCapability[];
}
export interface LabJob {
  id: string;
  project: string;
  state: string;
  status: string;
  created_at: number;
  phase?: string;
  progress?: number | null;
  phases?: Record<string, string>;
  result?: Experiment | null;
  error?: string | null;
  logs?: string;
}
export function defaultConfig(): ExperimentConfig {
  return {
    schema_version: "1.0",
    mode: "optimize",
    source_model_id: "",
    inputs: {
      schema_version: "1.0",
      mode: "random",
      shapes: {},
      seed: 42,
      files: [],
    },
    benchmark: {
      schema_version: "1.0",
      warmup: 10,
      runs: 100,
      batch_size: null,
      batch_input: null,
      intra_op_threads: 1,
      inter_op_threads: 1,
      session_optimization: "disable",
      provider: "CPUExecutionProvider",
      execution_mode: "sequential",
      concurrency: 1,
    },
    optimization: {
      schema_version: "1.0",
      backend_id: "onnxruntime.cpu",
      level: "extended",
      device: "cpu",
      precision: "preserve",
      environment: null,
      workflow: null,
      calibration_dataset_id: null,
    },
    validation: { schema_version: "1.0", samples: 3, atol: 1e-5, rtol: 1e-4 },
    timeout: 300,
    evaluation: {
      schema_version: "1.0",
      dataset_id: null,
      allow_calibration_reuse: false,
      task: "none",
      output_name: null,
      final_dataset_id: null,
      gates: {
        schema_version: "1.0",
        max_mean_abs_error: null,
        max_abs_error: null,
        max_changed_fraction: null,
        max_accuracy_drop: null,
      },
    },
  };
}
export function defaultWorkflow(): Workflow {
  return {
    schema_version: "1.0",
    id: "local_workflow",
    passes: [],
    output: {
      schema_version: "1.0",
      model_name: "optimized.onnx",
      save_external_data: true,
      external_data_threshold: 1024,
    },
    log_level: 1,
  };
}
export function metric(value: unknown, digits = 4): string {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toLocaleString("en-US", { maximumFractionDigits: digits })
    : "—";
}
export function percent(value: unknown): string {
  return typeof value === "number" && Number.isFinite(value)
    ? `${value > 0 ? "+" : ""}${value.toFixed(2)}%`
    : "—";
}
