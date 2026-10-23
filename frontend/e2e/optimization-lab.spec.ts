import { test, expect, type Page } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";

async function rpc(page: Page, method: string, params: unknown = {}) {
  return page.evaluate(
    async ({ method, params }) => {
      const response = await fetch("/api/rpc", {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-OpWeaver": "1" },
        body: JSON.stringify({ method, params }),
      });
      const body = await response.json();
      if (body.error) throw new Error(body.error);
      return body.result;
    },
    { method, params },
  );
}
async function setup(page: Page, model: string) {
  await page.goto("/");
  await expect(
    page.getByText("Worker connected", { exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "New", exact: true }).click();
  const dialog = page.getByRole("dialog");
  const project = path.resolve(`../projects/lab-${Date.now()}`);
  await dialog
    .getByLabel("Project name", { exact: true })
    .fill("Optimization Lab acceptance");
  await dialog.getByLabel("Project directory", { exact: true }).fill(project);
  await dialog
    .getByRole("combobox", { name: "Starting point" })
    .selectOption("current");
  await dialog
    .getByRole("button", { name: "Create project", exact: true })
    .click();
  await expect(dialog).toBeHidden();
  const job = await rpc(page, "model.import.start", {
    project,
    source: path.resolve(`../examples/onnx/${model}.onnx`),
  });
  await expect
    .poll(
      async () =>
        (await rpc(page, "model.import.status", { id: job.id })).state,
      { timeout: 30000 },
    )
    .toBe("COMPLETED");
  await page.reload();
  await expect(
    page.getByText("Worker connected", { exact: true }),
  ).toBeVisible();
  await page
    .locator(".workspace-tabs")
    .getByRole("button", { name: "Optimization", exact: true })
    .click();
  await expect(page.getByLabel("Lab source model")).toHaveValue(job.id);
  return { project, model: job.id };
}
async function complete(page: Page) {
  await expect(page.locator(".lab-status")).toHaveText("Succeeded", {
    timeout: 30000,
  });
  await expect(page.locator(".lab-error")).toHaveCount(0);
}

test("local PNG/JPEG classification: four split roles, real FP32/INT8 quality, portable export and restore", async ({
  page,
}) => {
  test.setTimeout(180000);
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  const olive =
    process.env.OPWEAVER_OLIVE_PYTHON ?? path.resolve("../.tools/olive-env");
  test.skip(!fs.existsSync(olive), "Independent Olive environment required");
  const directory = path.resolve("../examples/classification");
  const fixture = JSON.parse(
    fs.readFileSync(path.join(directory, "fixture.json"), "utf8"),
  );
  const { project } = await setup(page, "../classification/color-classifier");
  await page.getByLabel("Timed iterations", { exact: true }).fill("15");
  await page.getByLabel("Warmup iterations", { exact: true }).fill("2");
  await page.getByLabel("Declared batch size", { exact: true }).fill("2");
  await page.getByLabel("Batch input", { exact: true }).selectOption("image");
  await page
    .getByLabel("Dataset source", { exact: true })
    .selectOption("images");
  await page
    .getByLabel("Image class mapping", { exact: true })
    .fill(JSON.stringify(fixture.classes));
  await page.getByLabel("Image resize", { exact: true }).fill("[10,10]");
  await page.getByLabel("Image center crop", { exact: true }).fill("[8,8]");
  await page
    .getByLabel("Image dataset provenance", { exact: true })
    .selectOption("synthetic_test");
  for (const role of [
    "calibration",
    "validation",
    "diagnostic",
    "final_evaluation",
  ]) {
    await page.getByLabel("Dataset role", { exact: true }).selectOption(role);
    await page
      .getByLabel("Image CSV manifest", { exact: true })
      .fill(path.join(directory, fixture.datasets[role]));
    await page
      .getByRole("button", { name: "Validate & save dataset", exact: true })
      .click();
    await expect(page.locator(".dataset-panel")).toContainText("Role " + role, {
      timeout: 30000,
    });
    await expect(page.locator(".lab-error")).toHaveCount(0);
  }
  const datasets = await rpc(page, "lab.dataset.list", { project });
  expect(datasets).toHaveLength(4);
  expect(
    datasets.every(
      (d: any) => d.example_count === 12 && d.observed_class_count === 6,
    ),
  ).toBe(true);
  const final = datasets.find((d: any) => d.role === "final_evaluation");
  await expect(
    page
      .getByLabel("Calibration dataset", { exact: true })
      .locator(`option[value="${final.id}"]`),
  ).toHaveAttribute("disabled", "");
  await expect(
    page
      .getByLabel("Evaluation dataset", { exact: true })
      .locator(`option[value="${final.id}"]`),
  ).toHaveAttribute("disabled", "");
  await page
    .getByText("Task evaluator & optional quality gates", { exact: true })
    .click();
  await page
    .getByLabel("Task evaluator", { exact: true })
    .selectOption("classification_top1");
  await page
    .getByLabel("Classification output name", { exact: true })
    .fill("scores");
  // No random-input dynamic shape is entered: actual image batches supply it.
  await page
    .getByRole("button", { name: "Run baseline only", exact: true })
    .click();
  await complete(page);
  const quality = page.getByLabel("Classification task quality", {
    exact: true,
  });
  await expect(quality).toContainText("12 labeled examples");
  await expect(quality).toContainText("12 / 0");
  await expect(quality).toContainText("100%");
  let history = await rpc(page, "lab.history", { project });
  const baseline = await rpc(page, "lab.load", { project, id: history[0].id });
  expect(baseline.baseline_evaluation.correct).toBe(12);
  expect(baseline.final_evaluation.top5_accuracy).toBe(1);
  await expect(page.locator(".lab-result-title")).toContainText(
    "onnxruntime.cpu",
  );
  await page
    .getByLabel("Optimization backend", { exact: true })
    .selectOption("microsoft.olive");
  await page
    .getByLabel("Olive Python environment", { exact: true })
    .fill(olive);
  await page
    .getByRole("button", { name: "Save & detect environment", exact: true })
    .click();
  await expect(page.locator(".olive-environment")).toContainText("Available", {
    timeout: 45000,
  });
  await page
    .getByLabel("Olive pass type", { exact: true })
    .selectOption("OnnxStaticQuantization");
  await page.getByRole("button", { name: "Add pass", exact: true }).click();
  await page.getByLabel("1 per_channel", { exact: true }).check();
  await page
    .getByRole("button", { name: "Validate & preview", exact: true })
    .click();
  await expect(page.locator(".olive-message")).toContainText("validated", {
    timeout: 30000,
  });
  await page
    .getByRole("button", { name: "Run optimization experiment", exact: true })
    .click();
  await complete(page);
  await expect(quality).toContainText("percentage points");
  await expect(quality).toContainText("relative change");
  history = await rpc(page, "lab.history", { project });
  const report = await rpc(page, "lab.load", { project, id: history[0].id });
  expect(report.validation.task_quality.optimized_metrics.correct).toBe(12);
  expect(report.validation.task_quality.optimized_top5_accuracy).toBe(1);
  expect(report.final_evaluation.samples).toBe(12);
  expect(report.datasets.independent).toBe(true);
  expect(report.optimized.conditions_hash).toBe(
    report.baseline.conditions_hash,
  );
  const destination = path.resolve("../test-results/classification-export");
  fs.mkdirSync(destination, { recursive: true });
  await page
    .getByLabel("Experiment export directory", { exact: true })
    .fill(destination);
  await page
    .getByRole("button", { name: "Export model & reports", exact: true })
    .click();
  await expect(page.locator(".lab-exported")).toContainText(report.id);
  const exported = path.join(destination, `opweaver-${report.id}`);
  expect(
    fs.readFileSync(
      path.join(exported, "datasets/evaluation/source/replay.csv"),
      "utf8",
    ),
  ).toContain("image-0000.png,red");
  expect(
    fs.existsSync(
      path.join(exported, "datasets/final_evaluation/manifest.json"),
    ),
  ).toBe(true);
  fs.writeFileSync(
    path.resolve("../test-results/p1c1-e2e-measurements.json"),
    JSON.stringify({ project, baseline, report }, null, 2),
  );
  await page.reload();
  await expect(
    page.getByText("Worker connected", { exact: true }),
  ).toBeVisible();
  await page
    .locator(".workspace-tabs")
    .getByRole("button", { name: "Optimization", exact: true })
    .click();
  await expect(
    page.getByLabel("Experiment history", { exact: true }),
  ).toHaveValue(report.id);
  await expect(quality).toContainText("12 / 0");
  await expect(
    page.getByLabel("Final evaluation dataset", { exact: true }),
  ).toHaveValue(final.id);
  await page.screenshot({ path: "../test-results/p1c1-classification.png" });
  expect(errors).toEqual([]);
});

test("real static INT8 with mapped NPZ calibration, independent evaluation, quality gates and export", async ({
  page,
}) => {
  test.setTimeout(180000);
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  const olive =
    process.env.OPWEAVER_OLIVE_PYTHON ?? path.resolve("../.tools/olive-env");
  test.skip(!fs.existsSync(olive), "Independent Olive environment required");
  const fixture = JSON.parse(
    fs.readFileSync(
      path.resolve("../examples/quantization/fixture.json"),
      "utf8",
    ),
  );
  const { project } = await setup(page, "../quantization/quantization-cnn");
  await page.getByLabel("Shape for image", { exact: true }).fill("[2,3,8,8]");
  await page.getByLabel("Timed iterations", { exact: true }).fill("100");
  await page
    .getByLabel("Optimization backend", { exact: true })
    .selectOption("microsoft.olive");
  await page
    .getByLabel("Olive Python environment", { exact: true })
    .fill(olive);
  await page
    .getByRole("button", { name: "Save & detect environment", exact: true })
    .click();
  await expect(page.locator(".olive-environment")).toContainText("Available", {
    timeout: 45000,
  });
  await page
    .getByLabel("Olive workflow ID", { exact: true })
    .fill("static_npz_acceptance");
  await page
    .getByLabel("Olive pass type", { exact: true })
    .selectOption("OnnxStaticQuantization");
  await page.getByRole("button", { name: "Add pass", exact: true }).click();
  await page.getByLabel("1 per_channel", { exact: true }).check();
  for (const role of ["calibration", "evaluation"]) {
    await page.getByLabel("Dataset role", { exact: true }).selectOption(role);
    await page
      .getByLabel("Dataset NPZ paths", { exact: true })
      .fill(
        fixture.datasets[role].files
          .map((file: string) => path.resolve("../examples/quantization", file))
          .join("\n"),
      );
    await page
      .getByLabel("Dataset input mapping", { exact: true })
      .fill('{"image":"pixels"}');
    await page
      .getByRole("button", { name: "Validate & save dataset", exact: true })
      .click();
    await expect(page.locator(".dataset-panel")).toContainText(
      `Dataset preview · ${fixture.datasets[role].count} batches`,
      { timeout: 30000 },
    );
  }
  await page
    .getByText("Task evaluator & optional quality gates", { exact: true })
    .click();
  await page
    .getByLabel("Quality gate max_mean_abs_error", { exact: true })
    .fill("0");
  await page
    .getByRole("button", { name: "Validate & preview", exact: true })
    .click();
  await expect(page.locator(".olive-message")).toContainText("validated", {
    timeout: 30000,
  });
  await page
    .getByRole("button", { name: "Save workflow", exact: true })
    .click();
  await expect(page.locator(".olive-message")).toContainText("saved", {
    timeout: 30000,
  });
  await page
    .getByRole("button", { name: "Run optimization experiment", exact: true })
    .click();
  await complete(page);
  const summary = page.getByLabel("Quantization quality summary", {
    exact: true,
  });
  await expect(summary).toContainText("QDQ");
  await expect(summary).toContainText("Execution valid");
  await expect(summary).toContainText("Quality gate failed");
  await expect(summary).toContainText("Independent evaluation");
  const history = await rpc(page, "lab.history", { project });
  const id = history[0].id;
  const report = await rpc(page, "lab.load", { project, id });
  expect(report.validation.execution_valid).toBe(true);
  expect(report.validation.numerical_summary.mean_abs_error).toBeGreaterThan(0);
  expect(report.workflow_execution.calibration_usage[0].unique_batches).toBe(
    32,
  );
  expect(report.optimized.conditions_hash).toBe(
    report.baseline.conditions_hash,
  );
  const destination = path.resolve("../test-results/quantization-export");
  fs.mkdirSync(destination, { recursive: true });
  await page
    .getByLabel("Experiment export directory", { exact: true })
    .fill(destination);
  await page
    .getByRole("button", { name: "Export model & reports", exact: true })
    .click();
  await expect(page.locator(".lab-exported")).toContainText(id);
  expect(
    fs.existsSync(
      path.join(
        destination,
        `opweaver-${id}`,
        "datasets",
        "calibration",
        "manifest.json",
      ),
    ),
  ).toBe(true);
  await page.reload();
  await page
    .locator(".workspace-tabs")
    .getByRole("button", { name: "Optimization", exact: true })
    .click();
  await expect(page.getByLabel("Quantization quality summary")).toContainText(
    "Quality gate failed",
  );
  expect(
    await page.getByLabel("Evaluation dataset", { exact: true }).inputValue(),
  ).toBe(report.config.evaluation.dataset_id);
  await page
    .getByRole("button", { name: "Graph changes", exact: true })
    .click();
  await page
    .getByRole("button", { name: "View optimized graph", exact: true })
    .click();
  await expect(
    page.getByLabel("Model Inspector", { exact: true }),
  ).toContainText("Quantization · QDQ");
  expect(errors).toEqual([]);
});

test("real independent Olive ordered workflow, saved config, validation, export and restore", async ({
  page,
}) => {
  test.setTimeout(180000);
  const olive =
    process.env.OPWEAVER_OLIVE_PYTHON ?? path.resolve("../.tools/olive-env");
  test.skip(
    !fs.existsSync(olive),
    "Run scripts/setup_olive.ps1 for independent Olive acceptance",
  );
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  const { project } = await setup(page, "redundant-identity");
  await page
    .getByLabel("Optimization backend", { exact: true })
    .selectOption("microsoft.olive");
  await expect(page.locator(".olive-environment")).toContainText(
    "Not Installed",
  );
  await expect(
    page.getByRole("button", {
      name: "Run optimization experiment",
      exact: true,
    }),
  ).toBeDisabled();
  await page
    .getByLabel("Optimization backend", { exact: true })
    .selectOption("onnxruntime.cpu");
  await expect(
    page.getByRole("button", { name: "Run baseline only", exact: true }),
  ).toBeEnabled();
  await page
    .getByLabel("Optimization backend", { exact: true })
    .selectOption("microsoft.olive");
  await page.getByLabel("Olive Python environment").fill(olive);
  await page
    .getByRole("button", { name: "Save & detect environment", exact: true })
    .click();
  await expect(page.locator(".olive-environment")).toContainText("Available", {
    timeout: 45000,
  });
  await page.getByLabel("Olive workflow ID").fill("acceptance_workflow");
  await page.getByRole("button", { name: "Add pass", exact: true }).click();
  await page
    .getByLabel("Olive pass type")
    .selectOption("OnnxOpVersionConversion");
  await page.getByRole("button", { name: "Add pass", exact: true }).click();
  await page.getByLabel("Move pass 2 up", { exact: true }).click();
  await expect(page.locator(".olive-pass").first()).toContainText(
    "OnnxOpVersionConversion",
  );
  await page.getByLabel("Move pass 1 down", { exact: true }).click();
  await page.getByRole("button", { name: "Add pass", exact: true }).click();
  await page.getByLabel("Delete pass 3", { exact: true }).click();
  await expect(page.locator(".olive-pass")).toHaveCount(2);
  await page.getByLabel("Enable pass 2", { exact: true }).uncheck();
  await page
    .getByRole("button", { name: "Validate & preview", exact: true })
    .click();
  await expect(page.locator(".olive-message")).toContainText("validated", {
    timeout: 30000,
  });
  await page.getByLabel("Enable pass 2", { exact: true }).check();
  await page
    .getByText("Advanced generic workflow JSON", { exact: true })
    .click();
  const generic = page.getByLabel("Advanced Olive workflow", { exact: true });
  const workflow = JSON.parse(await generic.inputValue());
  await generic.fill(JSON.stringify({ ...workflow, shell: "unsupported" }));
  await page
    .getByRole("button", { name: "Validate & apply JSON", exact: true })
    .click();
  await expect(page.locator(".lab-error")).toContainText("Extra inputs", {
    timeout: 30000,
  });
  await generic.fill(JSON.stringify(workflow));
  await page
    .getByRole("button", { name: "Validate & apply JSON", exact: true })
    .click();
  await expect(page.locator(".olive-message")).toContainText("validated", {
    timeout: 30000,
  });
  await expect(page.locator(".lab-error")).toHaveCount(0);
  await page.getByLabel("Olive output model name").fill("olive-optimized.onnx");
  await page
    .getByRole("button", { name: "Save workflow", exact: true })
    .click();
  await expect(page.locator(".olive-message")).toContainText("saved", {
    timeout: 30000,
  });
  await page
    .getByLabel("Saved Olive workflows", { exact: true })
    .selectOption("acceptance_workflow");
  await expect(page.locator(".olive-pass")).toHaveCount(2);
  await expect(page.getByLabel("Olive output model name")).toHaveValue(
    "olive-optimized.onnx",
  );
  await expect(page.locator(".lab-error")).toHaveCount(0);
  await page.getByLabel("Timed iterations").fill("40");
  await page.getByLabel("Warmup iterations").fill("3");
  await page
    .getByRole("button", { name: "Run optimization experiment", exact: true })
    .click();
  await expect(page.locator(".lab-status")).toHaveText("Succeeded", {
    timeout: 60000,
  });
  await expect(page.locator(".lab-error")).toHaveCount(0);
  const records = await rpc(page, "lab.history", { project });
  const saved = await rpc(page, "lab.load", { project, id: records[0].id });
  expect(
    saved.workflow_execution.executed_passes.map((p: any) => p.pass_type),
  ).toEqual(["onnxpeepholeoptimizer", "onnxopversionconversion"]);
  expect(saved.workflow_execution.environment.versions["olive-ai"]).toBe(
    "0.13.0",
  );
  expect(saved.graph_comparison.original_nodes).toBe(3);
  expect(saved.graph_comparison.optimized_nodes).toBe(1);
  expect(saved.validation.passed).toBe(true);
  expect(saved.baseline.conditions_hash).toBe(saved.optimized.conditions_hash);
  expect(saved.baseline.latencies_ms).toHaveLength(40);
  const samplePaths = Array.from({ length: 3 }, (_, i) =>
    path.join(
      project,
      ".opweaver",
      "optimizations",
      saved.id,
      "inputs",
      `sample-${i}.npz`,
    ),
  );
  await page.getByLabel("Lab input mode", { exact: true }).selectOption("npz");
  await page
    .getByLabel("NPZ sample paths", { exact: true })
    .fill(samplePaths.join("\n"));
  await page
    .getByRole("button", { name: "Save workflow", exact: true })
    .click();
  await expect(page.locator(".olive-message")).toContainText("saved", {
    timeout: 30000,
  });
  const templates = await rpc(page, "lab.workflow.list", { project });
  expect(templates[0].config.inputs.files).toEqual(samplePaths);
  await page
    .getByLabel("NPZ sample paths", { exact: true })
    .fill("unsaved draft");
  await page
    .getByLabel("Saved Olive workflows", { exact: true })
    .selectOption("acceptance_workflow");
  await expect(
    page.getByLabel("NPZ sample paths", { exact: true }),
  ).toHaveValue(samplePaths.join("\n"));
  const parent = path.resolve("../test-results/olive-export");
  fs.mkdirSync(parent, { recursive: true });
  await page.getByLabel("Experiment export directory").fill(parent);
  await page
    .getByRole("button", { name: "Export model & reports", exact: true })
    .click();
  await expect(page.locator(".lab-exported")).toContainText(saved.id);
  expect(
    fs.readFileSync(
      path.join(parent, `opweaver-${saved.id}`, "workflow", "stdout.log"),
      "utf8",
    ),
  ).toContain("Running pass");
  await page
    .getByRole("button", { name: "Graph changes", exact: true })
    .click();
  await page
    .getByRole("button", { name: "View optimized graph", exact: false })
    .click();
  await expect(page.getByLabel("Imported model")).toHaveValue(
    saved.optimization.artifact.id,
  );
  await page.reload();
  await expect(
    page.getByText("Worker connected", { exact: true }),
  ).toBeVisible();
  await page
    .locator(".workspace-tabs")
    .getByRole("button", { name: "Optimization", exact: true })
    .click();
  await expect(
    page.getByLabel("Experiment history", { exact: true }),
  ).toHaveValue(saved.id);
  await page
    .getByRole("button", { name: "Configuration & logs", exact: true })
    .click();
  await expect(page.locator(".lab-results")).toContainText(
    "Olive workflow · requested, executed and observed",
  );
  await expect(
    page.getByLabel("Optimization backend", { exact: true }),
  ).toHaveValue("microsoft.olive");
  await page.screenshot({ path: "../test-results/olive-workflow.png" });
  expect(errors).toEqual([]);
});
test("real baseline, ORT optimization, validation, graphs, export and reopen", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  const source = path.resolve("../examples/onnx/external-cnn.onnx");
  const before = crypto
    .createHash("sha256")
    .update(fs.readFileSync(source))
    .digest("hex");
  const { project } = await setup(page, "external-cnn");
  await page.getByLabel("Shape for image").fill("[2,3,8,8]");
  await page.getByLabel("Timed iterations").fill("20");
  await page.getByLabel("Warmup iterations").fill("3");
  await page.getByLabel("Declared batch size").fill("2");
  await page.getByLabel("Batch input", { exact: true }).selectOption("image");
  await page
    .getByRole("button", { name: "Run baseline only", exact: true })
    .click();
  await complete(page);
  await expect(page.locator(".lab-table")).toContainText("requests/s");
  await page.getByLabel("Artifact optimization level").selectOption("extended");
  await page
    .getByRole("button", { name: "Run optimization experiment", exact: true })
    .click();
  await complete(page);
  await expect(page.locator(".lab-table")).toContainText(
    "Passed on recorded samples",
  );
  const records = await rpc(page, "lab.history", { project });
  const saved = await rpc(page, "lab.load", { project, id: records[0].id });
  expect(saved.baseline.latencies_ms.length).toBe(20);
  expect(saved.optimized.latencies_ms.length).toBe(20);
  expect(saved.baseline.conditions_hash).toBe(saved.optimized.conditions_hash);
  expect(saved.validation.samples).toBe(3);
  expect(saved.optimization.artifact.parent_model_id).toBe(
    saved.source_model_id,
  );
  await page.getByRole("button", { name: "Validation", exact: true }).click();
  await expect(page.locator(".lab-validation-heading")).toContainText("passed");
  await page
    .getByRole("button", { name: "Graph changes", exact: true })
    .click();
  await expect(page.locator(".lab-table")).toContainText(
    "com.microsoft::FusedConv",
  );
  await page
    .getByRole("button", { name: "View optimized graph", exact: false })
    .click();
  await expect(page.locator(".model-browser .vue-flow")).toBeVisible();
  await expect(page.getByLabel("Imported model")).toHaveValue(
    saved.optimization.artifact.id,
  );
  await page
    .locator(".workspace-tabs")
    .getByRole("button", { name: "Optimization", exact: true })
    .click();
  const exportParent = path.resolve("../test-results/lab-export");
  fs.mkdirSync(exportParent, { recursive: true });
  await page.getByLabel("Experiment export directory").fill(exportParent);
  await page
    .getByRole("button", { name: "Export model & reports", exact: true })
    .click();
  await expect(page.locator(".lab-exported")).toContainText(saved.id);
  const bundle = path.join(exportParent, `opweaver-${saved.id}`);
  const exported = JSON.parse(
    fs.readFileSync(path.join(bundle, "report.json"), "utf8"),
  );
  expect(
    fs.existsSync(path.join(bundle, exported.exported_files.optimized)),
  ).toBe(true);
  expect(exported.baseline.latencies_ms).toEqual(saved.baseline.latencies_ms);
  await page.reload();
  await expect(
    page.getByText("Worker connected", { exact: true }),
  ).toBeVisible();
  await page
    .locator(".workspace-tabs")
    .getByRole("button", { name: "Optimization", exact: true })
    .click();
  await expect(
    page.getByLabel("Experiment history", { exact: true }),
  ).toHaveValue(saved.id);
  await expect(page.getByLabel("Timed iterations")).toHaveValue("20");
  await complete(page);
  await page
    .getByLabel("Compare with experiment", { exact: true })
    .selectOption(records[1].id);
  await expect(page.locator(".lab-cross-comparison")).toContainText("Mean ms");
  await page.screenshot({ path: "../test-results/optimization-lab.png" });
  expect(
    crypto.createHash("sha256").update(fs.readFileSync(source)).digest("hex"),
  ).toBe(before);
  expect(errors).toEqual([]);
});
test("dynamic input errors and a real multi input output experiment", async ({
  page,
}) => {
  const { project } = await setup(page, "multi-input-output");
  await page
    .getByRole("button", { name: "Run optimization experiment", exact: true })
    .click();
  await expect(page.locator(".lab-error")).toContainText("Resolve all dynamic");
  await page.getByLabel("Shape for x", { exact: true }).fill("[2,4]");
  await page.getByLabel("Shape for y", { exact: true }).fill("[2,4]");
  await page.getByLabel("Timed iterations").fill("10");
  await page
    .getByRole("button", { name: "Run optimization experiment", exact: true })
    .click();
  await complete(page);
  await page.getByRole("button", { name: "Validation", exact: true }).click();
  await expect(page.locator(".lab-table tbody tr")).toHaveCount(9);
  await expect(page.locator(".lab-table")).toContainText("(exact)");
  const saved = (await rpc(page, "lab.history", { project }))[0];
  await page.getByLabel("Lab input mode").selectOption("npz");
  await page
    .getByLabel("NPZ sample paths")
    .fill(
      [0, 1, 2]
        .map((i) =>
          path.join(
            project,
            ".opweaver",
            "optimizations",
            saved.id,
            "inputs",
            `sample-${i}.npz`,
          ),
        )
        .join("\n"),
    );
  await page
    .getByRole("button", { name: "Run optimization experiment", exact: true })
    .click();
  await complete(page);
  const replay = await rpc(page, "lab.load", {
    project,
    id: (await rpc(page, "lab.history", { project }))[0].id,
  });
  expect(replay.config.inputs.mode).toBe("npz");
  expect(replay.validation.passed).toBe(true);
});
