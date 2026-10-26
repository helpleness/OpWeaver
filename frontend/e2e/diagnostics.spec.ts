import { test, expect, type Page } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";

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

test("real QDQ diagnostic: capture, ranking, both graph locations, weights, export and reload", async ({
  page,
}) => {
  test.setTimeout(180000);
  const olive =
    process.env.OPWEAVER_OLIVE_PYTHON ?? path.resolve("../.tools/olive-env");
  test.skip(!fs.existsSync(olive), "Independent Olive environment required");
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await expect(
    page.getByText("Worker connected", { exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "New", exact: true }).click();
  const dialog = page.getByRole("dialog");
  const project = path.resolve(`../projects/diagnostic-${Date.now()}`);
  await dialog
    .getByLabel("Project name", { exact: true })
    .fill("Quantization diagnostic acceptance");
  await dialog.getByLabel("Project directory", { exact: true }).fill(project);
  await dialog
    .getByRole("combobox", { name: "Starting point", exact: true })
    .selectOption("current");
  await dialog
    .getByRole("button", { name: "Create project", exact: true })
    .click();
  await expect(dialog).toBeHidden();
  const root = path.resolve("../examples/classification");
  const fixture = JSON.parse(
    fs.readFileSync(path.join(root, "fixture.json"), "utf8"),
  );
  const imported = await rpc(page, "model.import.start", {
    project,
    source: path.join(root, "color-classifier.onnx"),
  });
  await expect
    .poll(
      async () =>
        (await rpc(page, "model.import.status", { id: imported.id })).state,
      { timeout: 30000 },
    )
    .toBe("COMPLETED");
  const datasets: Record<string, any> = {};
  for (const role of [
    "calibration",
    "validation",
    "diagnostic",
    "final_evaluation",
  ]) {
    datasets[role] = await rpc(page, "lab.dataset.create", {
      project,
      config: {
        source_model_id: imported.id,
        role,
        mode: "images",
        name: `Images ${role}`,
        image: {
          manifest_path: path.join(root, fixture.datasets[role]),
          input_name: "image",
          class_mapping: fixture.classes,
          batch_size: 2,
          preprocessing: fixture.preprocessing,
          provenance: "synthetic_test",
        },
      },
    });
  }
  const quant = await rpc(page, "lab.start", {
    project,
    config: {
      source_model_id: imported.id,
      benchmark: { runs: 12, warmup: 2 },
      optimization: {
        backend_id: "microsoft.olive",
        environment: { python_path: olive },
        calibration_dataset_id: datasets.calibration.id,
        workflow: {
          passes: [
            {
              id: "quant",
              type: "OnnxStaticQuantization",
              parameters: { per_channel: true },
            },
          ],
        },
      },
      evaluation: { dataset_id: datasets.validation.id },
    },
  });
  await expect
    .poll(async () => (await rpc(page, "lab.status", { id: quant.id })).state, {
      timeout: 60000,
    })
    .toBe("COMPLETED");
  const experiment = await rpc(page, "lab.load", { project, id: quant.id });
  const quantized = experiment.optimization.artifact.id;
  await page.reload();
  await expect(
    page.getByText("Worker connected", { exact: true }),
  ).toBeVisible();
  await page
    .locator(".workspace-tabs")
    .getByRole("button", { name: "Optimization", exact: true })
    .click();
  await page
    .getByRole("button", { name: "Quantization diagnostics", exact: true })
    .click();
  await page
    .getByLabel("Diagnostic FP32 model", { exact: true })
    .selectOption(imported.id);
  await page
    .getByLabel("Diagnostic quantized model", { exact: true })
    .selectOption(quantized);
  await page
    .getByLabel("Diagnostic dataset", { exact: true })
    .selectOption(datasets.diagnostic.id);
  await expect(
    page
      .getByLabel("Diagnostic dataset", { exact: true })
      .locator(`option[value="${datasets.final_evaluation.id}"]`),
  ).toHaveCount(0);
  await page.getByLabel("Diagnostic max samples", { exact: true }).fill("2");
  await page.getByLabel("Diagnostic max tensors", { exact: true }).fill("32");
  await page
    .getByRole("button", { name: "Run layer diagnostic", exact: true })
    .click();
  await expect(page.locator(".diagnostic-status")).toHaveText("Succeeded", {
    timeout: 30000,
  });
  await expect(page.locator(".diagnostic-error")).toHaveCount(0);
  const history = await rpc(page, "diagnostic.history", { project });
  const report = await rpc(page, "diagnostic.load", {
    project,
    id: history[0].id,
  });
  expect(report.captures).toHaveLength(2);
  expect(report.matching_summary.coverage).toBe(1);
  expect(report.resources.temporary_cleaned).toBe(true);
  await expect(
    page.getByLabel("Tensor matching coverage", { exact: true }),
  ).toContainText("5 / 5");
  await expect(
    page.getByLabel("Layer error ranking").locator("tbody tr"),
  ).toHaveCount(5);
  const measured = report.records
    .filter((r: any) => r.kind === "activation")
    .sort(
      (a: any, b: any) =>
        b.aggregate.mae - a.aggregate.mae || a.id.localeCompare(b.id),
    );
  const target = measured[0];
  await expect(
    page.getByLabel("Layer error ranking").locator("tbody tr").first(),
  ).toHaveAttribute("data-record-id", target.id);
  await page.getByLabel("Diagnostic top N", { exact: true }).fill("2");
  await expect(
    page.getByLabel("Layer error ranking").locator("tbody tr"),
  ).toHaveCount(2);
  await page
    .getByLabel("Search diagnostic tensors", { exact: true })
    .fill(target.fp32_tensor);
  await expect(
    page.getByLabel("Layer error ranking").locator("tbody tr"),
  ).toHaveCount(1);
  await page
    .getByRole("button", { name: `Locate FP32 ${target.id}`, exact: true })
    .click();
  await expect(page.getByLabel("Imported model", { exact: true })).toHaveValue(
    imported.id,
  );
  await expect(
    page.locator(`.vue-flow__node[data-id="${target.fp32_node_id}"]`),
  ).toHaveClass(/selected/);
  await expect
    .poll(async () => {
      const box = await page
        .locator(`.vue-flow__node[data-id="${target.fp32_node_id}"]`)
        .boundingBox();
      const canvas = await page
        .locator(".import-graph-area .vue-flow")
        .boundingBox();
      return (
        !!box &&
        !!canvas &&
        Math.abs(box.y + box.height / 2 - canvas.y - canvas.height / 2) <
          canvas.height * 0.2
      );
    })
    .toBe(true);
  await expect(
    page
      .getByLabel("Model Inspector", { exact: true })
      .getByLabel("Diagnostic Tensor details", { exact: true }),
  ).toContainText(target.fp32_tensor);
  await expect(
    page.locator(".operator-node[data-diagnostic-level]").first(),
  ).toBeVisible();
  await page
    .locator(".workspace-tabs")
    .getByRole("button", { name: "Optimization", exact: true })
    .click();
  await expect(
    page.getByLabel("Diagnostic history", { exact: true }),
  ).toHaveValue(report.id);
  await page
    .getByRole("button", { name: `Locate INT8 ${target.id}`, exact: true })
    .click();
  await expect(page.getByLabel("Imported model", { exact: true })).toHaveValue(
    quantized,
  );
  await expect(
    page.locator(`.vue-flow__node[data-id="${target.quantized_node_id}"]`),
  ).toHaveClass(/selected/);
  await expect(
    page
      .getByLabel("Model Inspector", { exact: true })
      .getByLabel("Diagnostic Tensor details", { exact: true }),
  ).toContainText(target.quantized_tensor);
  await page
    .locator(".workspace-tabs")
    .getByRole("button", { name: "Optimization", exact: true })
    .click();
  await page
    .getByLabel("Diagnostic error kind", { exact: true })
    .selectOption("weight");
  await expect(
    page.getByLabel("Layer error ranking").locator("tbody tr"),
  ).toHaveCount(3);
  const destination = path.resolve("../test-results/diagnostic-export");
  fs.mkdirSync(destination, { recursive: true });
  await page
    .getByLabel("Diagnostic export directory", { exact: true })
    .fill(destination);
  await page
    .getByRole("button", { name: "Export diagnostic report", exact: true })
    .click();
  await expect(page.locator(".diagnostic-exported")).toContainText(report.id);
  const exported = JSON.parse(
    fs.readFileSync(
      path.join(destination, `opweaver-diagnostic-${report.id}`, "report.json"),
      "utf8",
    ),
  );
  expect(exported.records).toEqual(report.records);
  expect(
    fs.existsSync(
      path.join(project, ".opweaver/diagnostics", report.id, "temporary"),
    ),
  ).toBe(false);
  fs.writeFileSync(
    path.resolve("../test-results/p1c2-e2e-measurements.json"),
    JSON.stringify({ project, experiment, report }, null, 2),
  );
  await page.reload();
  await page
    .locator(".workspace-tabs")
    .getByRole("button", { name: "Optimization", exact: true })
    .click();
  await page
    .getByRole("button", { name: "Quantization diagnostics", exact: true })
    .click();
  await expect(
    page.getByLabel("Diagnostic history", { exact: true }),
  ).toHaveValue(report.id);
  await expect(page.locator(".diagnostic-status")).toHaveText("Succeeded");
  await expect(
    page.getByLabel("Tensor matching coverage", { exact: true }),
  ).toContainText("100%");
  await page.screenshot({ path: "../test-results/p1c2-diagnostics.png" });
  expect(errors).toEqual([]);
});
