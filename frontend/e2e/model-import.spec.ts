import { test, expect, type Page } from "@playwright/test";
import path from "node:path";
import fs from "node:fs";
import crypto from "node:crypto";
const hash = (p: string) =>
  crypto.createHash("sha256").update(fs.readFileSync(p)).digest("hex");
async function setup(page: Page) {
  await page.goto("/");
  await expect(
    page.getByText("Worker connected", { exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "New", exact: true }).click();
  const dialog = page.getByRole("dialog");
  await dialog
    .getByLabel("Project name", { exact: true })
    .fill("Model imports");
  await dialog
    .getByLabel("Project directory", { exact: true })
    .fill(path.resolve(`../projects/import-${Date.now()}`));
  await dialog
    .getByRole("combobox", { name: "Starting point" })
    .selectOption("current");
  await dialog
    .getByRole("button", { name: "Create project", exact: true })
    .click();
  await expect(dialog).toBeHidden();
  await page
    .locator(".workspace-tabs")
    .getByRole("button", { name: "Model Imports", exact: true })
    .click();
}
async function start(page: Page, name: string, python = false) {
  await page
    .locator(".import-toolbar")
    .getByRole("button", { name: "Import model", exact: true })
    .click();
  const dialog = page.getByRole("dialog", {
    name: "Import model",
    exact: true,
  });
  await dialog
    .getByLabel("Model source path")
    .fill(path.resolve(`../examples/${python ? "pytorch" : "onnx"}/${name}`));
  await dialog
    .getByRole("button", {
      name: "Detect format / inspect entry points",
      exact: true,
    })
    .click();
  await expect(dialog.locator(".import-detected")).toBeVisible();
  await dialog
    .getByRole("button", {
      name: python ? "Import and capture Python" : "Import and analyze",
      exact: true,
    })
    .click();
  await expect(dialog).toBeHidden();
}
async function ready(page: Page, name: string) {
  await expect(page.locator(".import-summary-strip strong")).toHaveText(name, {
    timeout: 45000,
  });
  await expect(page.locator(".import-progress")).toHaveCount(0, {
    timeout: 45000,
  });
  await expect(page.locator(".import-error")).toHaveCount(0);
}
test("ONNX CNN inspector, search, groups, persistence, error retention and Design Graph", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  const source = path.resolve("../examples/onnx/tiny-cnn.onnx"),
    before = hash(source);
  await setup(page);
  await start(page, "tiny-cnn.onnx");
  await ready(page, "Tiny CNN");
  await expect(page.locator(".model-overview-metrics")).toContainText("5");
  await expect(page.locator(".model-inspector")).toContainText("113");
  await expect(page.locator(".model-inspector")).toContainText("[B, 3, 8, 8]");
  await expect(page.locator(".model-inspector")).toContainText("Not measured");
  await page.getByLabel("Search model nodes").fill("encoder/pool");
  await page.locator(".model-search-results button").click();
  await expect(page.locator(".model-inspector")).toContainText("MaxPool");
  await expect(page.locator(".model-inspector")).toContainText("[B, 2, 3, 3]");
  await page.getByLabel("Group model nodes").selectOption("operator");
  await page.getByRole("button", { name: "Fit graph", exact: true }).click();
  await page.locator(".import-graph-area .vue-flow__node").first().click();
  await expect(page.locator(".collapsed-group")).toBeVisible();
  await page
    .getByRole("button", { name: "Expand all nodes", exact: true })
    .click();
  const artifactId = await page.getByLabel("Imported model").inputValue();
  await start(page, "corrupt.onnx");
  await expect(page.locator(".import-error")).toContainText(
    "previous valid analysis",
    { timeout: 45000 },
  );
  await expect(page.locator(".import-summary-strip strong")).toHaveText(
    "Tiny CNN",
  );
  await page.reload();
  await page
    .locator(".workspace-tabs")
    .getByRole("button", { name: "Model Imports", exact: true })
    .click();
  await ready(page, "Tiny CNN");
  await expect(page.getByLabel("Imported model")).toHaveValue(artifactId);
  await page.screenshot({ path: "../test-results/p0b-model-inspector.png" });
  expect(hash(source)).toBe(before);
  await page
    .locator(".workspace-tabs")
    .getByRole("button", { name: "Model Graph", exact: true })
    .click();
  await expect(page.locator('.vue-flow__node[data-id="head"]')).toContainText(
    "Linear",
  );
  expect(errors).toEqual([]);
});
test("custom operators, dynamic dimensions, external weights and independent If subgraphs", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await setup(page);
  await start(page, "custom.onnx");
  await ready(page, "Custom operator");
  await expect(page.locator(".model-inspector")).toContainText(
    "UNKNOWN_OPERATOR",
  );
  await page.getByLabel("Search model nodes").fill("vendor/custom");
  await page.locator(".model-search-results button").click();
  await expect(page.locator(".model-inspector")).toContainText("com.example");
  await expect(page.locator(".model-inspector")).toContainText("0.25");
  await start(page, "dynamic.onnx");
  await ready(page, "Dynamic shapes");
  await expect(page.locator(".model-inspector")).toContainText("[B, ?, 4]");
  await start(page, "external-cnn.onnx");
  await ready(page, "Tiny CNN");
  await expect(page.locator(".model-inspector")).toContainText("weights.bin");
  await expect(page.locator(".model-inspector")).toContainText("EXTERNAL_DATA");
  await start(page, "conditional.onnx");
  await ready(page, "Conditional branches");
  await page
    .getByLabel("Model subgraph")
    .selectOption({ label: "Then branch · 0/then_branch:0/" });
  await expect(page.locator(".import-graph-area")).toContainText("double");
  await expect(page.locator(".import-graph-area")).toContainText(
    "captured_input",
  );
  await start(page, "large-chain.onnx");
  await ready(page, "Large chain");
  await expect(page.getByLabel("Group model nodes")).toHaveValue("operator");
  await expect(page.locator(".import-graph-area .vue-flow__node")).toHaveCount(
    3,
  );
  await page.getByLabel("Search model nodes").fill("stack/layer_599");
  await page.locator(".model-search-results button").click();
  await expect(page.locator(".model-inspector")).toContainText(
    "stack/layer_599",
  );
  await page.reload();
  await page
    .locator(".workspace-tabs")
    .getByRole("button", { name: "Model Imports", exact: true })
    .click();
  await ready(page, "Large chain");
  await expect(page.getByLabel("Group model nodes")).toHaveValue("operator");
  expect(errors).toEqual([]);
});
test("unified PyTorch importer opens existing source collaboration", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await setup(page);
  await start(page, "residual_model.py", true);
  await ready(page, "ResidualModel");
  await expect(page.locator(".model-inspector")).toContainText(
    "PyTorch parameters",
  );
  await page.getByRole("button", { name: "Open Python", exact: true }).click();
  await expect(page.locator(".sync-status")).toHaveText("Read-only analysis");
  await expect(
    page.locator('[data-testid="python-editor"] .monaco-editor'),
  ).toBeVisible();
  await expect(
    page.getByRole("combobox", { name: "Model class or factory" }),
  ).toHaveValue("ResidualModel");
  expect(errors).toEqual([]);
});
