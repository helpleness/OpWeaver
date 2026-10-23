import { test, expect, type Page } from "@playwright/test";
import path from "node:path";

async function project(page: Page, name: string, template = "current") {
  await page.goto("/");
  await expect(
    page.getByText("Worker connected", { exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "New", exact: true }).click();
  const dialog = page.getByRole("dialog");
  const directory = path.resolve(`../projects/${name}-${Date.now()}`);
  await dialog.getByLabel("Project name", { exact: true }).fill(name);
  await dialog.getByLabel("Project directory", { exact: true }).fill(directory);
  await dialog
    .getByRole("combobox", { name: "Starting point", exact: true })
    .selectOption(template);
  await dialog
    .getByRole("button", { name: "Create project", exact: true })
    .click();
  await expect(dialog).toBeHidden();
  return directory;
}
async function readSource(page: Page, project: string, name = "src/model.py") {
  const response = await page.request.post("/api/rpc", {
    headers: { "X-OpWeaver": "1" },
    data: { method: "workspace.read", params: { project, path: name } },
  });
  const data = await response.json();
  expect(data.error).toBeUndefined();
  return data.result.text as string;
}
async function selectSourceText(page: Page, text: string) {
  const input = page.getByRole("textbox", {
    name: "Python source editor",
    exact: true,
  });
  await input.focus();
  await page.keyboard.press("Control+f");
  await expect(page.locator(".find-widget")).toBeVisible();
  await page.keyboard.press("Control+A");
  await page.keyboard.insertText(text);
  await page.keyboard.press("Escape");
}
async function saveEditor(page: Page) {
  await page
    .getByRole("button", { name: "Save Python file", exact: true })
    .click();
  await expect(page.locator(".source-actionbar")).toContainText("Source saved");
}
async function parse(page: Page, state: string) {
  await page.getByRole("button", { name: "Parse graph", exact: true }).click();
  await expect(page.locator(".sync-status")).toHaveText(state, {
    timeout: 45000,
  });
}
async function generate(page: Page) {
  await page
    .getByRole("button", { name: "Graph → Code · review diff", exact: true })
    .click();
  await page
    .getByRole("button", { name: "Preview source diff", exact: true })
    .click();
  const diff = page.getByRole("dialog", { name: "Review generated source" });
  await expect(diff.locator(".monaco-diff-editor")).toBeVisible();
  await diff
    .getByRole("button", { name: "Apply reviewed change", exact: true })
    .click();
  await expect(diff).toBeHidden();
}

test("Visual Graph ↔ Python, source navigation, stale graph on parse failure", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  const directory = await project(page, "visual-code");
  await page.getByRole("button", { name: "Code", exact: true }).click();
  await generate(page);
  await parse(page, "Synced");
  await page.getByRole("button", { name: "Fit graph", exact: true }).click();
  await page.locator('.vue-flow__node[data-id="head"]').click();
  await expect(
    page.locator('[data-testid="python-editor"] .selected-text').first(),
  ).toBeVisible();
  await page.getByLabel("out_features", { exact: true }).fill("5");
  await page.getByLabel("out_features", { exact: true }).press("Tab");
  await generate(page);
  await expect
    .poll(() => readSource(page, directory))
    .toContain("out_features=5");
  await parse(page, "Synced");
  let source = await readSource(page, directory);
  await selectSourceText(page, "nn.ReLU()");
  await page.keyboard.insertText("nn.GELU()");
  await saveEditor(page);
  await expect(page.locator(".sync-status")).toContainText("stale");
  await expect(page.locator('.vue-flow__node[data-id="relu"]')).toContainText(
    "ReLU",
  );
  await parse(page, "Synced");
  await expect(page.locator('.vue-flow__node[data-id="relu"]')).toContainText(
    "GELU",
  );
  await page.reload();
  await expect(page.locator(".sync-status")).toHaveText("Synced");
  await expect(page.locator('.vue-flow__node[data-id="relu"]')).toContainText(
    "GELU",
  );
  await page.getByRole("button", { name: "Code", exact: true }).click();
  source = await readSource(page, directory);
  const broken = source.replace(
    "return tensor_head",
    "if input_0.sum() > 0:\n            return tensor_head\n        return tensor_head",
  );
  await selectSourceText(page, "return tensor_head");
  await page.keyboard.insertText("if input_0.sum() > 0:");
  await page.keyboard.press("Enter");
  await page.keyboard.insertText("return tensor_head");
  await page.keyboard.press("Enter");
  await page.keyboard.press("Shift+Tab");
  await page.keyboard.insertText("return tensor_head");
  await saveEditor(page);
  await parse(page, "Parse failed · graph stale");
  await expect(page.locator('.vue-flow__node[data-id="relu"]')).toContainText(
    "GELU",
  );
  expect(await readSource(page, directory)).toBe(broken);
  await expect(page.locator(".code-error")).toContainText("control flow");
  await page.screenshot({ path: "../test-results/p0a-stale-graph.png" });
  expect(errors).toEqual([]);
});

test("Code-First import, exact source patch, reparse and durable reload", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  const directory = await project(page, "code-first", "empty");
  await page.getByRole("button", { name: "Code", exact: true }).click();
  await page
    .getByRole("button", { name: "Import Python file", exact: true })
    .first()
    .click();
  await page
    .getByLabel("Python source path")
    .fill(path.resolve("../examples/pytorch/residual_model.py"));
  await page
    .getByRole("button", { name: "Import source", exact: true })
    .click();
  await expect(
    page.getByRole("combobox", { name: "Model class or factory" }),
  ).toHaveValue("ResidualModel");
  await parse(page, "Read-only analysis");
  await expect(page.locator(".analysis-canvas .vue-flow__node")).toHaveCount(7);
  await expect(
    page.getByRole("button", { name: "Problems", exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Fit graph", exact: true }).click();
  await page.locator('.vue-flow__node[data-id="module_head"]').click();
  await page
    .getByRole("combobox", { name: "Source patch parameter" })
    .selectOption("out_features");
  await page.getByRole("textbox", { name: "Source patch value" }).fill("5");
  await page
    .getByRole("button", { name: "Review source patch", exact: true })
    .click();
  await expect(
    page.getByRole("dialog", { name: "Review generated source" }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Apply reviewed change", exact: true })
    .click();
  await parse(page, "Read-only analysis");
  const source = await readSource(page, directory, "src/residual_model.py");
  expect(source).toContain("self.head = nn.Linear(8, 5)");
  expect(source).toContain("def describe_model():");
  await page
    .getByRole("button", { name: "Tensor Inspector", exact: true })
    .click();
  await expect(page.locator(".tensor-result")).toContainText("[4, 5]");
  await expect(
    page.locator('.vue-flow__node[data-id="module_head"]'),
  ).toContainText("5");
  await page.getByRole("button", { name: "Hide inspector" }).click();
  await page.screenshot({ path: "../test-results/p0a-split.png" });
  await page.reload();
  await expect(page.locator(".sync-status")).toHaveText("Read-only analysis");
  await expect(
    page.getByRole("combobox", { name: "Model editing mode" }),
  ).toHaveValue("code-first");
  await expect(page.locator(".analysis-canvas .vue-flow__node")).toHaveCount(7);
  expect(errors).toEqual([]);
});

test("continuous dragging does not snap backwards or change cursor at child boundaries", async ({
  page,
}) => {
  await project(page, "pointer-stability");
  await page.getByRole("button", { name: "Fit graph", exact: true }).click();
  await page.waitForTimeout(200);
  const node = page.locator('.vue-flow__node[data-id="input"]');
  const box = (await node.boundingBox())!;
  await page.mouse.move(box.x + 35, box.y + 20);
  await page.mouse.down();
  const samples: number[] = [];
  for (let step = 1; step <= 35; step++) {
    await page.mouse.move(box.x + 35 + step * 3, box.y + 20);
    samples.push((await node.boundingBox())!.x);
  }
  expect(await node.evaluate((n) => getComputedStyle(n).cursor)).toBe(
    "grabbing",
  );
  await page.mouse.up();
  const released = (await node.boundingBox())!.x;
  for (let i = 1; i < samples.length; i++) {
    expect(samples[i] - samples[i - 1]).toBeGreaterThanOrEqual(-1);
    expect(samples[i] - samples[i - 1]).toBeLessThan(12);
  }
  expect(Math.abs(released - samples.at(-1)!)).toBeLessThan(2);
  expect(Math.abs(released - box.x - 105)).toBeLessThan(5);
});
