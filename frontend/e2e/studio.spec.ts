import { test, expect, type Locator, type Page } from "@playwright/test";
import path from "node:path";

async function connect(page: Page, source: Locator, target: Locator) {
  const a = await source.locator(".vue-flow__handle.source").boundingBox();
  const b = await target.locator(".vue-flow__handle.target").boundingBox();
  if (!a || !b) throw new Error("Ports are not visible");
  await page.mouse.move(a.x + a.width / 2, a.y + a.height / 2);
  await page.mouse.down();
  await page.mouse.move(b.x + b.width / 2, b.y + b.height / 2, { steps: 15 });
  await page.mouse.up();
}

test("empty canvas → connect → train → reopen → resume; responsive layout", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  await expect(
    page.getByText("Worker connected", { exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "New", exact: true }).click();
  const dialog = page.getByRole("dialog");
  await dialog
    .getByLabel("Project name", { exact: true })
    .fill("UI acceptance");
  await dialog
    .getByLabel("Project directory", { exact: true })
    .fill(path.resolve(`../projects/e2e-${Date.now()}`));
  await dialog
    .getByRole("button", { name: "Create project", exact: true })
    .click();
  await expect(dialog).toBeHidden();
  await expect(page.locator(".canvas-empty")).toBeVisible();
  await page
    .locator(".operator-item")
    .filter({ hasText: /^Input/ })
    .dragTo(page.locator(".canvas-wrap"), {
      targetPosition: { x: 220, y: 180 },
    });
  await expect(page.locator(".vue-flow__node")).toHaveCount(1);
  for (const name of ["Linear", "ReLU", "Linear", "Output"]) {
    await page
      .locator(".operator-item")
      .filter({ hasText: new RegExp(`^${name}`) })
      .click();
  }
  await expect(page.locator(".vue-flow__node")).toHaveCount(5);
  await page
    .getByRole("button", { name: "Arrange nodes", exact: true })
    .click();
  const nodes = page.locator(".vue-flow__node");
  const linears = nodes.filter({ hasText: /Linear/ });
  await linears.nth(1).click();
  await page.getByLabel("in_features", { exact: true }).fill("16");
  await page.getByLabel("in_features", { exact: true }).press("Tab");
  await page.getByLabel("out_features", { exact: true }).fill("3");
  await page.getByLabel("out_features", { exact: true }).press("Tab");
  await expect(linears.nth(1)).toContainText("16 → 3");
  await page.getByRole("button", { name: "Fit graph", exact: true }).click();
  await expect(page.locator(".canvas-caption")).toContainText("graph issues");
  // Wait for the explicit fit animation before using port coordinates.
  await page.waitForTimeout(350);
  await connect(page, nodes.filter({ hasText: /^Input/ }), linears.nth(0));
  await connect(page, linears.nth(0), nodes.filter({ hasText: /^ReLU/ }));
  await connect(page, nodes.filter({ hasText: /^ReLU/ }), linears.nth(1));
  await connect(page, linears.nth(1), nodes.filter({ hasText: /^Output/ }));
  await expect(page.locator(".vue-flow__edge")).toHaveCount(4);
  await expect(page.locator(".canvas-caption")).toContainText("Graph is valid");
  await nodes.filter({ hasText: /^Input/ }).click();
  await linears.nth(0).click({ modifiers: ["Shift"] });
  await expect(page.locator(".vue-flow__node.selected")).toHaveCount(2);
  await page.getByRole("button", { name: "Save", exact: true }).click();
  await expect(
    page.getByText("All changes saved", { exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Forward", exact: true }).click();
  await expect(
    page.getByText("FORWARD COMPLETE", { exact: true }),
  ).toBeVisible();
  await expect(page.locator(".tensor-result")).toContainText("[4, 3]");
  await page.getByRole("button", { name: "Train model", exact: true }).click();
  await expect(page.locator(".job-indicator")).toContainText("COMPLETED", {
    timeout: 45000,
  });
  await expect(page.locator(".loss-chart polyline")).toBeVisible();
  await page.screenshot({ path: "../test-results/studio-trained.png" });
  await page.getByRole("button", { name: "Console", exact: true }).click();
  await expect(page.locator(".console-output")).toContainText("Epoch 20/20");
  await page.reload();
  await expect(
    page.getByText("All changes saved", { exact: true }),
  ).toBeVisible();
  await expect(page.locator(".vue-flow__edge")).toHaveCount(4);
  await expect(page.locator(".job-indicator")).toContainText("COMPLETED");
  await page.getByRole("button", { name: "Resume", exact: true }).click();
  await page.getByLabel("Total target epochs").fill("22");
  await page
    .getByRole("button", { name: "Resume training", exact: true })
    .click();
  await expect(page.getByRole("dialog")).toBeHidden();
  await expect(page.locator(".job-indicator")).toContainText("COMPLETED", {
    timeout: 45000,
  });
  await expect(page.locator(".metric-readouts")).toContainText("22");
  await page.getByRole("button", { name: "Console", exact: true }).click();
  await expect(page.locator(".console-output")).toContainText(
    "Restored checkpoint at epoch 21",
  );
  await page.setViewportSize({ width: 960, height: 680 });
  await expect(
    page.getByRole("button", { name: "Train model", exact: true }),
  ).toBeVisible();
  const geometry = await page.locator(".canvas-wrap").boundingBox();
  expect(geometry?.width).toBeGreaterThan(300);
  expect(geometry?.height).toBeGreaterThan(180);
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth > window.innerWidth,
  );
  expect(overflow).toBe(false);
  await page.screenshot({ path: "../test-results/studio-960.png" });
  expect(errors).toEqual([]);
});
