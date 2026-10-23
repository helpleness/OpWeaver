import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  workers: 1,
  timeout: 90000,
  expect: { timeout: 15000 },
  use: {
    actionTimeout: 15000,
    baseURL: process.env.OPWEAVER_E2E_URL ?? "http://127.0.0.1:34115",
    channel: "chrome",
    headless: true,
    viewport: { width: 1440, height: 1000 },
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
  },
  outputDir: "../test-results/browser",
  reporter: "list",
});
