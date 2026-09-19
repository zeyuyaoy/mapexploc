import { defineConfig } from "@playwright/test";
import { fileURLToPath } from "node:url";

const ci = Boolean(process.env.CI);

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  forbidOnly: ci,
  retries: 0,
  webServer: ci
    ? [
        {
          command:
            "python -m uvicorn mapexploc.api:app --host 127.0.0.1 --port 8000",
          cwd: fileURLToPath(new URL("..", import.meta.url)),
          url: "http://127.0.0.1:8000/model",
          reuseExistingServer: false,
          timeout: 120000,
        },
        {
          command: "pnpm preview --host 127.0.0.1 --port 5173 --strictPort",
          url: "http://127.0.0.1:5173",
          reuseExistingServer: false,
          timeout: 120000,
        },
      ]
    : undefined,
  use: {
    baseURL: ci
      ? "http://127.0.0.1:5173"
      : process.env.MAPEXPLOC_UI_URL || "http://127.0.0.1:5173",
    headless: true,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    launchOptions:
      !ci && process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE
        ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE }
        : {},
  },
});
