import { expect, test } from "@playwright/test";
import { execFileSync } from "node:child_process";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

test("public API preserves approved prediction and explanation values", async ({
  baseURL,
}) => {
  test.setTimeout(120000);
  execFileSync(
    "python",
    [
      fileURLToPath(new URL("../../scripts/smoke_web.py", import.meta.url)),
      baseURL,
      "--timeout",
      "15",
    ],
    { stdio: "inherit", timeout: 90000 },
  );
});

test("public UI exposes owned inference and local report import only", async ({
  page,
}) => {
  await page.goto("/");
  await expect(
    page.getByRole("button", { name: /DeepLoc|native|Accurate|Fast \//i }),
  ).toHaveCount(0);
  await expect(page.getByLabel("DeepLoc model mode")).toHaveCount(0);
  await expect(
    page.getByRole("button", { name: "Import report", exact: true }),
  ).toBeVisible();
  const catalog = await page.request.get("/api/v3/models");
  expect(catalog.ok()).toBe(true);
  const { models } = await catalog.json();
  expect(models.map((model) => model.adapter_id)).toEqual(["default"]);
  expect(models[0].descriptor.preprocessing_id).toBe(
    "mapexploc:engineered-423:v1",
  );
});

test("schema-v3 import exposes diagnostics and preserves full exports", async ({
  page,
}) => {
  await page.goto("/");
  await page
    .getByLabel("Import complete explanation report")
    .setInputFiles("src/test-fixtures/report-v3.json");
  await expect(
    page.getByRole("heading", { name: "Explanation report" }),
  ).toBeVisible();
  await expect(page.getByText(/Complete report · schema v3/)).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Attribution stability" }),
  ).toBeVisible();
  await page.getByLabel("View", { exact: true }).selectOption("Global cohort");
  await page
    .getByText("Equal-group statistics and uncertainty", { exact: true })
    .click();
  await expect(
    page.getByRole("columnheader", { name: "95% bootstrap interval" }),
  ).toBeVisible();
  const pending = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export complete JSON" }).click();
  const download = await pending;
  expect(await readFile(await download.path(), "utf8")).toBe(
    await readFile("src/test-fixtures/report-v3.json", "utf8"),
  );
});
