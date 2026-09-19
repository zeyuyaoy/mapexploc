// Explicit real-report release gate; missing inputs fail rather than skip.
import { chromium, expect } from "@playwright/test";
import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

const [url, output, ...files] = process.argv.slice(2);
if (!url || !output || files.length === 0)
  throw new Error(
    "Usage: node scripts/qualify-reports.mjs URL evidence.json report.json [...]",
  );
const browser = await chromium.launch({ headless: true });
const results = [];
try {
  for (const file of files) {
    const original = await readFile(file, "utf8");
    const report = JSON.parse(original);
    const page = await browser.newPage();
    await page.goto(url);
    await page
      .getByLabel("Import complete explanation report")
      .setInputFiles(resolve(file));
    await expect(
      page.getByRole("heading", { name: "Explanation report", exact: true }),
    ).toBeVisible();
    await expect(page.getByLabel("Explained class")).toHaveValue(
      report.model.classes[0],
    );
    await page
      .getByLabel("Explained class")
      .selectOption(report.model.classes.at(-1));
    await page
      .getByLabel("View", { exact: true })
      .selectOption("Global cohort");
    await page
      .getByText("Equal-group statistics and uncertainty", { exact: true })
      .click();
    await expect(
      page.getByRole("columnheader", { name: "95% bootstrap interval" }),
    ).toBeVisible();
    const pending = page.waitForEvent("download");
    await page.getByRole("button", { name: "Export complete JSON" }).click();
    const download = await pending;
    expect(await readFile(await download.path(), "utf8")).toBe(original);
    results.push({
      model_id: report.model.model_id,
      mode: report.runtime?.model_mode,
      schema_version: report.schema_version,
      proteins: report.results.length,
      status: "passed",
      exact_export: true,
    });
    await page.close();
  }
  await writeFile(
    output,
    JSON.stringify({ status: "passed", results }, null, 2) + "\n",
  );
} finally {
  await browser.close();
}
