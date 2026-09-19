import {expect, test} from "@playwright/test";
import {readFile} from "node:fs/promises";

test("mode selection reaches backend identifiers and exports a bound CLI specification", async ({
                                                                                                    page,
                                                                                                }) => {
    const modes = ["fast", "accurate"].map((mode) => ({
        adapter_id: `fixture-${mode}`,
        mode,
        model_id: `DeepLoc-2.1-${mode === "fast" ? "Fast" : "Accurate"}`,
        readiness: "configured",
        device: "fixture",
        tradeoff: "Software contract fixture only",
    }));
    await page.route("**/v3/models", (route) =>
        route.fulfill({json: {models: modes}}),
    );
    await page.route("**/v3/predict", (route) => {
        const body = route.request().postDataJSON();
        const model = modes.find((m) => m.adapter_id === body.adapter_id);
        return route.fulfill({
            json: {
                model: {
                    model_id: model.model_id,
                    classes: ["Nucleus", "Extracellular"],
                    checkpoint_sha256: "a".repeat(64),
                    provenance: {mode: model.mode},
                },
                results: body.proteins.map((protein) => ({
                    protein,
                    probabilities: [0.7, 0.4],
                    decisions: ["Nucleus"],
                })),
            },
        });
    });
    await page.goto("/");
    await page.getByRole("button", {name: "DeepLoc Fast / Accurate"}).click();
    await page.getByLabel("DeepLoc model mode").selectOption("fixture-accurate");
    await page
        .getByLabel("Protein sequences (FASTA)")
        .fill(">p\nACDEFGHIKLMNPQRSTVWY");
    await page
        .getByRole("button", {name: "Predict with selected mode"})
        .click();
    await expect(
        page.getByRole("heading", {name: "DeepLoc-2.1-Accurate"}),
    ).toBeVisible();
    const pending = page.waitForEvent("download");
    await page
        .getByRole("button", {name: "Export exact run configuration"})
        .click();
    const download = await pending;
    const config = JSON.parse(await readFile(await download.path(), "utf8"));
    expect(config.mode).toBe("accurate");
    expect(config.expected_model_id).toBe("DeepLoc-2.1-Accurate");
    expect(config.configuration.method_profile).toBe("v2x-legacy");
    await page.getByLabel("DeepLoc model mode").selectOption("fixture-fast");
    await expect(
        page.getByRole("heading", {name: "DeepLoc-2.1-Accurate"}),
    ).toHaveCount(0);
});

test("schema-v3 import exposes diagnostics and preserves full exports", async ({
                                                                                   page,
                                                                               }) => {
    await page.goto("/");
    await page
        .getByLabel("Import complete explanation report")
        .setInputFiles("src/test-fixtures/report-v3.json");
    await expect(
        page.getByRole("heading", {name: "Explanation report"}),
    ).toBeVisible();
    await expect(page.getByText(/Complete report · schema v3/)).toBeVisible();
    await expect(
        page.getByRole("heading", {name: "Attribution stability"}),
    ).toBeVisible();
    await page.getByLabel("View", {exact: true}).selectOption("Global cohort");
    await page
        .getByText("Equal-group statistics and uncertainty", {exact: true})
        .click();
    await expect(
        page.getByRole("columnheader", {name: "95% bootstrap interval"}),
    ).toBeVisible();
    const pending = page.waitForEvent("download");
    await page.getByRole("button", {name: "Export complete JSON"}).click();
    const download = await pending;
    expect(await readFile(await download.path(), "utf8")).toBe(
        await readFile("src/test-fixtures/report-v3.json", "utf8"),
    );
});
