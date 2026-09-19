// Run only against a trusted operator API with real installed native assets.
import {chromium, expect} from "@playwright/test";
import {readFile, writeFile} from "node:fs/promises";

const [url, adapterId, mode, output] = process.argv.slice(2);
if (!url || !adapterId || !["fast", "accurate"].includes(mode) || !output)
    throw new Error(
        "Usage: node scripts/qualify-native.mjs URL ADAPTER_ID MODE evidence.json",
    );
const browser = await chromium.launch({headless: true});
try {
    const page = await browser.newPage();
    await page.goto(url);
    await page.getByRole("button", {name: "DeepLoc Fast / Accurate"}).click();
    await expect(page.getByLabel("DeepLoc model mode")).toContainText("Fast", {
        timeout: 60000,
    });
    const catalogUrl = new URL("/v3/models", url).href;
    const catalogue = await (await page.request.get(catalogUrl)).json();
    const selected = catalogue.models.find((m) => m.adapter_id === adapterId);
    expect(selected.mode).toBe(mode);
    expect(selected.readiness).toBe("configured");
    for (const entry of catalogue.models.filter((m) => m.mode))
        await expect(
            page.locator(`option[value="${entry.adapter_id}"]`),
        ).toHaveJSProperty("disabled", entry.readiness === "unavailable");
    await page.getByLabel("DeepLoc model mode").selectOption(adapterId);
    await page
        .getByLabel("Protein sequences (FASTA)")
        .fill(">native-ui\nACDEFGHIKLMNPQRSTVWY");
    const pending = page.waitForResponse((r) => r.url().endsWith("/v3/predict"), {
        timeout: 120000,
    });
    await page
        .getByRole("button", {name: "Predict with selected mode"})
        .click();
    const response = await pending;
    expect(response.status()).toBe(200);
    const prediction = await response.json();
    expect(prediction.model.provenance.mode).toBe(mode);
    expect(prediction.model.checkpoint_sha256).toBe(
        selected.expected_checkpoint_sha256,
    );
    await expect(
        page.getByRole("heading", {name: prediction.model.model_id}),
    ).toBeVisible();
    const pendingDownload = page.waitForEvent("download");
    await page
        .getByRole("button", {name: "Export exact run configuration"})
        .click();
    const download = await pendingDownload;
    const specification = JSON.parse(
        await readFile(await download.path(), "utf8"),
    );
    expect(specification.mode).toBe(mode);
    expect(specification.expected_checkpoint_sha256).toBe(
        prediction.model.checkpoint_sha256,
    );
    const refreshed = await (await page.request.get(catalogUrl)).json();
    expect(
        refreshed.models.find((m) => m.adapter_id === adapterId).readiness,
    ).toBe("configured");
    await writeFile(
        output,
        JSON.stringify(
            {
                status: "passed",
                native_prediction: prediction,
                exported_specification: specification,
                resident_readiness_verified: true,
                unavailable_controls_verified: true,
            },
            null,
            2,
        ) + "\n",
    );
} finally {
    await browser.close();
}
