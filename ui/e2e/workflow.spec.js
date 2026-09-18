import {expect, test} from "@playwright/test";
import {readFile} from "node:fs/promises";

test("live batch analysis, on-demand explanation and downloadable data", async ({
                                                                                    page,
                                                                                }) => {
    const explanationRequests = [];
    page.on("request", (request) => {
        if (request.url().endsWith("/explain")) explanationRequests.push(request);
    });
    await page.goto("/");
    await page.getByRole("button", {name: "Batch FASTA", exact: true}).click();
    await page
        .getByRole("button", {name: "Try an example", exact: true})
        .click();
    await page
        .getByRole("button", {name: "Analyze sequences", exact: true})
        .click();
    await expect(
        page.getByRole("heading", {name: "Your analysis"}),
    ).toBeVisible();
    expect(explanationRequests).toHaveLength(0);
    await page.getByRole("tab", {name: "Sequence", exact: true}).click();
    await expect(
        page.getByText("Amino-acid composition", {exact: true}),
    ).toBeVisible();
    await page.getByRole("tab", {name: "Explanation", exact: true}).click();
    await expect(page.getByText(/What influenced/)).toBeVisible();
    expect(explanationRequests).toHaveLength(1);
    await page.getByRole("tab", {name: "Prediction", exact: true}).click();
    await page.getByRole("tab", {name: "Explanation", exact: true}).click();
    expect(explanationRequests).toHaveLength(1);
    await page.getByText("Compare all proteins", {exact: false}).click();
    await page.getByRole("button", {name: "Length ↕", exact: true}).click();
    await expect(
        page.getByRole("columnheader", {name: "Length ↕"}),
    ).toHaveAttribute("aria-sort", "ascending");
    await page.getByText(/^Export/).click();
    const csvPromise = page.waitForEvent("download");
    await page
        .getByRole("button", {name: "Predictions CSV", exact: true})
        .click();
    const csv = await csvPromise;
    expect(csv.suggestedFilename()).toBe("mapexploc-predictions.csv");
    const csvText = await readFile(await csv.path(), "utf8");
    expect(csvText.split("\r\n").filter(Boolean)).toHaveLength(6);
    expect(csvText).toContain("probability_Nucleus");
    const jsonPromise = page.waitForEvent("download");
    await page
        .getByRole("button", {name: "Analysis JSON", exact: true})
        .click();
    const json = await jsonPromise;
    const data = JSON.parse(await readFile(await json.path(), "utf8"));
    expect(data.records).toHaveLength(5);
    expect(data.predictions.results).toHaveLength(5);
    expect(data.model.metadata_available).toBe(true);
    expect(Object.keys(data.explanations)).toEqual(["0"]);
    await page.getByRole("button", {name: /Back to sequences/}).click();
    await expect(page.getByRole("textbox")).toHaveValue(/SOCS2_HUMAN/);
});

test("file upload, actionable validation and small viewport", async ({
                                                                         page,
                                                                     }) => {
    await page.setViewportSize({width: 390, height: 844});
    await page.goto("/");
    await page.getByLabel("Upload FASTA file", {exact: true}).setInputFiles({
        name: "proteins.fasta",
        mimeType: "text/plain",
        buffer: Buffer.from(">one\nAAA\n>two\nCX"),
    });
    await page
        .getByRole("button", {name: "Analyze sequences", exact: true})
        .click();
    await expect(page.getByRole("alert")).toContainText(
        "two: unsupported residues (X)",
    );
    await page.getByRole("textbox").fill(">one\nAAA\n>two\nCCC");
    await page
        .getByRole("button", {name: "Analyze sequences", exact: true})
        .click();
    await expect(
        page.getByRole("heading", {name: "Your analysis"}),
    ).toBeVisible();
    expect(
        await page.evaluate(
            () =>
                document.documentElement.scrollWidth <=
                document.documentElement.clientWidth,
        ),
    ).toBe(true);
    await page.getByText("Model & methods", {exact: true}).click();
    expect(
        await page.evaluate(
            () =>
                document.documentElement.scrollWidth <=
                document.documentElement.clientWidth,
        ),
    ).toBe(true);
    await page.getByRole("tab", {name: "Prediction", exact: true}).focus();
    await page.keyboard.press("ArrowRight");
    await expect(
        page.getByRole("tab", {name: "Sequence", exact: true}),
    ).toBeFocused();
});
