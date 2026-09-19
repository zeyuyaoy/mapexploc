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
    expect(data.model.metadata.model_family).toBe("RandomForestClassifier");
    expect(data.model.metadata.evaluation_status).toBe("historical_holdout");
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
    await expect(
        page.getByText(/Evaluation: Version 1 held-out evaluation/),
    ).toBeVisible();
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

test("cancellation and explanation failure preserve the working analysis", async ({
                                                                                      page,
                                                                                  }) => {
    let cancelPending;
    const pending = new Promise((resolve) => {
        cancelPending = resolve;
    });
    let intercepted;
    const started = new Promise((resolve) => {
        intercepted = resolve;
    });
    await page.route("**/predict", async (route) => {
        intercepted();
        await pending;
        await route.abort();
    });
    await page.goto("/");
    await page.getByRole("textbox").fill("AAAA");
    await page
        .getByRole("button", {name: "Analyze sequences", exact: true})
        .click();
    await started;
    await page.getByRole("button", {name: "Cancel", exact: true}).click();
    // Release the test route after the application aborts the request.
    await page.unrouteAll({behavior: "ignoreErrors"});
    cancelPending();
    await expect(page.getByRole("textbox")).toHaveValue("AAAA");
    await page
        .getByRole("button", {name: "Analyze sequences", exact: true})
        .click();
    await expect(
        page.getByRole("heading", {name: "Your analysis"}),
    ).toBeVisible();
    await page.route("**/explain", (route) =>
        route.fulfill({
            status: 503,
            contentType: "application/json",
            body: JSON.stringify({detail: "Explanation temporarily unavailable"}),
        }),
    );
    await page.getByRole("tab", {name: "Explanation", exact: true}).click();
    await expect(page.getByRole("alert")).toContainText(
        "The model is unavailable",
    );
    await expect(page.getByText(/Predicted compartment:/)).toBeVisible();
    await page.unroute("**/explain");
    await page
        .getByRole("button", {name: "Retry explanation", exact: true})
        .click();
    await expect(page.getByText(/What influenced/)).toBeVisible();
});

test("complete report import, class/filter identity, overlays and unchanged export", async ({
                                                                                                page,
                                                                                            }) => {
    const fixture = JSON.parse(
        await readFile(
            new URL("../src/test-fixtures/report-v2.json", import.meta.url),
            "utf8",
        ),
    );
    await page.goto("/");
    await page
        .getByLabel("Import complete explanation report", {exact: true})
        .setInputFiles({
            name: "report.json",
            mimeType: "application/json",
            buffer: Buffer.from(JSON.stringify(fixture)),
        });
    await expect(
        page.getByRole("heading", {name: "Explanation report", exact: true}),
    ).toBeVisible();
    await page
        .getByLabel("Protein", {exact: true})
        .selectOption("outside-control");
    await page
        .getByRole("button", {name: "Protein order: A–Z", exact: true})
        .click();
    await expect(
        page.getByRole("heading", {name: "outside-control", exact: true}),
    ).toBeVisible();
    await page
        .getByLabel("Explained class", {exact: true})
        .selectOption("inside");
    await expect(
        page.getByRole("heading", {
            name: "Signed contributions for inside",
            exact: true,
        }),
    ).toBeVisible();
    await page.getByText("Annotation provenance", {exact: true}).click();
    await expect(page.getByText(/Synthetic UI fixture 1/)).toBeVisible();
    await page
        .getByLabel("Localization filter", {exact: true})
        .selectOption("inside");
    await expect(
        page.getByRole("heading", {name: "inside-control", exact: true}),
    ).toBeVisible();
    const downloadPromise = page.waitForEvent("download");
    await page
        .getByRole("button", {name: "Export complete JSON", exact: true})
        .click();
    const downloaded = await downloadPromise;
    expect(JSON.parse(await readFile(await downloaded.path(), "utf8"))).toEqual(
        fixture,
    );
    await page.getByLabel("View", {exact: true}).selectOption("Global cohort");
    await expect(page.getByText(/2 included \/ 2 supplied/)).toBeVisible();
    await page.setViewportSize({width: 390, height: 844});
    expect(
        await page.evaluate(
            () =>
                document.documentElement.scrollWidth <=
                document.documentElement.clientWidth,
        ),
    ).toBe(true);
    await page.setViewportSize({width: 320, height: 844});
    expect(
        await page.evaluate(
            () =>
                document.documentElement.scrollWidth <=
                document.documentElement.clientWidth,
        ),
    ).toBe(true);
});

test("live backend produces a complete all-class report", async ({page}) => {
    await page.goto("/");
    await page.getByRole("textbox").fill("MALWMRLLPLLALLALWGPDPAAA");
    await page
        .getByRole("button", {name: "Analyze sequences", exact: true})
        .click();
    await page
        .getByRole("button", {name: "Generate complete report", exact: true})
        .click();
    await expect(
        page.getByRole("heading", {name: "Explanation report", exact: true}),
    ).toBeVisible();
    await expect(
        page.getByLabel("Explained class", {exact: true}).locator("option"),
    ).toHaveCount(5);
    await expect(
        page.getByText(/^Method: tree\. Approximation: exact_tree_algorithm/),
    ).toBeVisible();
});
