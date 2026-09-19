import {afterEach, expect, it, vi} from "vitest";
import {render, screen} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import NativeWorkflow from "./components/NativeWorkflow";
import {compareReports} from "./comparison";
import {exportReportText, readReportText, validateReport} from "./reports";
import fixture from "./test-fixtures/report-v2.json";
import modern from "./test-fixtures/report-v3.json";

afterEach(() => vi.unstubAllGlobals());
it("selects the real backend mode and discards stale results", async () => {
    const user = userEvent.setup();
    const calls = [];
    vi.stubGlobal(
        "fetch",
        vi.fn(async (url, options) => {
            calls.push([url, options?.body]);
            if (url.endsWith("models"))
                return {
                    ok: true,
                    json: async () => ({
                        models: ["fast", "accurate"].map((mode) => ({
                            adapter_id: `native-${mode}`,
                            mode,
                            model_id: `DeepLoc-2.1-${mode === "fast" ? "Fast" : "Accurate"}`,
                            readiness: "configured",
                        })),
                    }),
                };
            const body = JSON.parse(options.body),
                mode = body.adapter_id.replace("native-", "");
            return {
                ok: true,
                json: async () => ({
                    model: {
                        model_id: body.expected_model_id,
                        checkpoint_sha256: mode.repeat(64).slice(0, 64),
                        classes: ["Nucleus", "Extracellular"],
                        provenance: {mode},
                    },
                    results: body.proteins.map((protein) => ({
                        protein,
                        probabilities: [0.7, 0.2],
                        decisions: ["Nucleus"],
                    })),
                }),
            };
        }),
    );
    render(<NativeWorkflow onBack={() => {
    }}/>);
    await screen.findByRole("option", {name: /Fast —/});
    await user.type(
        screen.getByLabelText("Protein sequences (FASTA)"),
        ">p\nACDEFGHIKLMNPQ",
    );
    await user.click(
        screen.getByRole("button", {name: "Predict with selected mode"}),
    );
    await screen.findByRole("heading", {name: "DeepLoc-2.1-Fast"});
    await user.selectOptions(
        screen.getByLabelText("DeepLoc model mode"),
        "native-accurate",
    );
    expect(
        screen.queryByRole("heading", {name: "DeepLoc-2.1-Fast"}),
    ).not.toBeInTheDocument();
    await user.click(
        screen.getByRole("button", {name: "Predict with selected mode"}),
    );
    await screen.findByRole("heading", {name: "DeepLoc-2.1-Accurate"});
    expect(JSON.parse(calls.at(-1)[1]).adapter_id).toBe("native-accurate");
    expect(screen.getByText(/--model-mode accurate/)).toBeInTheDocument();
});
it("preserves original report bytes and rejects mismatched comparison proteins", () => {
    const text = JSON.stringify(fixture) + "\n\n";
    expect(exportReportText(readReportText(text))).toBe(text);
    const other = structuredClone(fixture);
    other.results[0].sequence_sha256 = "0".repeat(64);
    const comparison = compareReports(fixture, other);
    expect(comparison.unmatched_left).toHaveLength(1);
    expect(comparison.unmatched_right).toHaveLength(1);
    other.results[1].explainer.reference_policy = "different";
    expect(
        compareReports(fixture, other).results[0].attribution_comparability,
    ).toBe("confounded_by_methodology");
});

it("validates schema-v3 equal-group values against complete local data", () => {
    expect(validateReport(modern)).toBe(modern);
    const corrupt = structuredClone(modern);
    corrupt.global_statistics[0].mean += 0.1;
    expect(() => validateReport(corrupt)).toThrow(/equal-group statistics/);
});
