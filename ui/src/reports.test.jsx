import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import ReportViewer from "./components/ReportViewer";
import fixture from "./test-fixtures/report-v2.json";
import { reportAttributionCsv, validateReport } from "./reports";

describe("portable report contract", () => {
  it("accepts complete data and exports every class and protein", () => {
    expect(validateReport(fixture)).toBe(fixture);
    expect(reportAttributionCsv(fixture).split("\r\n")).toHaveLength(41);
  });
  it("preserves nearest-even rounding in native multilabel fallback decisions", () => {
    const report = structuredClone(fixture);
    Object.assign(report.model, {
      task_type: "multilabel",
      thresholds: [0.5, 0.60004],
      threshold_comparison: "greater",
      empty_decision_policy: "nearest_threshold",
      fallback_round_decimals: 4,
    });
    for (const local of report.results) {
      local.probabilities = [0.12345, 0.22346];
      local.base_values = [...local.probabilities];
      local.attributions = local.attributions.map((row) => row.map(() => 0));
      local.residuals = [0, 0];
      local.decisions = [report.model.classes[1]];
    }
    for (const row of report.cohort.contributions) {
      row.mean_absolute = 0;
      row.mean_signed = 0;
    }
    expect(validateReport(report)).toBe(report);
  });
  it("rejects tensor, class, probability and annotation misalignment", () => {
    for (const corrupt of [
      (r) => {
        r.results[0].attributions[0].pop();
      },
      (r) => {
        r.results[0].explained_classes.reverse();
      },
      (r) => {
        r.results[0].probabilities[0] = 3;
      },
      (r) => {
        r.results[0].annotations[0].start = -1;
      },
      (r) => {
        r.results[1].protein.protein_id = r.results[0].protein.protein_id;
      },
      (r) => {
        r.results[0].decisions = ["unknown-compartment"];
      },
      (r) => {
        r.cohort.contributions[0].mean_signed += 0.1;
      },
    ]) {
      const report = structuredClone(fixture);
      corrupt(report);
      expect(() => validateReport(report)).toThrow(
        /Invalid explanation report/,
      );
    }
  });
  it("preserves protein identity across sorting/filtering and changes class", async () => {
    const user = userEvent.setup();
    render(<ReportViewer report={fixture} />);
    await user.selectOptions(
      screen.getByLabelText("Protein"),
      "outside-control",
    );
    await user.click(
      screen.getByRole("button", { name: "Protein order: A–Z" }),
    );
    expect(
      screen.getByRole("heading", { name: "outside-control" }),
    ).toBeInTheDocument();
    await user.selectOptions(
      screen.getByLabelText("Explained class"),
      "inside",
    );
    expect(
      screen.getByRole("heading", { name: "Signed contributions for inside" }),
    ).toBeInTheDocument();
    await user.click(screen.getByText("Annotation provenance"));
    expect(screen.getByText(/Synthetic UI fixture 1/)).toBeVisible();
    await user.selectOptions(
      screen.getByLabelText("Localization filter"),
      "inside",
    );
    expect(
      screen.getByRole("heading", { name: "inside-control" }),
    ).toBeInTheDocument();
    await user.selectOptions(screen.getByLabelText("View"), "Global cohort");
    expect(
      screen.getByRole("heading", { name: "synthetic-ui-test" }),
    ).toBeInTheDocument();
    expect(screen.getByText(/2 included \/ 2 supplied/)).toBeInTheDocument();
  });
});
