import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, expect, it, vi } from "vitest";
import App from "./App";
import { request } from "./api";

vi.mock("./api", () => ({ request: vi.fn() }));
const prediction = {
  model_classes: ["Nucleus", "Cytoplasm"],
  results: [
    {
      index: 0,
      sequence_length: 3,
      prediction: "Nucleus",
      confidence: 0.6,
      probabilities: [
        { label: "Nucleus", probability: 0.6 },
        { label: "Cytoplasm", probability: 0.4 },
      ],
    },
  ],
};

function respond(path, options) {
  if (path === "/model") return Promise.resolve({ metadata_available: false });
  if (path === "/features")
    return Promise.resolve({
      results: options.body.sequences.map((s, index) => ({
        index,
        sequence_length: s.length,
        composition: { A: 1, C: 0 },
        gravy: 1.8,
        isoelectric_point: 5.5,
      })),
    });
  if (path === "/predict")
    return Promise.resolve({
      ...prediction,
      results: options.body.sequences.map((s, index) => ({
        ...prediction.results[0],
        index,
        sequence_length: s.length,
      })),
    });
  return Promise.resolve({
    results: [
      {
        base_value: 0.5,
        feature_contributions: [
          { feature: "aa_A", value: 1, contribution: 0.1 },
        ],
      },
    ],
  });
}

beforeEach(() => {
  vi.clearAllMocks();
  request.mockImplementation(respond);
});

it("keeps live external model controls outside the public application", () => {
  render(<App />);
  expect(
    screen.queryByRole("button", { name: /DeepLoc|Accurate/ }),
  ).not.toBeInTheDocument();
  expect(
    screen.getByText(/Live predictions use MAP-ExPLoc-owned models only/),
  ).toBeInTheDocument();
  expect(
    screen.getByRole("button", { name: "Import report" }),
  ).toBeInTheDocument();
  expect(request).not.toHaveBeenCalled();
});

async function analyze() {
  const user = userEvent.setup();
  await user.type(screen.getByLabelText("Protein sequences"), "AAA");
  await user.click(screen.getByRole("button", { name: "Analyze sequences" }));
  await screen.findByRole("heading", { name: "Your analysis" });
  return user;
}

it("submits only explicitly, separates screens and preserves input on return", async () => {
  render(<App />);
  expect(request).not.toHaveBeenCalled();
  const user = await analyze();
  expect(screen.queryByRole("textbox")).not.toBeInTheDocument();
  expect(request.mock.calls.some(([path]) => path === "/explain")).toBe(false);
  await user.click(screen.getByRole("button", { name: /Back to sequences/ }));
  expect(screen.getByRole("textbox")).toHaveValue("AAA");
});
it("loads explanations on demand, caches them and supports keyboard tabs", async () => {
  render(<App />);
  const user = await analyze();
  await user.click(screen.getByRole("tab", { name: "Explanation" }));
  await screen.findByText("What influenced Nucleus?");
  await user.click(screen.getByRole("tab", { name: "Prediction" }));
  await user.click(screen.getByRole("tab", { name: "Explanation" }));
  expect(
    request.mock.calls.filter(([path]) => path === "/explain"),
  ).toHaveLength(1);
  fireEvent.keyDown(screen.getByRole("tab", { name: "Explanation" }), {
    key: "ArrowLeft",
  });
  expect(screen.getByRole("tab", { name: "Sequence" })).toHaveFocus();
  expect(screen.getByText("Amino-acid composition")).toBeVisible();
});
it("keeps predictions available when SHAP fails and allows retry", async () => {
  request.mockImplementation((path, options) =>
    path === "/explain"
      ? Promise.reject(new Error("Explanation unavailable"))
      : respond(path, options),
  );
  render(<App />);
  const user = await analyze();
  await user.click(screen.getByRole("tab", { name: "Explanation" }));
  expect(await screen.findByRole("alert")).toHaveTextContent(
    "Explanation unavailable",
  );
  request.mockImplementation(respond);
  await user.click(screen.getByRole("button", { name: "Retry explanation" }));
  await screen.findByText("What influenced Nucleus?");
  expect(screen.getByText(/Predicted compartment/)).toHaveTextContent(
    "Nucleus",
  );
});
it("does not replace current analysis with an obsolete response after cancellation", async () => {
  let release;
  request.mockImplementation((path, options) =>
    path === "/predict"
      ? new Promise((resolve) => {
          release = resolve;
        })
      : respond(path, options),
  );
  render(<App />);
  const user = userEvent.setup();
  await user.type(screen.getByRole("textbox"), "AAA");
  await user.click(screen.getByRole("button", { name: "Analyze sequences" }));
  await user.click(screen.getByRole("button", { name: "Cancel" }));
  await act(async () => release(prediction));
  expect(screen.getByRole("textbox")).toBeVisible();
  expect(screen.queryByText("Your analysis")).not.toBeInTheDocument();
});
it("compares and selects batch records without explaining every sequence", async () => {
  render(<App />);
  const user = userEvent.setup();
  await user.click(screen.getByRole("button", { name: "Batch FASTA" }));
  await user.type(screen.getByRole("textbox"), ">one\nAAA\n>two\nCCCC");
  await user.click(screen.getByRole("button", { name: "Analyze sequences" }));
  await screen.findByText("Your analysis");
  await user.click(screen.getByText("Compare all proteins"));
  await user.click(screen.getByRole("button", { name: "two", exact: true }));
  expect(screen.getByLabelText("Selected protein")).toHaveValue("1");
  await user.click(screen.getByRole("tab", { name: "Sequence" }));
  expect(within(screen.getByRole("tabpanel")).getByText("4 aa")).toBeVisible();
  expect(request.mock.calls.some(([path]) => path === "/explain")).toBe(false);
});
it("shows model errors and does not discard the input", async () => {
  request.mockRejectedValue(new Error("The model is unavailable."));
  render(<App />);
  const user = userEvent.setup();
  await user.type(screen.getByRole("textbox"), "AAA");
  await user.click(screen.getByRole("button", { name: "Analyze sequences" }));
  expect(await screen.findByRole("alert")).toHaveTextContent(
    "model is unavailable",
  );
  expect(screen.getByRole("textbox")).toHaveValue("AAA");
});
it("ignores a stale SHAP result after switching proteins", async () => {
  let finish;
  request.mockImplementation((path, options) =>
    path === "/explain" && options.body.sequences[0] === "AAA"
      ? new Promise((resolve) => {
          finish = resolve;
        })
      : respond(path, options),
  );
  render(<App />);
  const user = userEvent.setup();
  await user.click(screen.getByRole("button", { name: "Batch FASTA" }));
  await user.type(screen.getByRole("textbox"), ">one\nAAA\n>two\nCCC");
  await user.click(screen.getByRole("button", { name: "Analyze sequences" }));
  await screen.findByText("Your analysis");
  await user.click(screen.getByRole("tab", { name: "Explanation" }));
  await user.selectOptions(screen.getByLabelText("Selected protein"), "1");
  await screen.findByText("What influenced Nucleus?");
  await act(async () =>
    finish({ results: [{ base_value: 0.999, feature_contributions: [] }] }),
  );
  await waitFor(() =>
    expect(screen.queryByText(/0.9990/)).not.toBeInTheDocument(),
  );
});

it.each([
  ["residues", ["A".repeat(10_001), "A".repeat(10_000)], false],
  ["proteins", Array(21).fill("AAA"), false],
  ["exact boundary", Array(20).fill("A".repeat(1000)), true],
])("checks complete report limits for %s", async (_, sequences, allowed) => {
  render(<App />);
  const user = userEvent.setup();
  await user.click(screen.getByRole("button", { name: "Batch FASTA" }));
  fireEvent.change(screen.getByRole("textbox"), {
    target: {
      value: sequences.map((sequence, i) => `>p${i}\n${sequence}`).join("\n"),
    },
  });
  await user.click(screen.getByRole("button", { name: "Analyze sequences" }));
  await screen.findByText("Your analysis");
  await user.click(
    screen.getByRole("button", { name: "Generate complete report" }),
  );
  expect(request.mock.calls.some(([path]) => path === "/v2/analyze")).toBe(
    allowed,
  );
  if (!allowed)
    expect(screen.getByRole("alert")).toHaveTextContent(
      "Complete reports support at most 20 sequences and 20,000 total residues",
    );
});
