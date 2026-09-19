import { describe, expect, it } from "vitest";
import {
  analysisJson,
  LIMITS,
  parseSequences,
  predictionCsv,
} from "./analysis";

describe("biological input", () => {
  it("normalizes single sequences and FASTA without changing residues", () => {
    expect(parseSequences("ac de\nFG")[0].sequence).toBe("ACDEFG");
    expect(
      parseSequences(">one first\nACDE\n>two\nMKT", "batch").map((r) => r.id),
    ).toEqual(["one", "two"]);
  });
  it("rejects unsupported residues, duplicate IDs and empty FASTA records", () => {
    expect(() => parseSequences("AX")).toThrow("X");
    expect(() => parseSequences(">one\nAA\n>one second\nCC", "batch")).toThrow(
      "unique",
    );
    expect(() => parseSequences(">one\n>two\nCC", "batch")).toThrow(
      "one: the sequence is empty",
    );
    expect(() => parseSequences(">\nAAA")).toThrow("identifier");
    expect(() => parseSequences("AAAA", "batch")).toThrow("headers");
    expect(() => parseSequences(">a\nAAA\n>b\nCCC")).toThrow("Switch to Batch");
  });
  it("enforces per-record, batch and total limits", () => {
    expect(() => parseSequences("A".repeat(LIMITS.sequence + 1))).toThrow(
      "50,000",
    );
    expect(() =>
      parseSequences(
        Array.from({ length: 26 }, (_, i) => `>p${i}\nAA`).join("\n"),
        "batch",
      ),
    ).toThrow("25 sequences");
    expect(() =>
      parseSequences(
        Array.from(
          { length: 2 },
          (_, i) => `>p${i}\n${"A".repeat(25001)}`,
        ).join("\n"),
        "batch",
      ),
    ).toThrow("50,000");
  });
});
it("exports aligned numeric probabilities and escaped record names", () => {
  const analysis = {
    createdAt: "2026-09-18",
    records: [{ id: "=A1", name: 'a,"b', sequence: "AAA" }],
    prediction: {
      model_classes: ["Nucleus"],
      results: [
        {
          index: 0,
          sequence_length: 3,
          prediction: "Nucleus",
          confidence: 0.8,
          probabilities: [{ label: "Nucleus", probability: 0.8 }],
        },
      ],
    },
    features: null,
    model: null,
  };
  expect(predictionCsv(analysis)).toContain('"\'=A1","a,""b"');
  expect(predictionCsv(analysis)).toContain('"0.8"');
  expect(JSON.parse(analysisJson(analysis, {})).records[0].sequence).toBe(
    "AAA",
  );
});
