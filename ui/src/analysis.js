export const LIMITS = { records: 100, sequence: 100_000, total: 1_000_000 };
const AMINO_ACIDS = /^[ACDEFGHIKLMNPQRSTVWY]+$/;

export function parseSequences(text, mode = "single") {
  const raw = text.trim();
  if (!raw) throw new Error("Enter a protein sequence or upload a FASTA file.");
  let records;
  if (raw.startsWith(">")) {
    records = [];
    for (const line of raw.split(/\r?\n/)) {
      if (line.startsWith(">")) {
        const name = line.slice(1).trim();
        if (!name)
          throw new Error(
            `Record ${records.length + 1}: add a FASTA identifier after >.`,
          );
        records.push({ id: name.split(/\s/)[0], name, sequence: "" });
      } else if (line.trim()) records.at(-1).sequence += line;
    }
  } else {
    if (mode === "batch")
      throw new Error(
        "Batch input needs FASTA headers, starting with >protein-id.",
      );
    records = [{ id: "sequence-1", name: "Your sequence", sequence: raw }];
  }
  if (mode === "single" && records.length !== 1)
    throw new Error(
      "Multiple records found. Switch to Batch FASTA to analyze them together.",
    );
  if (records.length > LIMITS.records)
    throw new Error(`Use at most ${LIMITS.records} sequences per analysis.`);
  const ids = new Set();
  let total = 0;
  records = records.map((record) => {
    if (ids.has(record.id))
      throw new Error(`${record.id}: FASTA identifiers must be unique.`);
    ids.add(record.id);
    const sequence = record.sequence.replace(/\s/g, "").toUpperCase();
    if (!sequence) throw new Error(`${record.id}: the sequence is empty.`);
    if (!AMINO_ACIDS.test(sequence)) {
      const invalid = [
        ...new Set(sequence.replace(/[ACDEFGHIKLMNPQRSTVWY]/g, "")),
      ].join(", ");
      throw new Error(
        `${record.id}: unsupported residues (${invalid}). Use the 20 standard amino-acid letters.`,
      );
    }
    if (sequence.length > LIMITS.sequence)
      throw new Error(`${record.id}: exceeds 100,000 residues.`);
    total += sequence.length;
    return { ...record, sequence };
  });
  if (total > LIMITS.total)
    throw new Error(
      "This batch exceeds 1,000,000 residues. Split it into smaller files.",
    );
  return records;
}

function csvCell(value) {
  let text = String(value);
  // Prevent spreadsheets from interpreting FASTA headers as formulas.
  if (/^[=+@\-\t\r]/.test(text)) text = `'${text}`;
  return `"${text.replaceAll('"', '""')}"`;
}

export function predictionCsv(analysis) {
  const classes = analysis.prediction.model_classes;
  const rows = [
    [
      "protein_id",
      "protein_name",
      "length",
      "prediction",
      "model_probability",
      ...classes.map((label) => `probability_${label}`),
    ],
  ];
  for (const row of analysis.prediction.results) {
    const record = analysis.records[row.index];
    rows.push([
      record.id,
      record.name,
      row.sequence_length,
      row.prediction,
      row.confidence,
      ...classes.map(
        (label) =>
          row.probabilities.find((item) => item.label === label)?.probability ??
          "",
      ),
    ]);
  }
  return rows.map((row) => row.map(csvCell).join(",")).join("\r\n") + "\r\n";
}

export function analysisJson(analysis, explanations) {
  return JSON.stringify(
    {
      schema_version: 1,
      status: "partial_interactive_session",
      explained_protein_count: Object.keys(explanations).length,
      requested_protein_count: analysis.records.length,
      analyzed_at: analysis.createdAt,
      records: analysis.records,
      predictions: analysis.prediction,
      sequence_features: analysis.features,
      model: analysis.model,
      explanations,
      notes: [
        "Probabilities are uncalibrated model outputs.",
        "SHAP explains engineered features, not residue-level causality.",
      ],
    },
    null,
    2,
  );
}

export function download(text, filename, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
