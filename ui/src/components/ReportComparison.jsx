import { useState } from "react";
import { readReportText, verifySequenceHashes } from "../reports";
import { compareReports } from "../comparison";
import { download } from "../analysis";

export default function ReportComparison({ report }) {
  const [comparison, setComparison] = useState(null);
  const [error, setError] = useState("");

  async function load(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      if (file.size > 100_000_000)
        throw new Error("Choose a report smaller than 100 MB.");
      const other = readReportText(await file.text());
      await verifySequenceHashes(other);
      setComparison(compareReports(report, other));
      setError("");
    } catch (e) {
      setError(e.message);
      setComparison(null);
    }
    event.target.value = "";
  }

  return (
    <details>
      <summary>Compare another model report</summary>
      <label>
        Second completed report
        <input type="file" accept=".json" onChange={load} />
      </label>
      {error && <p role="alert">{error}</p>}
      {comparison && (
        <>
          <h3>
            {comparison.left_model} / {comparison.right_model}
          </h3>
          <p>{comparison.interpretation}</p>
          <p>
            {comparison.results.length} matched proteins;{" "}
            {comparison.unmatched_left.length} unmatched in the first report;{" "}
            {comparison.unmatched_right.length} unmatched in the second.
          </p>
          {comparison.results.map((r) => (
            <section key={r.protein_id}>
              <h4>{r.protein_id}</h4>
              <p>
                Decision agreement (Jaccard): {r.decision_jaccard.toFixed(3)} ·{" "}
                {r.attribution_comparability}
              </p>
              <table>
                <thead>
                  <tr>
                    <th>Class</th>
                    <th>Probability difference</th>
                    <th>Attribution rank correlation</th>
                    <th>Sign agreement</th>
                    <th>Top-region coverage overlap</th>
                  </tr>
                </thead>
                <tbody>
                  {r.classes.map((c) => (
                    <tr key={c.class_id}>
                      <th>{c.class_id}</th>
                      <td>{c.probability_difference.toPrecision(4)}</td>
                      <td>
                        {c.magnitude_spearman?.toFixed(3) ??
                          "Not comparable / undefined"}
                      </td>
                      <td>
                        {c.sign_agreement?.toFixed(3) ??
                          "Not comparable / undefined"}
                      </td>
                      <td>
                        {c.top_region_residue_coverage_jaccard?.toFixed(3) ??
                          "Not regional"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <details>
                <summary>Biological-region concordance by model</summary>
                <p>
                  Positive attribution density in fully contained regions only;
                  this descriptive overlap is not causal evidence.
                </p>
                <pre>
                  {JSON.stringify(r.biological_region_concordance, null, 2)}
                </pre>
              </details>
            </section>
          ))}
          <button
            className="secondary-button"
            onClick={() =>
              download(
                JSON.stringify(comparison, null, 2),
                "mapexploc-comparison.json",
                "application/json",
              )
            }
          >
            Export comparison
          </button>
        </>
      )}
    </details>
  );
}
