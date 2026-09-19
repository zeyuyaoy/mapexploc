import { useState } from "react";
import { download } from "../analysis";
import { exportReportText, reportAttributionCsv } from "../reports";
import ReportComparison from "./ReportComparison";

export default function ReportViewer({ report }) {
  const [selected, setSelected] = useState(
    report.results[0].protein.protein_id,
  );
  const [classId, setClassId] = useState(report.model.classes[0]);
  const [compartment, setCompartment] = useState("");
  const [descending, setDescending] = useState(false);
  const [view, setView] = useState("Protein");
  const visible = report.results
    .filter((r) => !compartment || r.decisions.includes(compartment))
    .toSorted(
      (a, b) =>
        (descending ? -1 : 1) *
        a.protein.protein_id.localeCompare(b.protein.protein_id),
    );
  const local =
    visible.find((r) => r.protein.protein_id === selected) || visible[0];
  const ci = report.model.classes.indexOf(classId);
  const regional = local?.features[0].kind === "sequence_region";
  const ranked =
    local?.features
      .map((f, i) => ({
        ...f,
        value: local.attributions[ci][i],
        measured: local.feature_values[i],
      }))
      .toSorted((a, b) => Math.abs(b.value) - Math.abs(a.value)) || [];
  const maximum = Math.max(...ranked.map((f) => Math.abs(f.value)), 0.00001);
  const global =
    report.cohort?.contributions
      .filter((r) => r.class_id === classId)
      .toSorted((a, b) => b.mean_absolute - a.mean_absolute) || [];
  return (
    <div className="report-viewer">
      <div className="result-heading">
        <div>
          <p className="eyebrow">
            Complete report · schema v{report.schema_version}
          </p>
          <h1>Explanation report</h1>
          <p>
            {report.model.model_id} · {report.results.length} proteins ·{" "}
            {report.model.task_type}
          </p>
        </div>
        <div className="report-actions">
          <button
            className="secondary-button"
            onClick={() =>
              download(
                exportReportText(report),
                "mapexploc-report.json",
                "application/json",
              )
            }
          >
            Export complete JSON
          </button>
          <button
            className="secondary-button"
            onClick={() =>
              download(
                reportAttributionCsv(report),
                "mapexploc-attributions.csv",
                "text/csv",
              )
            }
          >
            Export all attributions
          </button>
        </div>
      </div>
      <p className="science-note">
        Biological overlap describes model behavior, not causal proof. Region
        contributions apply to entire intervals; engineered descriptors apply to
        the whole protein.
      </p>
      {report.structured_warnings?.map((w) => (
        <p className="science-note" role="status" key={`${w.scope}:${w.code}`}>
          {w.message}
        </p>
      ))}
      {report.runtime && (
        <p>
          {report.runtime.model_mode || "External model"} ·{" "}
          {report.runtime.embedding_model || "Native feature representation"} ·{" "}
          {report.runtime.device} · {report.runtime.precision}
        </p>
      )}
      <div className="report-controls">
        <label>
          View
          <select
            aria-label="View"
            value={view}
            onChange={(e) => setView(e.target.value)}
          >
            <option>Protein</option>
            <option disabled={!report.cohort}>Global cohort</option>
          </select>
        </label>
        <label>
          Explained class
          <select
            aria-label="Explained class"
            value={classId}
            onChange={(e) => setClassId(e.target.value)}
          >
            {report.model.classes.map((c) => (
              <option key={c}>{c}</option>
            ))}
          </select>
        </label>
        {view === "Protein" && (
          <>
            <label>
              Localization filter
              <select
                aria-label="Localization filter"
                value={compartment}
                onChange={(e) => setCompartment(e.target.value)}
              >
                <option value="">All compartments</option>
                {report.model.classes.map((c) => (
                  <option key={c}>{c}</option>
                ))}
              </select>
            </label>
            <button
              className="secondary-button"
              aria-pressed={descending}
              onClick={() => setDescending(!descending)}
            >
              Protein order: {descending ? "Z–A" : "A–Z"}
            </button>
            <label>
              Protein
              <select
                aria-label="Protein"
                value={local?.protein.protein_id || ""}
                onChange={(e) => setSelected(e.target.value)}
              >
                {visible.map((r) => (
                  <option key={r.protein.protein_id}>
                    {r.protein.protein_id}
                  </option>
                ))}
              </select>
            </label>
          </>
        )}
      </div>
      {view === "Global cohort" ? (
        <section>
          <h2>{report.cohort.cohort_id}</h2>
          <p>{report.cohort.selection_criteria}</p>
          <p>{report.cohort.aggregation}</p>
          <p>
            {report.cohort.included_count} included /{" "}
            {report.cohort.input_count} supplied. This summary always covers the
            full declared cohort.
          </p>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Feature/category</th>
                  <th>Mean absolute</th>
                  <th>Mean signed</th>
                  <th>Proteins</th>
                  <th>Regions/features</th>
                </tr>
              </thead>
              <tbody>
                {global.map((r) => (
                  <tr key={r.feature_id}>
                    <th>{r.feature_id}</th>
                    <td>{r.mean_absolute.toPrecision(5)}</td>
                    <td>{r.mean_signed.toPrecision(5)}</td>
                    <td>{r.protein_count}</td>
                    <td>{r.region_count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {report.global_statistics?.length > 0 && (
            <details>
              <summary>Equal-group statistics and uncertainty</summary>
              <p>
                Exploratory summaries; density is a separate statistic and does
                not replace additive SHAP.
              </p>
              <table>
                <thead>
                  <tr>
                    <th>Feature/category</th>
                    <th>Statistic</th>
                    <th>Group mean</th>
                    <th>95% bootstrap interval</th>
                    <th>Groups</th>
                  </tr>
                </thead>
                <tbody>
                  {report.global_statistics
                    .filter((r) => r.class_id === classId)
                    .map((r) => (
                      <tr key={`${r.feature_id}:${r.statistic}`}>
                        <th>{r.feature_id}</th>
                        <td>{r.statistic}</td>
                        <td>{r.mean.toPrecision(5)}</td>
                        <td>
                          {r.interval_95
                            .map((v) => v.toPrecision(5))
                            .join(" – ")}
                        </td>
                        <td>
                          {r.group_count}
                          {r.group_count < 20 ? " · small sample" : ""}
                        </td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </details>
          )}
        </section>
      ) : local ? (
        <section key={local.protein.protein_id}>
          <h2>{local.protein.protein_id}</h2>
          <p>
            {local.protein.sequence.length} residues · Native decisions:{" "}
            {local.decisions.join(", ") || "None above threshold"}
          </p>
          <p>
            {classId}: {local.probabilities[ci].toPrecision(6)} · Baseline:{" "}
            {local.base_values[ci].toPrecision(6)} · Reconstruction residual:{" "}
            {local.residuals[ci].toExponential(2)}
          </p>
          <p className="science-note">
            Method: {local.explainer.method}. Approximation:{" "}
            {local.explainer.stability?.status || "not assessed"}. Reference:{" "}
            {local.explainer.reference_policy}.
          </p>
          {report.diagnostics
            ?.filter((d) => d.protein_id === local.protein.protein_id)
            .map((d) => (
              <section key={d.protein_id}>
                <h3>Attribution stability</h3>
                {d.stability.length ? (
                  d.stability
                    .filter((s) => s.class_id === classId)
                    .map((s) => (
                      <p className="science-note" key={s.comparison}>
                        {s.comparison}: <strong>{s.status}</strong> · maximum
                        change {s.max_absolute_delta.toPrecision(4)} · rank
                        correlation{" "}
                        {s.magnitude_spearman?.toFixed(3) ?? "undefined"} · sign
                        agreement {s.sign_agreement?.toFixed(3) ?? "undefined"}
                      </p>
                    ))
                ) : (
                  <p>Sensitivity checks were not requested.</p>
                )}
                <details>
                  <summary>
                    Faithfulness, reference shift and supervision overlap
                  </summary>
                  <pre>
                    {JSON.stringify(
                      {
                        faithfulness: d.faithfulness.filter(
                          (f) => f.class_id === classId,
                        ),
                        distribution_shift: d.distribution_shift,
                        supervision_overlap: d.supervision_overlap,
                      },
                      null,
                      2,
                    )}
                  </pre>
                </details>
              </section>
            ))}
          {regional && (
            <>
              <h3>Sequence regions and biological annotations</h3>
              <p>
                Positions use one-based inclusive coordinates. Each annotation
                has its own track; overlaps are retained.
              </p>
              <div
                className="region-track"
                aria-label="Region attribution track"
              >
                {local.features.map((f, i) => (
                  <div
                    key={f.feature_id}
                    title={`${f.start + 1}–${f.end}: ${local.attributions[ci][i]}`}
                    style={{
                      width: `${((f.end - f.start) / local.protein.sequence.length) * 100}%`,
                      background:
                        local.attributions[ci][i] >= 0 ? "#22785d" : "#ad5265",
                      opacity:
                        0.2 +
                        (0.8 * Math.abs(local.attributions[ci][i])) / maximum,
                    }}
                  >
                    <span>{f.start + 1}</span>
                  </div>
                ))}
              </div>
            </>
          )}
          {regional &&
            local.annotations.map((a) => (
              <div className="annotation-record" key={a.annotation_id}>
                <p>
                  <strong>{a.kind}</strong> · {a.start + 1}–{a.end}
                  {a.uncertain
                    ? " · uncertain boundaries (excluded from exact-coordinate statistics)"
                    : ""}
                </p>
                <div className="annotation-track">
                  <span
                    style={{
                      marginLeft: `${(a.start / local.protein.sequence.length) * 100}%`,
                      width: `${((a.end - a.start) / local.protein.sequence.length) * 100}%`,
                    }}
                  />
                </div>
                <details>
                  <summary>Annotation provenance</summary>
                  <p>
                    {a.accession}
                    {a.isoform ? ` / ${a.isoform}` : ""} · {a.source}{" "}
                    {a.source_version} · retrieved {a.retrieved_at}
                  </p>
                  <p>{a.source_url}</p>
                  <pre>{JSON.stringify(a.evidence, null, 2)}</pre>
                </details>
              </div>
            ))}
          {regional && !local.annotations.length && (
            <p>No positional annotations supplied for this exact sequence.</p>
          )}
          <h3>Signed contributions for {classId}</h3>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>
                    {regional
                      ? "Region (inclusive positions)"
                      : "Engineered descriptor"}
                  </th>
                  <th>Contribution</th>
                  <th>Value</th>
                  <th>Definition</th>
                </tr>
              </thead>
              <tbody>
                {ranked.map((f) => (
                  <tr key={f.feature_id}>
                    <th>
                      {regional ? `${f.start + 1}–${f.end}` : f.feature_id}
                    </th>
                    <td
                      className={
                        f.value >= 0 ? "positive-text" : "negative-text"
                      }
                    >
                      {f.value >= 0 ? "+" : ""}
                      {f.value.toPrecision(6)}
                    </td>
                    <td>{f.measured}</td>
                    <td>{f.definition}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <details>
            <summary>Sequence and explanation configuration</summary>
            <pre>{local.protein.sequence}</pre>
            <pre>{JSON.stringify(local.explainer, null, 2)}</pre>
          </details>
          {local.warnings.map((w) => (
            <p className="science-note" key={w}>
              {w}
            </p>
          ))}
        </section>
      ) : (
        <p>No proteins have a native prediction in this compartment.</p>
      )}
      <ReportComparison report={report} />
      <details>
        <summary>Model and reproducibility metadata</summary>
        <pre>
          {JSON.stringify(
            {
              model: report.model,
              configuration: report.configuration,
              software: report.software,
              warnings: report.warnings,
            },
            null,
            2,
          )}
        </pre>
      </details>
    </div>
  );
}
