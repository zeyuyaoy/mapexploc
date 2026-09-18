import { useEffect, useMemo, useRef, useState } from "react";
import FeatureContributions from "./components/FeatureContributions";
import { ProbabilityChart, SequenceChart } from "./components/Charts";
import {
  analysisJson,
  download,
  parseSequences,
  predictionCsv,
} from "./analysis";
import { request } from "./api";
import examples from "./examples.json";

const TABS = ["Prediction", "Sequence", "Explanation"];
const DOCS = "https://github.com/zeyuyaoy/mapexploc/blob/main/docs/index.md";

export default function App() {
  const [input, setInput] = useState("");
  const [mode, setMode] = useState("single");
  const [step, setStep] = useState("entry");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [analysis, setAnalysis] = useState(null);
  const [selected, setSelected] = useState(0);
  const [tab, setTab] = useState("Prediction");
  const [explanations, setExplanations] = useState({});
  const [detailError, setDetailError] = useState("");
  const [detailLoading, setDetailLoading] = useState(false);
  const [explanationFailure, setExplanationFailure] = useState(null);
  const [retry, setRetry] = useState(0);
  const [sort, setSort] = useState({ key: "index", direction: 1 });
  const activeRequest = useRef(null);
  const generation = useRef(0);
  const fileInput = useRef(null);
  const title = useRef(null);
  const tabs = useRef([]);
  const preview = useMemo(() => {
    try {
      return { records: parseSequences(input, mode) };
    } catch (issue) {
      return { error: issue.message };
    }
  }, [input, mode]);

  useEffect(() => () => activeRequest.current?.abort(), []);
  useEffect(() => {
    title.current?.focus({ preventScroll: true });
    window.scrollTo(0, 0);
  }, [step]);

  useEffect(() => {
    if (
      step !== "results" ||
      tab !== "Explanation" ||
      !analysis ||
      explanations[selected]
    )
      return;
    const controller = new AbortController();
    request("/explain", {
      body: { sequences: [analysis.records[selected].sequence], top_n: 12 },
      signal: controller.signal,
    })
      .then((payload) => {
        if (!controller.signal.aborted) {
          setExplanationFailure(null);
          setExplanations((previous) => ({
            ...previous,
            [selected]: payload.results[0],
          }));
        }
      })
      .catch((issue) => {
        if (!controller.signal.aborted)
          setExplanationFailure({
            analysis,
            selected,
            retry,
            message: issue.message,
          });
      });
    return () => controller.abort();
  }, [analysis, selected, tab, step, retry, explanations]);

  function cancel() {
    generation.current += 1;
    activeRequest.current?.abort();
    setLoading(false);
  }

  function changeInput(value) {
    cancel();
    setInput(value);
    setError("");
  }

  async function upload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    cancel();
    if (file.size > 4_000_000) {
      setError("Choose a text FASTA file smaller than 4 MB.");
      return;
    }
    const version = generation.current;
    try {
      const text = await file.text();
      if (version !== generation.current) return;
      setMode("batch");
      setInput(text);
      setError("");
    } catch {
      setError(
        "This file could not be read. Try pasting the FASTA text instead.",
      );
    }
    event.target.value = "";
  }

  function loadExample() {
    const items = mode === "single" ? examples.slice(0, 1) : examples;
    changeInput(
      items
        .map((item) => `>${item.id} ${item.name}\n${item.sequence}`)
        .join("\n\n"),
    );
  }

  async function analyze(event) {
    event.preventDefault();
    cancel();
    setError("");
    let records;
    try {
      records = parseSequences(input, mode);
    } catch (issue) {
      setError(issue.message);
      return;
    }
    const controller = new AbortController();
    activeRequest.current = controller;
    const version = generation.current;
    setLoading(true);
    try {
      const body = { sequences: records.map((record) => record.sequence) };
      const [prediction, features, model] = await Promise.all([
        request("/predict", { body, signal: controller.signal }),
        request("/features", { body, signal: controller.signal }).catch(
          (issue) => ({ error: issue.message }),
        ),
        request("/model", { signal: controller.signal }).catch(() => null),
      ]);
      if (controller.signal.aborted || version !== generation.current) return;
      if (
        prediction.results?.length !== records.length ||
        !Array.isArray(prediction.model_classes)
      )
        throw new Error(
          "The service returned an incomplete prediction. Please retry.",
        );
      setAnalysis({
        records,
        prediction,
        features,
        model,
        createdAt: new Date().toISOString(),
      });
      setExplanations({});
      setExplanationFailure(null);
      setDetailError("");
      setDetailLoading(false);
      setSelected(0);
      setTab("Prediction");
      setStep("results");
    } catch (issue) {
      if (!controller.signal.aborted && version === generation.current)
        setError(issue.message);
    } finally {
      if (version === generation.current) setLoading(false);
    }
  }

  async function retryFeatures() {
    if (!analysis) return;
    const snapshot = analysis;
    setDetailLoading(true);
    setDetailError("");
    const controller = new AbortController();
    activeRequest.current?.abort();
    activeRequest.current = controller;
    try {
      const features = await request("/features", {
        body: { sequences: snapshot.records.map((record) => record.sequence) },
        signal: controller.signal,
      });
      if (!controller.signal.aborted)
        setAnalysis((current) =>
          current === snapshot ? { ...current, features } : current,
        );
    } catch (issue) {
      if (!controller.signal.aborted) setDetailError(issue.message);
    } finally {
      if (!controller.signal.aborted) setDetailLoading(false);
    }
  }

  const result = analysis?.prediction.results[selected];
  const record = analysis?.records[selected];
  const explanation = explanations[selected];
  const explanationError =
    explanationFailure?.analysis === analysis &&
    explanationFailure.selected === selected &&
    explanationFailure.retry === retry
      ? explanationFailure.message
      : "";
  const explanationLoading =
    tab === "Explanation" && !explanation && !explanationError;
  const rows = analysis
    ? [...analysis.prediction.results].sort((a, b) => {
        const first =
          sort.key === "name" ? analysis.records[a.index].name : a[sort.key];
        const second =
          sort.key === "name" ? analysis.records[b.index].name : b[sort.key];
        return (
          (typeof first === "string"
            ? first.localeCompare(second)
            : first - second) * sort.direction
        );
      })
    : [];

  function sortBy(key) {
    setSort((current) => ({
      key,
      direction: current.key === key ? -current.direction : 1,
    }));
  }

  function tabKey(event, index) {
    let next;
    if (event.key === "ArrowRight") next = (index + 1) % TABS.length;
    if (event.key === "ArrowLeft")
      next = (index + TABS.length - 1) % TABS.length;
    if (event.key === "Home") next = 0;
    if (event.key === "End") next = TABS.length - 1;
    if (next !== undefined) {
      event.preventDefault();
      setTab(TABS[next]);
      tabs.current[next]?.focus();
    }
  }

  return (
    <div className="app-shell">
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <header className="site-header">
        <a
          className="brand"
          href="#main"
          onClick={() => {
            cancel();
            setStep("entry");
          }}
        >
          MAP-<span>ExPLoc</span>
        </a>
        <nav aria-label="Main navigation">
          {step === "results" && (
            <button
              className="text-button"
              onClick={() => {
                cancel();
                setStep("entry");
              }}
            >
              ← Back to sequences
            </button>
          )}
          <a href={DOCS} target="_blank" rel="noreferrer">
            Documentation <span className="sr-only">(opens in a new tab)</span>
          </a>
        </nav>
      </header>
      <main
        id="main"
        className={step === "entry" ? "entry-main" : "results-main"}
      >
        {step === "entry" ? (
          <>
            <div className="intro">
              <h1 ref={title} tabIndex="-1">
                Explore protein localization
              </h1>
              <p>Start with a sequence. Explore the evidence.</p>
            </div>
            <form onSubmit={analyze} noValidate>
              <div
                className="input-mode"
                role="group"
                aria-label="Input format"
              >
                {["single", "batch"].map((value) => (
                  <button
                    key={value}
                    type="button"
                    aria-pressed={mode === value}
                    onClick={() => {
                      cancel();
                      setMode(value);
                      setError("");
                    }}
                  >
                    {value === "single" ? "Single sequence" : "Batch FASTA"}
                  </button>
                ))}
              </div>
              <label className="sr-only" htmlFor="sequence-input">
                Protein sequences
              </label>
              <textarea
                id="sequence-input"
                value={input}
                onChange={(event) => changeInput(event.target.value)}
                placeholder={
                  mode === "single"
                    ? "Paste an amino-acid sequence or FASTA"
                    : ">protein-id\nPaste its amino-acid sequence here"
                }
                spellCheck="false"
                autoCapitalize="characters"
                aria-invalid={Boolean(error)}
                aria-describedby="input-help input-status"
              />
              <div className="upload-actions">
                <button
                  type="button"
                  className="secondary-button"
                  onClick={() => fileInput.current?.click()}
                >
                  <span aria-hidden="true">↥</span> Upload FASTA
                </button>
                <button
                  type="button"
                  className="secondary-button"
                  onClick={loadExample}
                >
                  Try an example
                </button>
                <input
                  className="sr-only"
                  ref={fileInput}
                  type="file"
                  accept=".fa,.faa,.fasta,.txt"
                  onChange={upload}
                  aria-label="Upload FASTA file"
                  tabIndex="-1"
                />
              </div>
              <p id="input-help" className="input-help">
                20 standard amino acids · up to 100 sequences
              </p>
              <div
                id="input-status"
                className="input-status"
                aria-live="polite"
              >
                {input.trim() &&
                  (preview.records
                    ? `${preview.records.length} valid ${preview.records.length === 1 ? "sequence" : "sequences"} · ${preview.records.reduce((sum, item) => sum + item.sequence.length, 0).toLocaleString()} residues`
                    : preview.error)}
              </div>
              {error && (
                <p className="error-state" role="alert">
                  {error}
                </p>
              )}
              <button
                className="primary-button analyze-button"
                type="submit"
                disabled={loading}
              >
                {loading ? "Analyzing sequences…" : "Analyze sequences"}
              </button>
              {loading && (
                <div className="loading-message">
                  <span role="status">
                    Computing sequence features and predictions…
                  </span>
                  <button
                    type="button"
                    className="text-button"
                    onClick={cancel}
                  >
                    Cancel
                  </button>
                </div>
              )}
              <p className="privacy-note">
                Sequences are sent only when you analyze.
              </p>
            </form>
          </>
        ) : (
          analysis && (
            <>
              <div className="results-heading">
                <h1 ref={title} tabIndex="-1">
                  Your analysis
                </h1>
                <details className="export-menu">
                  <summary>
                    Export <span aria-hidden="true">⌄</span>
                  </summary>
                  <div>
                    <button
                      onClick={() =>
                        download(
                          predictionCsv(analysis),
                          "mapexploc-predictions.csv",
                          "text/csv;charset=utf-8",
                        )
                      }
                    >
                      Predictions CSV
                    </button>
                    <button
                      onClick={() =>
                        download(
                          analysisJson(analysis, explanations),
                          "mapexploc-analysis.json",
                          "application/json",
                        )
                      }
                    >
                      Analysis JSON
                    </button>
                  </div>
                </details>
              </div>
              <label className="sr-only" htmlFor="protein-select">
                Selected protein
              </label>
              <select
                id="protein-select"
                value={selected}
                onChange={(event) => setSelected(Number(event.target.value))}
              >
                {analysis.records.map((item, index) => (
                  <option key={item.id} value={index}>
                    {item.name} · {index + 1} of {analysis.records.length}
                  </option>
                ))}
              </select>
              <p className="prediction-summary">
                Predicted compartment: <strong>{result.prediction}</strong>
              </p>
              <div className="tabs" role="tablist" aria-label="Analysis view">
                {TABS.map((name, index) => (
                  <button
                    key={name}
                    ref={(element) => {
                      tabs.current[index] = element;
                    }}
                    role="tab"
                    id={`tab-${name}`}
                    aria-selected={tab === name}
                    aria-controls="analysis-panel"
                    tabIndex={tab === name ? 0 : -1}
                    onKeyDown={(event) => tabKey(event, index)}
                    onClick={() => setTab(name)}
                  >
                    {name}
                  </button>
                ))}
              </div>
              <section
                id="analysis-panel"
                role="tabpanel"
                aria-labelledby={`tab-${tab}`}
                tabIndex="0"
                className="detail-panel"
              >
                {tab === "Prediction" && (
                  <>
                    <ProbabilityChart probabilities={result.probabilities} />
                    <p className="science-note">
                      ⓘ Model probabilities are not experimental certainty.
                    </p>
                  </>
                )}
                {tab === "Sequence" &&
                  (analysis.features?.results?.[selected] ? (
                    <SequenceChart
                      features={analysis.features.results[selected]}
                      record={record}
                    />
                  ) : (
                    <div>
                      <p className="error-state" role="alert">
                        {detailError ||
                          analysis.features?.error ||
                          "Sequence descriptors could not be loaded."}
                      </p>
                      <button
                        className="secondary-button"
                        disabled={detailLoading}
                        onClick={retryFeatures}
                      >
                        Retry sequence descriptors
                      </button>
                    </div>
                  ))}
                {tab === "Explanation" && (
                  <>
                    {explanationLoading && (
                      <p className="loading-message" role="status">
                        Calculating feature contributions…
                      </p>
                    )}
                    {explanationError && (
                      <>
                        <p className="error-state" role="alert">
                          {explanationError}
                        </p>
                        <button
                          className="secondary-button"
                          onClick={() => setRetry((value) => value + 1)}
                        >
                          Retry explanation
                        </button>
                      </>
                    )}
                    {explanation && (
                      <FeatureContributions
                        prediction={result.prediction}
                        contributions={explanation.feature_contributions}
                        baseValue={explanation.base_value}
                      />
                    )}
                  </>
                )}
              </section>
              {analysis.records.length > 1 && (
                <details className="compare-disclosure">
                  <summary>
                    Compare all proteins{" "}
                    <span>{analysis.records.length} sequences</span>
                  </summary>
                  <div className="table-scroll">
                    <table>
                      <caption className="sr-only">
                        Batch predictions. Select a protein to inspect it.
                      </caption>
                      <thead>
                        <tr>
                          {[
                            ["name", "Protein"],
                            ["sequence_length", "Length"],
                            ["prediction", "Prediction"],
                            ["confidence", "Model probability"],
                          ].map(([key, label]) => (
                            <th
                              key={key}
                              aria-sort={
                                sort.key === key
                                  ? sort.direction === 1
                                    ? "ascending"
                                    : "descending"
                                  : "none"
                              }
                            >
                              <button onClick={() => sortBy(key)}>
                                {label} ↕
                              </button>
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {rows.map((row) => (
                          <tr
                            key={row.index}
                            className={
                              selected === row.index ? "selected-row" : ""
                            }
                          >
                            <th>
                              <button
                                onClick={() => {
                                  setSelected(row.index);
                                  document
                                    .getElementById("protein-select")
                                    ?.focus();
                                }}
                              >
                                {analysis.records[row.index].name}
                              </button>
                            </th>
                            <td>{row.sequence_length} aa</td>
                            <td>{row.prediction}</td>
                            <td>{(row.confidence * 100).toFixed(1)}%</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </details>
              )}
              <details className="model-disclosure">
                <summary>Model & methods</summary>
                <p>
                  Model probabilities are uncalibrated. Predictions are limited
                  to the configured model’s classes. Feature explanations
                  describe the model, not biological mechanisms.
                </p>
                {analysis.model?.metadata_available ? (
                  <>
                    <h2>
                      {analysis.model.metadata.name || "Configured model"}
                    </h2>
                    <p>{analysis.model.metadata.scope}</p>
                    <p>
                      Method:{" "}
                      {{
                        random_forest: "Random Forest",
                        RandomForestClassifier: "Random Forest",
                        extra_trees: "Extra Trees",
                        ExtraTreesClassifier: "Extra Trees",
                      }[analysis.model.metadata.model_family] ||
                        analysis.model.metadata.model_family ||
                        "Not recorded"}
                      {" · "}Evaluation:{" "}
                      {{
                        independent_confirmation: "Independent confirmation",
                        historical_diagnostic: "Previously inspected benchmark",
                        historical_holdout: "Version 1 held-out evaluation",
                        development_only: "Development validation only",
                      }[analysis.model.metadata.evaluation_status] ||
                        "Not recorded"}
                    </p>

                    <p>
                      Source:{" "}
                      {analysis.model.metadata.source?.includes(
                        "rest.uniprot.org",
                      )
                        ? "UniProtKB/Swiss-Prot"
                        : analysis.model.metadata.source || "Not recorded"}{" "}
                      · Release:{" "}
                      {analysis.model.metadata.release || "Not recorded"}
                    </p>
                    <p>
                      Training samples:{" "}
                      {analysis.model.metadata.sample_count ?? "Not recorded"}
                    </p>
                    {analysis.model.metadata.evaluation && (
                      <p>
                        Evaluation macro-F1:{" "}
                        {analysis.model.metadata.evaluation.macro_f1?.toFixed(
                          3,
                        ) ?? "Not recorded"}
                        . See the model card for the split and class-specific
                        limitations.
                      </p>
                    )}
                    <p>{analysis.model.metadata.limitations}</p>
                  </>
                ) : (
                  <p>
                    Provenance and evaluation metadata are unavailable for this
                    model. Do not assume these predictions have been
                    scientifically validated.
                  </p>
                )}
                <a href={DOCS} target="_blank" rel="noreferrer">
                  Read the methods and model card
                </a>
              </details>
            </>
          )
        )}
      </main>
      <footer>
        Research and education · Validate findings with experimental evidence.
      </footer>
    </div>
  );
}
