import { useEffect, useRef, useState } from "react";
import { request } from "../api";
import { download, parseSequences } from "../analysis";

export default function NativeWorkflow({ onBack }) {
  const [models, setModels] = useState([]);
  const [selected, setSelected] = useState("");
  const [input, setInput] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const active = useRef(null);
  const generation = useRef(0);
  const model = models.find((m) => m.adapter_id === selected);
  useEffect(() => {
    const controller = new AbortController();
    request("/v3/models", { signal: controller.signal })
      .then((data) => {
        if (controller.signal.aborted) return;
        const entries = data.models.filter((m) =>
          ["fast", "accurate"].includes(m.mode),
        );
        setModels(entries);
        setSelected(
          (entries.find((m) => m.mode === "fast") || entries[0])?.adapter_id ||
            "",
        );
      })
      .catch((e) => {
        if (!controller.signal.aborted) setError(e.message);
      });
    return () => {
      controller.abort();
      active.current?.abort();
    };
  }, []);

  function invalidate() {
    generation.current += 1;
    active.current?.abort();
    setBusy(false);
    setResult(null);
    setError("");
  }

  async function predict(event) {
    event.preventDefault();
    invalidate();
    const current = generation.current;
    const controller = new AbortController();
    active.current = controller;
    try {
      const proteins = parseSequences(input, "batch").map((r) => ({
        protein_id: r.id,
        sequence: r.sequence,
      }));
      setBusy(true);
      const response = await request("/v3/predict", {
        signal: controller.signal,
        body: {
          adapter_id: selected,
          expected_model_id: model.model_id,
          expected_checkpoint_sha256: model.expected_checkpoint_sha256 || null,
          proteins,
        },
      });
      if (controller.signal.aborted || current !== generation.current) return;
      if (
        response.model.model_id !== model.model_id ||
        response.model.provenance.mode !== model.mode ||
        response.results.length !== proteins.length ||
        response.results.some(
          (r, i) =>
            r.protein.protein_id !== proteins[i].protein_id ||
            r.protein.sequence !== proteins[i].sequence,
        )
      )
        throw new Error(
          "Returned model/protein identity differs from the requested analysis.",
        );
      setResult(response);
    } catch (e) {
      if (!controller.signal.aborted) setError(e.message);
    } finally {
      if (current === generation.current) setBusy(false);
    }
  }

  function prepare() {
    const configuration = {
      method_profile: "v2x-legacy",
      explainer: "region_kernel",
      seed: 42,
      references: 4,
      coalition_budget: 512,
      cohort_id: result.results.length > 1 ? "user-cohort" : null,
      selection_criteria: "All proteins in the exported FASTA; no sampling",
    };
    download(
      JSON.stringify(
        {
          schema_version: 1,
          adapter: "deeploc2",
          mode: model.mode,
          expected_model_id: result.model.model_id,
          expected_checkpoint_sha256: result.model.checkpoint_sha256,
          configuration,
        },
        null,
        2,
      ),
      "mapexploc-run.json",
      "application/json",
    );
  }

  return (
    <section className="report-viewer">
      <p className="eyebrow">Native pretrained models</p>
      <h1>DeepLoc model mode</h1>
      <button className="text-button" onClick={onBack}>
        Use the existing RF predictor
      </button>
      <form onSubmit={predict}>
        <label>
          Model mode
          <select
            aria-label="DeepLoc model mode"
            value={selected}
            onChange={(e) => {
              invalidate();
              setSelected(e.target.value);
            }}
          >
            {models.map((m) => (
              <option
                key={m.adapter_id}
                value={m.adapter_id}
                disabled={m.readiness === "unavailable"}
              >
                {m.mode === "fast"
                  ? "Fast — lower compute, higher throughput"
                  : "Accurate — ProtT5, substantially higher compute"}
                {m.readiness === "unavailable" ? " (unavailable)" : ""}
              </option>
            ))}
          </select>
        </label>
        {!models.length && (
          <p>
            No DeepLoc modes are configured by this service operator. Completed
            reports can still be imported.
          </p>
        )}
        {model && (
          <>
            <p className="science-note">
              {model.tradeoff}. Accurate may improve predictive performance for
              some inputs; it is not necessarily preferable for every analysis.
              Explanations run through the CLI.
            </p>
            <p>
              Device: {model.device || "adapter-defined"}.{" "}
              {model.estimate?.status ||
                "No calibrated runtime estimate available"}
              .
            </p>
            {model.issues?.map((issue) => (
              <p role="alert" key={issue}>
                {issue}
              </p>
            ))}
          </>
        )}
        <label htmlFor="native-proteins">Protein sequences (FASTA)</label>
        <textarea
          id="native-proteins"
          value={input}
          onChange={(e) => {
            invalidate();
            setInput(e.target.value);
          }}
          rows={8}
        />
        <button
          className="primary-button"
          disabled={busy || !model || model.readiness === "unavailable"}
        >
          {busy ? "Running native prediction…" : "Predict with selected mode"}
        </button>
        {busy && (
          <button type="button" onClick={invalidate}>
            Cancel
          </button>
        )}
      </form>
      {error && (
        <p role="alert" className="error">
          {error}
        </p>
      )}
      {result && (
        <section>
          <h2>{result.model.model_id}</h2>
          <p>
            Checkpoint: <code>{result.model.checkpoint_sha256}</code>
          </p>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Protein</th>
                  <th>Native decisions</th>
                  {result.model.classes.map((c) => (
                    <th key={c}>{c}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {result.results.map((r) => (
                  <tr key={r.protein.protein_id}>
                    <th>{r.protein.protein_id}</th>
                    <td>{r.decisions.join(", ") || "None above threshold"}</td>
                    {r.probabilities.map((p, i) => (
                      <td key={result.model.classes[i]}>{p.toPrecision(5)}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <h3>Prepare a complete explanation</h3>
          <p className="science-note">
            Each protein may require thousands of native evaluations. Accurate
            is substantially more expensive. The CLI records progress and can
            resume completed work; import its report here when finished.
            Historical signal-peptide validation did not outperform the matched
            null at conventional significance (p = 0.0779).
          </p>
          <button className="secondary-button" onClick={prepare}>
            Export exact run configuration
          </button>
          <button
            className="secondary-button"
            onClick={() =>
              download(
                result.results
                  .map(
                    (r) => `>${r.protein.protein_id}\n${r.protein.sequence}\n`,
                  )
                  .join(""),
                "proteins.fasta",
                "text/plain",
              )
            }
          >
            Export input FASTA
          </button>
          <pre>{`mapexploc analyze --adapter deeploc2 --adapter-config /path/to/operator-config.json --model-mode ${model.mode} --run-spec mapexploc-run.json --fasta proteins.fasta --cache-dir results/cache --restart-dir results/restart-${model.mode} --output-dir results/${model.mode}`}</pre>
          <details>
            <summary>Native provenance</summary>
            <pre>{JSON.stringify(result.model, null, 2)}</pre>
          </details>
        </section>
      )}
    </section>
  );
}
