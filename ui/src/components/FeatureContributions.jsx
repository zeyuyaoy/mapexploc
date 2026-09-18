export function featureLabel(name) {
  if (name === "length") return "Sequence length";
  if (name === "gravy") return "Hydrophobicity (GRAVY)";
  if (name === "isoelectric_point") return "Isoelectric point";
  if (name.startsWith("aa_")) return `${name.slice(3)} residue frequency`;
  if (name.startsWith("dp_")) return `${name.slice(3)} dipeptide frequency`;
  return name.replaceAll("_", " ");
}

export default function FeatureContributions({
  prediction,
  contributions,
  baseValue,
}) {
  const max = Math.max(
    ...contributions.map((item) => Math.abs(item.contribution)),
    0.00001,
  );
  return (
    <figure className="chart">
      <figcaption>What influenced {prediction}?</figcaption>
      <p className="muted">
        Positive contributions support this class; negative contributions oppose
        it.
      </p>
      <div className="shap-chart" aria-hidden="true">
        {contributions.map((item) => (
          <div className="shap-row" key={item.feature}>
            <span>{featureLabel(item.feature)}</span>
            <div className="signed-track">
              <i />
              <b
                className={item.contribution >= 0 ? "positive" : "negative"}
                style={{
                  left: `${item.contribution >= 0 ? 50 : 50 - (Math.abs(item.contribution) / max) * 50}%`,
                  width: `${(Math.abs(item.contribution) / max) * 50}%`,
                }}
              />
            </div>
            <strong>
              {item.contribution > 0 ? "+" : ""}
              {item.contribution.toFixed(4)}
            </strong>
          </div>
        ))}
        <div className="shap-axis">
          <span>−{max.toFixed(3)}</span>
          <span>0</span>
          <span>+{max.toFixed(3)}</span>
        </div>
        <p className="axis-label">
          SHAP contribution to predicted-class probability
        </p>
      </div>
      <p className="science-note">
        Model baseline: {baseValue.toFixed(4)}. Only the largest contributions
        are shown; these alone need not sum to the prediction. These are
        engineered-feature explanations, not residue-level or causal biological
        evidence.
      </p>
      <details className="data-disclosure">
        <summary>View feature values and contributions</summary>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Feature</th>
                <th>Measured value</th>
                <th>SHAP contribution</th>
              </tr>
            </thead>
            <tbody>
              {contributions.map((item) => (
                <tr key={item.feature}>
                  <th>{featureLabel(item.feature)}</th>
                  <td>{item.value.toFixed(4)}</td>
                  <td>{item.contribution.toFixed(6)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </figure>
  );
}
