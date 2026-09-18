export function ProbabilityChart({probabilities}) {
    const sorted = [...probabilities].sort(
        (a, b) => b.probability - a.probability,
    );
    return (
        <figure className="chart" aria-label="Class probabilities">
            <figcaption>Class probabilities</figcaption>
            <div aria-hidden="true" className="bar-chart">
                {sorted.map((item, index) => (
                    <div className="bar-row" key={item.label}>
                        <span className="bar-label">{item.label}</span>
                        <div className="bar-area">
                            <div
                                className={`bar ${index === 0 ? "primary-bar" : ""}`}
                                style={{width: `${item.probability * 100}%`}}
                            />
                        </div>
                        <strong>{(item.probability * 100).toFixed(1)}%</strong>
                    </div>
                ))}
                <div className="axis">
                    <span>0</span>
                    <span>0.2</span>
                    <span>0.4</span>
                    <span>0.6</span>
                    <span>0.8</span>
                    <span>1.0</span>
                </div>
                <p className="axis-label">Probability</p>
            </div>
            <details className="data-disclosure">
                <summary>View probability values</summary>
                <table>
                    <caption className="sr-only">Probability data</caption>
                    <thead>
                    <tr>
                        <th>Compartment</th>
                        <th>Probability</th>
                    </tr>
                    </thead>
                    <tbody>
                    {sorted.map((item) => (
                        <tr key={item.label}>
                            <th>{item.label}</th>
                            <td>{item.probability.toFixed(4)}</td>
                        </tr>
                    ))}
                    </tbody>
                </table>
            </details>
        </figure>
    );
}

export function SequenceChart({features, record}) {
    const entries = Object.entries(features.composition);
    const upper = Math.max(...entries.map(([, value]) => value), 0.01);
    return (
        <>
            <div className="descriptors">
                <div>
                    <span>Length</span>
                    <strong>{features.sequence_length.toLocaleString()} aa</strong>
                </div>
                <div>
                    <span>Hydrophobicity · GRAVY</span>
                    <strong>{features.gravy.toFixed(3)}</strong>
                </div>
                <div>
                    <span>Isoelectric point · pI</span>
                    <strong>{features.isoelectric_point.toFixed(2)}</strong>
                </div>
            </div>
            <figure className="chart">
                <figcaption>Amino-acid composition</figcaption>
                <p className="muted">
                    Fraction of the sequence made up by each residue. Plot range: 0–
                    {(upper * 100).toFixed(1)}%.
                </p>
                <div className="composition" aria-hidden="true">
                    {entries.map(([aa, value]) => (
                        <div className="residue" key={aa}>
                            <span className="residue-value">{(value * 100).toFixed(1)}%</span>
                            <div className="composition-track">
                                <span style={{height: `${(value / upper) * 100}%`}}/>
                            </div>
                            <strong>{aa}</strong>
                        </div>
                    ))}
                </div>
                <details className="data-disclosure">
                    <summary>View composition values</summary>
                    <table>
                        <thead>
                        <tr>
                            <th>Residue</th>
                            <th>Fraction</th>
                        </tr>
                        </thead>
                        <tbody>
                        {entries.map(([aa, value]) => (
                            <tr key={aa}>
                                <th>{aa}</th>
                                <td>{value.toFixed(4)}</td>
                            </tr>
                        ))}
                        </tbody>
                    </table>
                </details>
            </figure>
            <details className="data-disclosure">
                <summary>View analyzed sequence</summary>
                <p className="sequence-preview">{record.sequence}</p>
            </details>
            <p className="science-note">
                These descriptors summarize the input sequence. They do not identify
                targeting signals or causal motifs.
            </p>
        </>
    );
}
