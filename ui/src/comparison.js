function ranks(values) {
    const sorted = values
        .map((value, index) => ({value, index}))
        .sort((a, b) => a.value - b.value);
    const result = [];
    for (let i = 0; i < sorted.length;) {
        let j = i + 1;
        while (j < sorted.length && sorted[j].value === sorted[i].value) j++;
        for (let k = i; k < j; k++) result[sorted[k].index] = (i + j - 1) / 2;
        i = j;
    }
    return result;
}

function correlation(x, y) {
    const a = ranks(x),
        b = ranks(y),
        am = a.reduce((s, v) => s + v, 0) / a.length,
        bm = b.reduce((s, v) => s + v, 0) / b.length;
    const numerator = a.reduce((s, v, i) => s + (v - am) * (b[i] - bm), 0);
    const denominator = Math.sqrt(
        a.reduce((s, v) => s + (v - am) ** 2, 0) *
        b.reduce((s, v) => s + (v - bm) ** 2, 0),
    );
    return denominator ? numerator / denominator : null;
}

const identity = (r) =>
    JSON.stringify([r.protein.protein_id, r.sequence_sha256]);
const game = (r) =>
    JSON.stringify({
        method: r.explainer.method,
        reference_policy: r.explainer.reference_policy,
        region_policy: r.explainer.region_policy,
        references: r.explainer.references,
        pool: r.explainer.reference_pool_id,
        features: r.features.map((f) => [f.feature_id, f.start, f.end]),
    });

export function compareReports(left, right) {
    const other = new Map(right.results.map((r) => [identity(r), r]));
    const classes = left.model.classes.filter((c) =>
        right.model.classes.includes(c),
    );
    const paired = new Set();
    const results = [];
    for (const a of left.results) {
        const b = other.get(identity(a));
        if (!b) continue;
        paired.add(identity(a));
        const compatible = game(a) === game(b),
            union = new Set([...a.decisions, ...b.decisions]);
        results.push({
            protein_id: a.protein.protein_id,
            sequence_sha256: a.sequence_sha256,
            decision_jaccard: union.size
                ? a.decisions.filter((c) => b.decisions.includes(c)).length / union.size
                : 1,
            attribution_comparability: compatible
                ? "common_reference_game"
                : "confounded_by_methodology",
            biological_region_concordance: [
                [left, a],
                [right, b],
            ].map(([report, local]) => ({
                model_id: report.model.model_id,
                annotations: local.annotations.flatMap((ann) =>
                    local.explained_classes.map((label, ci) => {
                        const indices = local.features
                            .map((f, i) => ({f, i}))
                            .filter(
                                ({f}) =>
                                    !ann.uncertain &&
                                    f.start != null &&
                                    f.end != null &&
                                    ann.start <= f.start &&
                                    f.end <= ann.end,
                            );
                        return {
                            annotation_id: ann.annotation_id,
                            class_id: label,
                            complete_region_count: indices.length,
                            positive_attribution_density: indices.length
                                ? indices.reduce(
                                (sum, {i}) =>
                                    sum + Math.max(local.attributions[ci][i], 0),
                                0,
                            ) / indices.reduce((sum, {f}) => sum + f.end - f.start, 0)
                                : null,
                            status: indices.length
                                ? "descriptive_overlap"
                                : "insufficient_resolution",
                        };
                    }),
                ),
            })),
            classes: classes.map((c) => {
                const ia = a.explained_classes.indexOf(c),
                    ib = b.explained_classes.indexOf(c),
                    x = a.attributions[ia],
                    y = b.attributions[ib];
                const selected = x
                    .map((v, i) => i)
                    .filter(
                        (i) =>
                            Math.abs(x[i]) > 0.005 || (compatible && Math.abs(y[i]) > 0.005),
                    );
                const coverage = (r, index) =>
                    new Set(
                        r.features
                            .map((f, i) => ({f, v: Math.abs(r.attributions[index][i])}))
                            .sort((a, b) => b.v - a.v)
                            .slice(0, Math.max(1, Math.ceil(0.2 * r.features.length)))
                            .flatMap(({f}) =>
                                f.start == null
                                    ? []
                                    : Array.from(
                                        {length: f.end - f.start},
                                        (_, i) => f.start + i,
                                    ),
                            ),
                    );
                const ca = coverage(a, ia),
                    cb = coverage(b, ib),
                    cu = new Set([...ca, ...cb]);
                return {
                    class_id: c,
                    probability_difference: b.probabilities[ib] - a.probabilities[ia],
                    magnitude_spearman: compatible
                        ? correlation(x.map(Math.abs), y.map(Math.abs))
                        : null,
                    sign_agreement:
                        compatible && selected.length
                            ? selected.filter((i) => Math.sign(x[i]) === Math.sign(y[i]))
                            .length / selected.length
                            : null,
                    top_region_residue_coverage_jaccard: cu.size
                        ? [...ca].filter((v) => cb.has(v)).length / cu.size
                        : null,
                };
            }),
        });
    }
    return {
        schema_version: 1,
        kind: "model_behavior_comparison",
        left_model: left.model.model_id,
        right_model: right.model.model_id,
        results,
        unmatched_left: left.results
            .filter((r) => !paired.has(identity(r)))
            .map((r) => r.protein.protein_id),
        unmatched_right: right.results
            .filter((r) => !paired.has(identity(r)))
            .map((r) => r.protein.protein_id),
        interpretation:
            "Model-specific behavior; disagreement is not evidence that one model is biologically wrong. Different reference games confound attribution comparisons.",
    };
}
