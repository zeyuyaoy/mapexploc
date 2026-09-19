// Match NumPy's round-to-even rule for native fallback decisions.
function roundNativeProbability(value, decimals) {
  const scale = 10 ** decimals;
  const scaled = value * scale;
  const lower = Math.floor(scaled);
  return (
    (scaled - lower === 0.5 ? lower + (lower % 2) : Math.round(scaled)) / scale
  );
}

// Views preserve the full report without modifying it.
const originalReportText = new WeakMap();

export function readReportText(text) {
  const report = validateReport(JSON.parse(text));
  originalReportText.set(report, text);
  return report;
}

export function exportReportText(report) {
  return originalReportText.get(report) ?? JSON.stringify(report, null, 2);
}

export function validateReport(report) {
  const fail = (message) => {
    throw new Error(`Invalid explanation report: ${message}`);
  };
  if (
    ![2, 3].includes(report?.schema_version) ||
    report.status !== "complete" ||
    report.output_space !== "probability"
  )
    fail("a complete schema v2 or v3 probability report is required.");
  if (report.schema_version === 3) {
    if (
      !report.runtime ||
      report.runtime.checkpoint_sha256 !== report.model?.checkpoint_sha256 ||
      report.runtime.preprocessing_id !== report.model?.preprocessing_id ||
      !/^[0-9a-f]{64}$/.test(report.methodology_sha256) ||
      !Array.isArray(report.diagnostics) ||
      !Array.isArray(report.global_statistics) ||
      !Array.isArray(report.structured_warnings)
    )
      fail("incomplete v3 runtime or diagnostic metadata.");
    if (
      JSON.stringify(report.diagnostics.map((d) => d.protein_id)) !==
      JSON.stringify(report.results?.map((r) => r.protein.protein_id))
    )
      fail("diagnostic protein identities disagree.");
    for (const row of report.global_statistics)
      if (
        ![row.mean, row.median, row.trimmed_mean, ...row.interval_95].every(
          Number.isFinite,
        ) ||
        row.group_count < 1 ||
        row.protein_count < row.group_count
      )
        fail("invalid group statistics.");
  }
  const classes = report.model?.classes;
  if (
    !Array.isArray(classes) ||
    !classes.length ||
    new Set(classes).size !== classes.length
  )
    fail("invalid model classes.");
  if (!Array.isArray(report.results) || !report.results.length)
    fail("no protein results.");
  const ids = new Set();
  for (const local of report.results) {
    const protein = local.protein;
    if (
      !protein?.protein_id ||
      ids.has(protein.protein_id) ||
      !/^[ACDEFGHIKLMNPQRSTVWY]+$/.test(protein.sequence)
    )
      fail("invalid or duplicate protein identity.");
    ids.add(protein.protein_id);
    if (JSON.stringify(local.explained_classes) !== JSON.stringify(classes))
      fail("class order differs from the model.");
    const features = local.features;
    if (
      !Array.isArray(features) ||
      !features.length ||
      new Set(features.map((f) => f.feature_id)).size !== features.length
    )
      fail("invalid feature identities.");
    if (
      !Array.isArray(local.feature_values) ||
      local.feature_values.length !== features.length ||
      !local.feature_values.every(Number.isFinite)
    )
      fail("invalid feature values.");
    for (const field of [
      "probabilities",
      "base_values",
      "residuals",
      "attributions",
    ]) {
      if (
        !Array.isArray(local[field]) ||
        local[field].length !== classes.length
      )
        fail("class axes disagree.");
    }
    if (
      !local.probabilities.every((p) => Number.isFinite(p) && p >= 0 && p <= 1)
    )
      fail("invalid probabilities.");
    if (
      report.model.task_type === "multiclass" &&
      Math.abs(local.probabilities.reduce((a, b) => a + b, 0) - 1) > 1e-6
    )
      fail("multiclass probabilities do not sum to one.");
    if (!["multiclass", "multilabel"].includes(report.model.task_type))
      fail("unknown task type.");
    if (
      !Array.isArray(local.decisions) ||
      local.decisions.some((d) => !classes.includes(d))
    )
      fail("unknown predicted class.");
    let decisions;
    if (report.model.task_type === "multiclass") {
      const maximum = Math.max(...local.probabilities);
      decisions = [classes[local.probabilities.indexOf(maximum)]];
    } else {
      const thresholds = report.model.thresholds;
      if (
        !Array.isArray(thresholds) ||
        thresholds.length !== classes.length ||
        !thresholds.every((t) => Number.isFinite(t) && t >= 0 && t <= 1)
      )
        fail("invalid native thresholds.");
      decisions = classes.filter((_, i) =>
        report.model.threshold_comparison === "greater"
          ? local.probabilities[i] > thresholds[i]
          : local.probabilities[i] >= thresholds[i],
      );
      if (
        !decisions.length &&
        report.model.empty_decision_policy === "nearest_threshold"
      ) {
        const decimals = report.model.fallback_round_decimals;
        const distances = local.probabilities.map(
          (p, i) =>
            (decimals == null ? p : roundNativeProbability(p, decimals)) -
            thresholds[i],
        );
        const maximum = Math.max(...distances);
        decisions = [classes[distances.lastIndexOf(maximum)]];
      }
    }
    if (JSON.stringify(decisions) !== JSON.stringify(local.decisions))
      fail("decisions disagree with native class scores and policy.");
    local.attributions.forEach((values, i) => {
      if (
        !Array.isArray(values) ||
        values.length !== features.length ||
        !values.every(Number.isFinite) ||
        !Number.isFinite(local.base_values[i]) ||
        !Number.isFinite(local.residuals[i])
      )
        fail("invalid attribution tensor.");
      const residual =
        local.base_values[i] +
        values.reduce((a, b) => a + b, 0) -
        local.probabilities[i];
      if (
        Math.abs(residual) > 1e-5 ||
        Math.abs(residual - local.residuals[i]) > 1e-9
      )
        fail("probability reconstruction failed.");
    });
    const regions = features.filter((f) => f.kind === "sequence_region");
    if (
      regions.length &&
      (regions.length !== features.length ||
        regions[0].start !== 0 ||
        regions.at(-1).end !== protein.sequence.length)
    )
      fail("regions must cover the sequence.");
    features.forEach((f, i) => {
      if (!["sequence_region", "engineered_descriptor"].includes(f.kind))
        fail("unsupported attribution kind.");
      if (
        f.kind === "sequence_region" &&
        (!Number.isInteger(f.start) ||
          !Number.isInteger(f.end) ||
          f.end <= f.start ||
          (i && features[i - 1].end !== f.start))
      )
        fail("invalid region coordinates.");
    });
    if (!Array.isArray(local.annotations))
      fail("missing annotation collection.");
    local.annotations.forEach((a) => {
      if (
        a.protein_id !== protein.protein_id ||
        a.sequence_sha256 !== local.sequence_sha256 ||
        !Number.isInteger(a.start) ||
        !Number.isInteger(a.end) ||
        a.start < 0 ||
        a.end <= a.start ||
        a.end > protein.sequence.length ||
        !a.source ||
        !a.source_version ||
        !a.source_url
      )
        fail("annotation provenance/coordinates do not match the protein.");
    });
  }
  if (
    report.results.length > 1 &&
    (!report.cohort ||
      report.cohort.included_count !== report.results.length ||
      JSON.stringify(report.cohort.protein_ids) !==
        JSON.stringify(report.results.map((r) => r.protein.protein_id)))
  )
    fail("cohort membership is incomplete.");
  if (report.cohort) {
    const expected = new Map();
    for (const local of report.results) {
      local.explained_classes.forEach((classId, ci) => {
        const within = new Map();
        local.features.forEach((feature, fi) => {
          const identity =
            feature.kind === "sequence_region"
              ? feature.category
              : feature.feature_id;
          if (!identity) fail("missing global feature/category identity.");
          const key = JSON.stringify([classId, identity]);
          if (!within.has(key)) within.set(key, []);
          within.get(key).push(local.attributions[ci][fi]);
        });
        for (const [key, values] of within) {
          if (!expected.has(key))
            expected.set(key, {
              absolute: 0,
              signed: 0,
              proteins: 0,
              regions: 0,
            });
          const row = expected.get(key);
          row.absolute +=
            values.reduce((sum, v) => sum + Math.abs(v), 0) / values.length;
          row.signed += values.reduce((sum, v) => sum + v, 0) / values.length;
          row.proteins += 1;
          row.regions += values.length;
        }
      });
    }
    const actual = report.cohort.contributions;
    if (!Array.isArray(actual) || actual.length !== expected.size)
      fail("incomplete global contributions.");
    const seen = new Set();
    for (const row of actual) {
      const key = JSON.stringify([row.class_id, row.feature_id]);
      const reference = expected.get(key);
      if (
        !reference ||
        seen.has(key) ||
        !Number.isFinite(row.mean_absolute) ||
        !Number.isFinite(row.mean_signed) ||
        Math.abs(row.mean_absolute - reference.absolute / reference.proteins) >
          1e-10 ||
        Math.abs(row.mean_signed - reference.signed / reference.proteins) >
          1e-10 ||
        row.protein_count !== reference.proteins ||
        row.region_count !== reference.regions
      )
        fail("global values do not reproduce from local explanations.");
      seen.add(key);
    }
  }
  if (report.schema_version === 3) {
    const expected = new Map();
    if (report.results.length > 1)
      for (const local of report.results) {
        const group = local.protein.group || local.sequence_sha256;
        local.explained_classes.forEach((label, ci) => {
          const within = new Map();
          local.features.forEach((feature, fi) => {
            const identity =
              feature.kind === "sequence_region"
                ? feature.category
                : feature.feature_id;
            const v = local.attributions[ci][fi];
            const measures = { signed: v, absolute: Math.abs(v) };
            if (feature.kind === "sequence_region")
              Object.assign(measures, {
                signed_density: v / (feature.end - feature.start),
                absolute_density: Math.abs(v) / (feature.end - feature.start),
              });
            for (const [statistic, value] of Object.entries(measures)) {
              const key = JSON.stringify([label, identity, statistic]);
              if (!within.has(key)) within.set(key, []);
              within.get(key).push(value);
            }
          });
          for (const [key, values] of within) {
            if (!expected.has(key))
              expected.set(key, { groups: new Map(), proteins: 0 });
            const row = expected.get(key);
            row.proteins++;
            if (!row.groups.has(group)) row.groups.set(group, []);
            row.groups
              .get(group)
              .push(values.reduce((a, b) => a + b, 0) / values.length);
          }
        });
      }
    if (report.global_statistics.length !== expected.size)
      fail("incomplete equal-group statistics.");
    const seen = new Set();
    for (const row of report.global_statistics) {
      const key = JSON.stringify([row.class_id, row.feature_id, row.statistic]),
        reference = expected.get(key);
      if (!reference || seen.has(key))
        fail("invalid equal-group feature identity.");
      seen.add(key);
      const values = [...reference.groups.values()]
        .map((v) => v.reduce((a, b) => a + b, 0) / v.length)
        .sort((a, b) => a - b);
      const mean = (v) => v.reduce((a, b) => a + b, 0) / v.length,
        n = values.length,
        trim = Math.floor(n * 0.1);
      const median =
        n % 2
          ? values[Math.floor(n / 2)]
          : (values[n / 2 - 1] + values[n / 2]) / 2;
      if (
        Math.abs(row.mean - mean(values)) > 1e-10 ||
        Math.abs(row.median - median) > 1e-10 ||
        Math.abs(row.trimmed_mean - mean(values.slice(trim, n - trim))) >
          1e-10 ||
        row.group_count !== n ||
        row.protein_count !== reference.proteins ||
        row.interval_95.length !== 2 ||
        row.interval_95[0] > row.interval_95[1] ||
        row.interval_95[0] < values[0] - 1e-10 ||
        row.interval_95[1] > values.at(-1) + 1e-10
      )
        fail("equal-group statistics disagree with local explanations.");
    }
    for (const diagnostic of report.diagnostics) {
      if (
        !Array.isArray(diagnostic.stability) ||
        !Array.isArray(diagnostic.faithfulness)
      )
        fail("missing diagnostic arrays.");
      for (const metric of diagnostic.stability)
        if (
          !classes.includes(metric.class_id) ||
          !Number.isFinite(metric.max_absolute_delta) ||
          metric.max_absolute_delta < 0 ||
          ![
            "stable",
            "unstable",
            "uninformative",
            "methodological_sensitivity",
          ].includes(metric.status)
        )
          fail("invalid stability diagnostic.");
    }
  }
  return report;
}

export async function verifySequenceHashes(report) {
  for (const local of report.results) {
    const digest = await crypto.subtle.digest(
      "SHA-256",
      new TextEncoder().encode(local.protein.sequence),
    );
    const hash = Array.from(new Uint8Array(digest), (v) =>
      v.toString(16).padStart(2, "0"),
    ).join("");
    if (hash !== local.sequence_sha256)
      throw new Error("Protein sequence checksum mismatch.");
  }
  return report;
}

function cell(value) {
  const text = String(value ?? "");
  return `"${(/^[=+@-]/.test(text) ? "'" : "") + text.replaceAll('"', '""')}"`;
}

export function reportAttributionCsv(report) {
  const rows = [
    [
      "protein_id",
      "class",
      "feature_id",
      "kind",
      "start_0",
      "end_exclusive",
      "value",
      "shap",
      "base_value",
      "residual",
    ],
  ];
  report.results.forEach((local) =>
    local.explained_classes.forEach((label, ci) =>
      local.features.forEach((feature, fi) =>
        rows.push([
          local.protein.protein_id,
          label,
          feature.feature_id,
          feature.kind,
          feature.start,
          feature.end,
          local.feature_values[fi],
          local.attributions[ci][fi],
          local.base_values[ci],
          local.residuals[ci],
        ]),
      ),
    ),
  );
  return rows
    .map((row) =>
      row.map((v) => (typeof v === "number" ? String(v) : cell(v))).join(","),
    )
    .join("\r\n");
}
