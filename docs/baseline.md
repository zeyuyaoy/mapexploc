# Reproducing the human baseline

The repository separates a tiny **smoke-test fixture** from a curated **scientific baseline**. Smoke-test predictions are not performance evidence. The [model card](model-card.md) reports the actual baseline's limitations and held-out scores.

This page preserves version 1 reproduction. The [executed model comparison](model-improvement.md) documents the separate expanded-data Random Forest/Extra Trees workflow and why version 1 remains the default.

## Use the evaluated artifact

From the repository root, after installing the package:

```bash
MAPEXPLOC_MODEL_PATH=examples/models/human-baseline.joblib \
  python -m uvicorn mapexploc.api:app --host 127.0.0.1 --port 8000
```

The existing root `model.pkl` is not replaced. The human artifact uses 423 features and predicts Cytoplasm, Membrane, Mitochondrion, Nucleus or Secreted. Its library versions and checksum are in the model card and `examples/models/human-baseline.report.json`.

## Retrain the frozen prepared snapshot, offline

The checked-in `examples/baseline/dataset.csv` retains evidence, groups and partitions. Its exact checksum is verified before training; the accompanying manifest records the source release, filters and zero-match similarity audit. Ordinary tests do not download data, run MMseqs2, or retrain this full benchmark.

```bash
mapexploc baseline-train --directory examples/baseline \
  --output-model artifacts/reproduced-human.joblib --jobs 4
```

This preserves the test partition. It writes the model, `.report.json`, and `.predictions.csv` without overwriting an existing artifact. Use the model card's exact library versions for the closest numerical reproduction; timings and serialized byte hashes may differ across environments.

## Refresh from UniProt explicitly

Install MMseqs2 separately (for example `brew install mmseqs2` on macOS or the official distribution for your operating system). It is a preparation dependency, not an inference dependency.

```bash
mapexploc baseline-download --directory artifacts/new-human-snapshot
mapexploc baseline-prepare --directory artifacts/new-human-snapshot --threads 4
mapexploc baseline-train --directory artifacts/new-human-snapshot \
  --output-model artifacts/new-human-snapshot/model.joblib --jobs 4
```

Downloading retrieves only the public reviewed-human dataset. No user sequence is sent to UniProt. Existing snapshots are never silently refreshed. A new release can change accepted proteins and scores; it is a new benchmark, not an exact reproduction of the previous release.

Preparation uses direct experimental localization evidence, excludes conflicting assignments, limits each class to 500 proteins with seed 42, and removes duplicate sequences. MMseqs2 all-against-all search produces connected similarity components at 30% identity and 80% bidirectional coverage. Approximately 20% is held out with at least ten test proteins per class. Both cross-partition search directions must have zero qualifying matches. Insufficient class coverage or detected leakage stops preparation; the filters are never relaxed automatically.

The 80% training partition is used for three-fold grouped model selection. No oversampling occurs. The distributed model remains fitted only to training proteins after test evaluation.

## Files and provenance

- `source.json` / `source.headers`: raw source and release headers (local artifacts).
- `dataset.csv` / `manifest.json`: frozen prepared data, evidence, groups, partitions and checksums.
- `all-pairs.tsv` / `audit-*.tsv`: detected similarities and empty successful cross-partition audits.
- `*.joblib`: trusted versioned Random Forest artifact.
- `*.report.json` / `*.predictions.csv`: complete evaluation, timings, versions, selected parameters and individual test
  predictions.

UniProt data is attributed to the UniProt Consortium under [CC BY 4.0](https://www.uniprot.org/help/license). Read the [UniProt localization guide](https://www.uniprot.org/help/subcellular_location)and [MMseqs2 documentation](https://github.com/soedinglab/MMseqs2) for source semantics and search limitations.

## Scientific environment compatibility

The exact original scientific library versions are recorded in `examples/models/scientific-environment.txt` in the repository. To use those versions, install from the repository root with:

```bash
python -m pip install -e . -c examples/models/scientific-environment.txt
```

The original artifact is unchanged. Compatibility checks with NumPy 2.5.3, pandas 3.0.6 and scikit-learn 1.9.1 reproduce all 364 held-out labels and probabilities (tolerance 1e-12), with SHAP additivity checked on ten proteins. Scikit-learn still emits a version warning for an artifact created with 1.9.0; this targeted regression check does not establish general cross-version pickle compatibility. Use the recorded environment for exact benchmark reproduction.
