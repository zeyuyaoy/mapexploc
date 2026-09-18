# Human baseline model card

This is a compact, reproducible scientific baseline for research and interface testing, **not a validated biological
localization service**. Its strongest held-out class is Membrane; Cytoplasm and Secreted are substantially weaker.

## Identity and provenance

- Model ID: `human-2026_03-56fb89f09b33`.
- Source: reviewed human canonical proteins from UniProtKB/Swiss-Prot release **2026_03**, retrieved 2026-09-18T04:45:
  40.032133+00:00.
- Dataset: **1,822 proteins**, five classes, capped at 500 proteins per class; **1,458 training** and **364 test**
  proteins.
- Source SHA-256: `14be5581ec2412c7da60889232c2b9922b98373ef5a76d1350c460054447e967`.
- Prepared CSV SHA-256: `56fb89f09b33ec32a0d1c443612840f77f49e8e6eb67621901663640124729a3`.
- Artifact SHA-256: `cd8ecb31c3c177efcf136fa84791839c6ca185ce3525a77aae9155a87b062082`.
- Artifact size: 5,004,497 bytes.
- Attribution: UniProt Consortium, [CC BY 4.0](https://www.uniprot.org/help/license). The repository's MIT license does
  not replace this data attribution.

## Curation and split

Only `ECO:0000269` evidence attached to the location is accepted. Protein-existence evidence alone is insufficient.
Isoform/product-specific locations, ambiguous residues, unmapped compartments and conflicting locations are excluded.
Generic “Membrane” is excluded; Cell membrane / Plasma membrane map to the model label Membrane. Exact duplicate
sequences are removed; conflicting duplicate labels are excluded.

The frozen CSV retains accessions, canonical sequences, labels, supporting evidence, similarity groups and split
assignments. MMseqs2 18-8cc5c searches at 30% identity and 80% coverage of both sequences; connected components of
detected pairs stay together. Searches in both train-to-test and test-to-train directions found **zero qualifying
cross-partition matches**. This threshold does not exclude all remote or domain-level homology, and search sensitivity
is finite.

Three-fold stratified group cross-validation selects among 128/256 trees, depth 12/24, and minimum leaf size 1/3.
Features are the fixed 423-column schema, scaled within the pipeline. Class weights are balanced; SMOTE is disabled. The
test set is never used for fitting or parameter selection, and the distributed artifact is **not refitted on test
proteins**.

## Measured performance

| Metric                 | Random Forest | Prior dummy classifier |
|------------------------|--------------:|-----------------------:|
| Macro-F1               |         0.522 |                  0.086 |
| Weighted F1            |         0.547 |                  0.118 |
| Balanced accuracy      |         0.529 |                  0.200 |
| Log loss               |         1.187 |                  1.542 |
| Multiclass Brier score |         0.597 |                  0.775 |
| Top-label ECE, 10 bins |         0.162 |                  0.000 |

Brier score is the mean sum of squared class-probability errors (range 0–2); lower is better. ECE compares maximum
probability with observed correctness in ten equally spaced bins. These measurements do not make the probabilities
calibrated.

| Class         | Test proteins | Precision | Recall |    F1 |
|---------------|--------------:|----------:|-------:|------:|
| Cytoplasm     |            79 |     0.348 |  0.291 | 0.317 |
| Membrane      |           100 |     0.675 |  0.770 | 0.720 |
| Mitochondrion |            39 |     0.525 |  0.538 | 0.532 |
| Nucleus       |           100 |     0.641 |  0.590 | 0.615 |
| Secreted      |            46 |     0.404 |  0.457 | 0.429 |

Confusion matrix: rows are observed labels, columns are predicted labels.

| Observed / predicted | Cytoplasm | Membrane | Mitochondrion | Nucleus | Secreted |
|----------------------|----------:|---------:|--------------:|--------:|---------:|
| Cytoplasm            |        23 |       16 |            13 |      20 |        7 |
| Membrane             |         6 |       77 |             0 |       6 |       11 |
| Mitochondrion        |         8 |        4 |            21 |       1 |        5 |
| Nucleus              |        22 |        6 |             5 |      59 |        8 |
| Secreted             |         7 |       11 |             1 |       6 |       21 |

Selected parameters: `{'rf__max_depth': 24, 'rf__min_samples_leaf': 3, 'rf__n_estimators': 128}`. Grouped training CV
macro-F1: **0.568**. This is a single held-out evaluation, without confidence intervals or external validation.

## Runtime

Measured on macOS-27.0-arm64-arm-64bit-Mach-O, 12 logical CPUs, 4 search workers. Timings are medians of five warm
repetitions; they exclude HTTP/browser overhead and are observations, not latency guarantees.

| Operation                                       |     Time |
|-------------------------------------------------|---------:|
| Feature extraction, one test protein            |  4.88 ms |
| Prediction, one precomputed feature row         |  5.59 ms |
| Prediction, 100 precomputed feature rows        |  5.21 ms |
| SHAP, one precomputed feature row               | 12.55 ms |
| Feature extraction and complete model selection |   9.60 s |

## Limitations and intended use

Human canonical proteins with one experimentally supported compartment only. Multilocalized, isoform-specific,
unsupported and ambiguous annotations were excluded. Probabilities are uncalibrated; class-balanced curation does not
represent natural prevalence. Sequence grouping at 30% identity and 80% bidirectional coverage does not exclude all
remote or domain-level homology. SHAP describes engineered features, not biological causality.

Use for repeatable demonstrations, integration tests and comparison with improved research methods. The model always
selects among its five known classes; it cannot establish that an input belongs to those classes, infer multiple
localizations, or reliably generalize to other organisms. Full canonical precursor sequences can differ from the mature
protein whose localization is annotated. The example proteins are chosen by accession and length from the held-out
partition, not by whether the model predicts them correctly.

## Reproduce and inspect

See [baseline reproduction](baseline.md). Machine-readable metrics, confusion matrix, individual held-out predictions
and runtime details live beside the trusted model under `examples/models/`. Library versions used for the artifact:

```json
{
  "biopython": "1.88",
  "numpy": "2.5.2",
  "pandas": "3.0.5",
  "scikit-learn": "1.9.0",
  "shap": "0.52.0"
}
```

Pickle-based models must come from a trusted source. Use matching library versions to load this artifact or retrain it
in your environment; serialization is not guaranteed across scikit-learn versions.
