# Research software and evidence standards

The code can validate its inputs and reproduce calculations; it cannot establish annotation truth, biological independence or clinical utility by itself. A passing test suite is software evidence, not localization validation.

## Current scientific state

| System / record                       | Status                                                     | Permitted interpretation                                                               |
|---------------------------------------|------------------------------------------------------------|----------------------------------------------------------------------------------------|
| Served 423-feature forest v1          | Original artifact retained                                 | Historical 364-protein holdout macro-F1 0.5223; not a fresh independent test           |
| Expanded forest                       | Archived, not promoted                                     | Historical benchmark comparison only                                                   |
| Adaptive nested development procedure | Completed internal experiment                              | Macro-F1 0.6094 versus 0.5462 on matched outer folds; conditional internal uncertainty |
| Frozen 63-feature logistic candidate  | Exact fitted artifact retained                             | Fixed-configuration development diagnostic 0.6006; no external confirmation            |
| External validation study             | Model and methods locked locally; cohort/reviewers pending | No external performance result and no completed independent adjudication               |
| Endogenous biological validation      | Not performed                                              | No wet-lab confirmation                                                                |

These populations and procedures differ. Do not subtract scores across rows as though they were a paired experiment. The structured annotation target does not establish exclusive biological localization. Notes, ambiguity and multilocalization require the independently adjudicated reference standard specified in the [external protocol](external-validation.md).

## Metric and training contracts

`classification_metrics` is the shared implementation for the research workflows and generic RF/k-NN evaluation helpers. It fixes the class universe, preserves probability-column order, checks finite normalized probabilities and rejects unknown truth/prediction labels. Macro-F1 includes all declared classes, using zero F1 for zero-denominator classes. AUC/AP are null when an outcome is absent. Brier loss is the mean **sum** of squared class errors, with range 0–2; ECE is an empirical binned diagnostic, not a calibration guarantee. Generic reports retain legacy keys and add common probability metrics. No generic helper creates statistical independence. See the [scikit-learn metric definitions](https://scikit-learn.org/stable/modules/model_evaluation.html).

Pass precomputed biological dependence groups to model selection. k-NN now uses stratified or grouped folds with sufficient class support and checks neighbor counts against the smallest actual training fold. Small-sample KFold scores with missing classes are not a defensible substitute. RF's no-CV tiny-data mode remains available for workflow fixtures and returns no validation score. Report selection scores as selection scores. See [grouped cross-validation guidance](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data).

`aopc` averages precomputed signed reference-minus-perturbed score drops. The legacy `insertion_deletion` function never computed curve AUC; it is now a deprecated alias for the explicitly named `mean_absolute_score_change`. Do not report output change as causal attribution or biological faithfulness. NaN, infinite, empty and malformed perturbation inputs are rejected.

The DAT parser is a legacy convenience parser with coarse synonyms and discarded notes. It is not the structured JSON curator or independent annotation adjudication. Do not silently convert `Other`, missing measurements or ambiguous evidence into negative training labels.

## Pre-evaluation maintenance revision 1.1

At the start of this cleanup, the original external-study lock failed: 23 referenced files differed from recorded hashes. Canonical Black/import formatting restored all 20 changed Python files to their original hashes before semantic edits. Remaining differences were protocol/document/template formatting. Both fitted-model hashes matched throughout. The incoming files and integrity inventory were preserved under `artifacts/scientific-maintenance/`; this is a working audit, not a historical result.

The original `examples/validation/external-v1/freeze.json` and all scientific result archives are retained. **The active execution manifest is `freeze-v1.1.json`.** It records the original manifest's hash and the maintenance amendment, rather than silently overwriting a lock. The scientific protocol, exact model bytes, feature definitions, single primary estimand, eligibility rules, bootstrap specification and decision thresholds are unchanged. Software/doc corrections were made before any external evaluation; no external-study receipt exists at this revision's creation.

The amendment includes shared generic evaluation, safer exploratory training, localization-set equality independent of ordering, accurate docstrings and consistent scientific documentation. It does not claim improved predictive performance. New tests use artificial cases; no annotation review is marked complete by them.

```bash
PYTHONPATH=src python -m mapexploc.external_validation verify
```

Use the exact Python and dependency versions recorded in the active manifest. `execution-v1.1.tar.gz` and its SHA-256 sidecar preserve the active manifest and every  file it hashes. Extract into a new directory, enter that directory, and use its own `src` on `PYTHONPATH` for locked execution. The included code must be the code actually imported. The earlier scientific-revision archive has its own execution snapshot; do not rewrite its implementation hashes to match a later checkout.

The one-use study receipt remains at `examples/validation/external-v1/primary-evaluation-started.json`, independent of which manifest filename is supplied. A maintenance revision cannot reset the study. The lock tool refuses to create a revision after this receipt exists. Independent custodianship and registration are still required: local files are not tamper-proof escrow. A future change after outcome access requires a new study/cohort, not another version number attached to the same test.

## Maintenance record

| Finding                                                               | Correction                                                                                                            | Validation                                                                |
|-----------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| Formatting drift broke the execution lock                             | Canonical formatting, preserved original manifest, explicit pre-evaluation revision and execution snapshot            | File hashes; frozen-artifact regression checks                            |
| Duplicate RF/k-NN metric logic could diverge and omit classes         | One fixed-class evaluator with explicit undefined outcomes and legacy output keys                                     | Missing-class, nonlexicographic probability order and unknown-label tests |
| k-NN used unstratified small-sample folds and ignored test arguments  | Require supported stratified/grouped folds; reject unused test inputs, conflicting labels and oversized neighborhoods | Synthetic training-contract tests                                         |
| Perturbation function name overstated its calculation                 | Explicit mean absolute score change; compatible deprecated alias                                                      | Hand-calculated finite-input tests                                        |
| Review agreement depended on list ordering                            | Compare localization sets without forcing one label                                                                   | Reversed-order multilocalization test                                     |
| Overview/API/Methods mixed historical, internal and external evidence | Separate populations, fitted models, development selection and pending confirmation                                   | Documentation review and strict build                                     |

## Contribution and retention rules

- Inspect existing source, tests and study records before editing. Keep inference schemas and unrelated work stable unless a justified change is documented.
- Keep data acquisition, cohort construction, exploratory selection and final evaluation separate. Never rerun a held-out study to choose a favorable outcome.
- Record exact inputs, code/dependencies, seeds, group assignments, predictions, uncertainty method and failed experiments. State what is conditional or unknown.
- Preserve original model/data/result hashes. Archive or supersede records explicitly; do not delete unfavorable experiments or replace historical numbers during cleanup.
- Format/check `src`, `tests` and `scripts`; do not mass-format archived evidence. Tests should exercise meaningful contracts, not assert that every model has high accuracy on synthetic examples.
- Do not call software validation biological confirmation or make state-of-the-art claims from these experiments. Changes to code quality do not supply new science.

Run Python tests, Ruff, Black, mypy, strict documentation, the frontend checks and package build when changing their respective surfaces. Record environmental blockers separately from source failures. General installation dependency ranges and the scientific execution environment serve different purposes; see `REQUIREMENTS.md`in the repository root and the [installation guide](quickstart.md).
