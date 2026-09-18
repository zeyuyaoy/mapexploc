# MAP-ExPLoc

Analyze protein sequences, compare localization probabilities and inspect feature-level explanations through a Python package, CLI, HTTP API and focused two-screen research interface.

## Start here

- [First analysis](quickstart.md): install, run the evaluated sample model and open the interface.
- [Interface guide](ui.md): FASTA batches, charts, accessible controls and exports.
- [Model card](model-card.md): measured performance and scientific limitations.
- [Reproduce the baseline](baseline.md): frozen data, provenance, grouping and evaluation.
- [Model comparison](model-improvement.md): executed Random Forest/Extra Trees experiments and the promotion decision.
- [Scientific revision](research-revision.md): completed internal validation and the selected 63-feature logistic research candidate.
- [External validation](external-validation.md): frozen study awaiting independent adjudication and an eligible external cohort; no external score or wet-lab outcome.
- [Research software standards](research-software.md): evidence status, metric contracts and maintenance/change-control instructions.

## Use the package

- [Python API](api.md)
- [HTTP requests and responses](reporting-schema.md)
- [Model adapters](adapter-guide.md)
- [SHAP explanation guide](explainer-guide.md)
- [Troubleshooting](troubleshooting.md)

The served tree-model interface uses a fixed 423-feature representation; the separate research logistic model uses 63 selected global/terminal features. Model probabilities are uncalibrated, and SHAP explains the tree model's engineered features rather than causal biological mechanisms. The included human baseline has a held-out macro-F1 of 0.522; it is a reproducible reference, not evidence of general-purpose localization accuracy.
