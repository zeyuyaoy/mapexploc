# MAP-ExPLoc

Analyze protein sequences, compare localization probabilities and inspect feature-level explanations through a Python
package, CLI, HTTP API and focused two-screen research interface.

## Start here

- [First analysis](quickstart.md): install, run the evaluated sample model and open the interface.
- [Interface guide](ui.md): FASTA batches, charts, accessible controls and exports.
- [Model card](model-card.md): measured performance and scientific limitations.
- [Reproduce the baseline](baseline.md): frozen data, provenance, grouping and evaluation.
- [Model improvement plan](model-improvement.md): proposed experiments and default-model promotion criteria.

## Use the package

- [Python API](api.md)
- [HTTP requests and responses](reporting-schema.md)
- [Model adapters](adapter-guide.md)
- [SHAP explanation guide](explainer-guide.md)
- [Troubleshooting](troubleshooting.md)

MAP-ExPLoc uses a fixed 423-feature representation. Model probabilities are uncalibrated, and SHAP explains the model's
engineered features rather than causal biological mechanisms. The included human baseline has a held-out macro-F1 of
0.522; it is a reproducible reference, not evidence of general-purpose localization accuracy.
