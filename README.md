# MAP-ExPLoc

An explainable protein subcellular-localization toolkit for research and education, originating in the 2025 ISCB YBS
Student Challenge. Its Python package, CLI, FastAPI service and React interface share a deterministic **423-feature
schema**.

## Start here

Use **Python 3.12+**, **Node 24**, and **pnpm 11.18.0**. Follow the [quickstart](docs/quickstart.md) to install the
package, configure the evaluated model and start the interface. Choose **Try an example → Analyze sequences**, then
explore Prediction, Sequence and Explanation tabs.

Repository startup selects the evaluated human baseline through a checksum-verified `config/default-model.json`.
`MAPEXPLOC_MODEL_PATH` explicitly overrides it. Installed wheels require a trusted model path; the UI never selects or
downloads a model itself.

## Scientific reference

The supplied Random Forest was trained on 1,458 curated human proteins and tested on 364, with similarity groups kept
apart. Held-out macro-F1 is **0.522** versus **0.086** for a prior dummy classifier. Cytoplasm and Secreted remain weak
classes. Probabilities are uncalibrated; SHAP describes engineered features, not causal biological mechanisms. This is a
demonstration baseline, not a validated biological localization service.

The expanded-data comparison reached **0.600** macro-F1 on the same previously inspected benchmark. It lacked
independent confirmation and exceeded latency gates, so the served default remains version 1.

The subsequent internal study selected a **63-feature logistic-regression research candidate**, with fixed-configuration
development macro-F1 **0.6006**. This is a different population and evaluation design from the historical scores above;
do not subtract them to estimate improvement. The exact fitted candidate is frozen for a single external study.
Independent annotation adjudication, external scoring and new wet-lab confirmation remain pending. The service still
uses the forest.

- [Model card](docs/model-card.md): scope, measured performance and limitations.
- [Baseline reproduction](docs/baseline.md): frozen data, provenance and scientific environment.
- [Model comparison](docs/model-improvement.md): executed CPU experiments, results and promotion decision.
- [Scientific revision](docs/research-revision.md): nested validation, simpler baselines, calibration, ablations, cohort
  audit and reproducible research outputs.
- [External validation study](docs/external-validation.md): frozen logistic model, blinded adjudication, independence
  screening and one primary falsification test.

## Guides

- [Documentation index](docs/index.md) and [dependency profiles](REQUIREMENTS.md).
- [CLI smoke test](docs/quickstart.md#smoke-test-the-cli) and [Python API](docs/api.md).
- [HTTP schema](docs/reporting-schema.md), [interface guide](docs/ui.md) and [troubleshooting](docs/troubleshooting.md).

Small offline examples ship in package distributions. The curated scientific dataset and model remain repository assets.
Only the 20 standard amino acids are accepted; whitespace and case are normalized.

## Development checks

Install the development profile described in [dependency profiles](REQUIREMENTS.md), then run:

```bash
python -m pytest
python -m ruff check src tests scripts
python -m black --check src tests scripts
python -m mypy src
python -m mkdocs build --strict
pnpm --dir ui check
uv build
```

Ordinary tests use small synthetic fits and selection workflows; they do not download scientific data or launch the full
scientific experiments. Generated outputs belong under `artifacts/` or `results/`; curated reference assets under
`examples/` remain versioned.

See [research software standards](docs/research-software.md) for evidence status, metric conventions, artifact change
control and the maintenance record.

## License

Code: [MIT](LICENSE). UniProt-derived data: UniProt Consortium, [CC BY 4.0](https://www.uniprot.org/help/license). See
the model card for attribution and source hashes.
