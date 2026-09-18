# MAP-ExPLoc

An explainable protein subcellular-localization toolkit for research and education,
originating in the 2025 ISCB YBS Student Challenge. Its Python package, CLI,
FastAPI service and React interface share a deterministic **423-feature schema**.

## Start here

Use **Python 3.12+**, **Node 24**, and **pnpm 11.18.0**.
Follow the [quickstart](docs/quickstart.md) to install the package, configure the
evaluated model and start the interface. Choose **Try an example → Analyze
sequences**, then explore Prediction, Sequence and Explanation tabs.

The API currently defaults to a local `model.pkl` when `MAPEXPLOC_MODEL_PATH` is
unset. The quickstart explicitly selects the evaluated human baseline instead.
The UI never selects or downloads a model itself.

## Scientific reference

The supplied Random Forest was trained on 1,458 curated human proteins and tested
on 364, with similarity groups kept apart. Held-out macro-F1 is **0.522** versus
**0.086** for a prior dummy classifier. Cytoplasm and Secreted remain weak classes.
Probabilities are uncalibrated; SHAP describes engineered features, not causal
biological mechanisms. This is a demonstration baseline, not a validated
biological localization service.

- [Model card](docs/model-card.md): scope, measured performance and limitations.
- [Baseline reproduction](docs/baseline.md): frozen data, provenance and scientific environment.
- [Model improvement plan](docs/model-improvement.md): proposed CPU experiments and promotion gates.

## Guides

- [Documentation index](docs/index.md) and [dependency profiles](REQUIREMENTS.md).
- [CLI smoke test](docs/quickstart.md#smoke-test-the-cli) and [Python API](docs/api.md).
- [HTTP schema](docs/reporting-schema.md), [interface guide](docs/ui.md) and [troubleshooting](docs/troubleshooting.md).

Small offline examples ship in package distributions. The curated scientific
dataset and model remain repository assets. Only the 20 standard amino acids are
accepted; whitespace and case are normalized.

## Development checks

Install the development profile described in [dependency profiles](REQUIREMENTS.md), then run:

```bash
python -m pytest
python -m ruff check src tests
python -m black --check src tests
python -m mypy src
python -m mkdocs build --strict
pnpm --dir ui check
uv build
```

Ordinary tests do not download scientific data or run model selection. Generated
outputs belong under `artifacts/` or `results/`; curated reference assets under
`examples/` remain versioned.

## License

Code: [MIT](LICENSE). UniProt-derived data: UniProt Consortium,
[CC BY 4.0](https://www.uniprot.org/help/license). See the model card for attribution
and source hashes.
