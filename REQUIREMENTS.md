# Dependency profiles

Python 3.12 or newer is required. `pyproject.toml` is the authoritative dependency declaration.

- `pip install -e .`: prediction, training, CLI, API and JSON SHAP explanations.
- `pip install -e ".[plots]"`: optional SHAP summary plots.
- `pip install -e ".[dev,docs,plots]"`: tests, checks, documentation and plotting.

The three requirements files are thin wrappers for these profiles, respectively `requirements-minimal.txt`,
`requirements.txt`, and `requirements-dev.txt`. Run them from the repository root. `conda env create -f environment.yml`
installs the development profile through pip in a Python 3.12 Conda environment.

Retired notebooks remain in Git history. Scientific assets are organized as follows:

- `examples/baseline` and `examples/models`: original curated dataset and served forest.
- `examples/experiments/human-v2`: expanded tree comparison.
- `examples/experiments/research-revision`: completed internal study, execution
  snapshot and fitted logistic candidate.
- `examples/validation/external-v1`: external-study protocol, blinded forms,
  freeze manifests and software checks; no external biological results.

Generated working runs are ignored. Preserve source snapshots and old manifests; never replace scientific evidence with
regenerated output carrying the same name.

The frontend requires Node 24 and pnpm 11.18.0. Use `pnpm install --frozen-lockfile` inside `ui`; `pnpm check` verifies
generated examples, formatting, lint, tests and build.

The original baseline scientific runtime is recorded in `examples/models/scientific-environment.txt`. It preserves the
measured artifact's library versions separately from general installation requirements. Do not replace or refit that
artifact merely to update dependencies.

The expanded-data CPU comparison records its exact environment separately in
`examples/experiments/human-v2/scientific-environment.txt`, including transitive dependencies, with hardware and MMseqs2
versions in `execution-environment.json`. Version 1 predictions and SHAP additivity were regression-checked under the
newer scikit-learn 1.9.1 environment; the original artifact and its 1.9.0 environment record remain unchanged.
Cross-version pickle warnings are expected and are not a general portability guarantee. See
the [comparison report](docs/model-improvement.md).

MkDocs remains on its 1.x plugin API for compatibility with Material and mkdocstrings; strict documentation builds
validate that combination.

The external-study execution lock requires its exact recorded Python and inference dependency versions; general package
minimums are not an exact-reproduction environment. Use the active manifest and execution snapshot specified
in [research software standards](docs/research-software.md). Dependency upgrades do not authorize retraining the frozen
logistic model or repeating the external test.
