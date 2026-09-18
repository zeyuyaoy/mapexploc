# Dependency profiles

Python 3.12 or newer is required. `pyproject.toml` is the authoritative dependency declaration.

- `pip install -e .`: prediction, training, CLI, API and JSON SHAP explanations.
- `pip install -e ".[plots]"`: optional SHAP summary plots.
- `pip install -e ".[dev,docs,plots]"`: tests, checks, documentation and plotting.

The three requirements files are thin wrappers for these profiles, respectively
`requirements-minimal.txt`, `requirements.txt`, and `requirements-dev.txt`.
Run them from the repository root. `conda env create -f environment.yml` installs
the development profile through pip in a Python 3.12 Conda environment.

Historical notebooks and benchmark snapshots have been removed; committed versions
remain in Git history. Current reproducible scientific assets live in `examples/baseline`
and `examples/models`. Generated results are ignored.

The frontend requires Node 24 and pnpm 11.18.0. Use `pnpm install --frozen-lockfile`
inside `ui`; `pnpm check` verifies generated examples, formatting, lint, tests and build.

The original baseline scientific runtime is recorded in
`examples/models/scientific-environment.txt`. It preserves the measured artifact's
library versions separately from general installation requirements. Do not replace
or refit that artifact merely to update dependencies.

MkDocs remains on its 1.x plugin API for compatibility with Material and
mkdocstrings; strict documentation builds validate that combination.
