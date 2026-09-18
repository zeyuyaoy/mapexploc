# Contributing

Read [research software standards](docs/research-software.md) before changing data, models, evaluation or scientific
claims. Preserve archived studies and unrelated work. An existing score is evidence about its recorded cohort and
procedure only.

Install the development profile from [REQUIREMENTS.md](REQUIREMENTS.md). Run:

```bash
python -m ruff check src tests scripts
python -m black --check src tests scripts
python -m mypy src
python -m pytest
python -m mkdocs build --strict
pnpm --dir ui check
uv build
```

Use small synthetic fixtures for software regressions. Do not download scientific data, refit reference artifacts or run
the external test as part of ordinary CI. Describe the problem, scientific consequence, change and relevant validation
in each contribution. Distinguish computational reproducibility from external validity.
