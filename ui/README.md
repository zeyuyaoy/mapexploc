# MAP-ExPLoc UI

A React/Vite research interface with separate sequence-entry and results screens. Supports single sequences and FASTA
batches, comparison, probability/composition/SHAP charts, model provenance and CSV/JSON downloads.

Requires **Node 24** and **pnpm 11.18.0**.

```bash
pnpm install --frozen-lockfile
pnpm dev
pnpm check
```

From the repository root, run the API in another terminal:

```bash
MAPEXPLOC_MODEL_PATH=examples/models/human-baseline.joblib \
  python -m uvicorn mapexploc.api:app --host 127.0.0.1 --port 8000
```

`MAPEXPLOC_MODEL_PATH` selects the model in the Python API; it is not a frontend setting. Vite proxies API routes to
`http://127.0.0.1:8000` by default. Set `MAPEXPLOC_API_PROXY` when starting Vite to change that development target, or
`VITE_API_BASE_URL` at build time for another API origin. Prefer a same-origin reverse proxy in production.

`pnpm check` verifies example synchronization, formatting, lint, offline interaction/parser tests and the production
build. `pnpm test` runs Vitest. The esbuild installation script is explicitly allowed in `pnpm-workspace.yaml`; no broad
script approval is required.

See [UI guide](../docs/ui.md), [quickstart](../docs/quickstart.md), and [model card](../docs/model-card.md). Source
examples are attributed to UniProt (CC BY 4.0) and taken from the frozen held-out partition; no input is sent to UniProt
by the UI.

Optional browser regression suite (with API and Vite running):

```bash
pnpm exec playwright install chromium
pnpm test:e2e
```

Use `MAPEXPLOC_UI_URL` for another local UI port. `PLAYWRIGHT_CHROMIUM_EXECUTABLE` optionally selects an already
installed test Chromium. These tests verify actual downloaded CSV/JSON contents, FASTA upload, on-demand SHAP, keyboard
controls and mobile overflow.

Examples are generated from `src/mapexploc/examples/proteins.json` at the repository root. After editing that canonical
file, run `pnpm examples:sync`. ESLint 10 tracks JSX references natively; capitalized unused variables are no longer
excluded from lint. Vite uses the automatic JSX runtime.
