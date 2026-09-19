### Explainable Subcellular Localization Predictor

MAP-ExPLoc is a model-agnostic framework for explaining sequence-based protein-localization predictions. It wraps pretrained predictors, generates local and cohort-level Shapley Additive exPlanations (SHAP), and maps contributions to engineered descriptors or sequence regions with biological annotation overlays.

Supported workflows include a human Random Forest and native **DeepLoc 2.1 Fast**, which has passed native prediction parity checks. A separate **63-feature logistic-regression model** was selected through grouped internal validation for human annotation prediction.

> [!NOTE]
> This project won
> the [2025 ISCB YBS Student Challenge](https://www.iscb.org/ybs2025/programme-agenda/student-challenge)
> at [ISMB/ECCB 2025](https://www.iscb.org/ismbeccb2025/home).

#### Features

- Adapters with explicit class, preprocessing and checkpoint metadata
- Random Forest predictions for five human annotation classes: Cytoplasm, Membrane, Mitochondrion, Nucleus and Secreted
- Local/research DeepLoc Fast inference across ten native multilabel compartments
- Signed TreeSHAP feature contributions and sequence-region KernelSHAP
- Biological overlays with exact sequence matching, coordinate checks and annotation provenance
- Python, CLI and React interfaces, batch FASTA input, and complete JSON/CSV/HTML reports

The public website serves **MAP-ExPLoc-owned models only**, currently the bundled Random Forest. DeepLoc runs through local/research workflows with separately obtained assets; the website can display completed reports without executing their source models. See [local development and Vercel deployment](docs/research-software.md#local-interface-and-public-deployment).

> [!WARNING]
> Explanations describe model behavior under a specified reference distribution, not causal biological mechanisms.
> Region SHAP is not individual-residue attribution. Accurate/ProtT5 integration exists, but inference and native parity
> remain unqualified after a memory-preflight failure.

#### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/zeyuyaoy/mapexploc.git
   cd mapexploc
   ```

2. Install with Python 3.12 or newer:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install -e .
   ```

   This is a general installation. Saved scientific artifacts require the [pinned model-specific environments](docs/research-software.md#environments-and-assets). DeepLoc additionally requires its licensed standalone package, external weights and separate Python 3.11.15 worker.

3. Start the local prediction API:

   ```bash
   python -m uvicorn mapexploc.api:app --host 127.0.0.1 --port 8000
   ```

   Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for request schemas. Repository startup selects the bundled Random Forest; the research logistic model has a [separate inference workflow](docs/research-software.md#reproduce-the-evidence).

4. Optionally start the web interface in another terminal, using Node 24 and pnpm 11.18.0:

   ```bash
   pnpm --dir ui install --frozen-lockfile
   pnpm --dir ui dev
   ```

   The interface supports owned-model prediction, feature explanations and completed report import. Use the [analysis CLI](docs/research-software.md#reports-and-extension) for DeepLoc inference and explanations.

Dependencies are declared in [pyproject.toml](pyproject.toml). Optional plots require `python -m pip install -e '.[plots]'`. The Vercel runtime is pinned separately to Python 3.12.

#### Feature Engineering

The Random Forest uses **423 features**: 20 amino-acid frequencies, 400 overlapping dipeptide frequencies, sequence length, GRAVY hydrophobicity and isoelectric point.

The research logistic model uses **63 features**: 23 global descriptors excluding dipeptides, plus amino-acid composition in the first and last 50 residues. Length is log-transformed and inputs standardized using training data only.

DeepLoc uses protein-language-model representations. Region explanations perturb sequence intervals and recompute native inference; embedding dimensions are not treated as biological features.

#### Results

The human predictor study uses curated UniProtKB/Swiss-Prot sequences with experimental localization evidence.

| Model or procedure               | Evaluation                                                           | Macro-F1 |
|----------------------------------|----------------------------------------------------------------------|---------:|
| Served Random Forest             | Original 364-protein holdout                                         |   0.5223 |
| Nested model-selection procedure | Grouped internal validation on 1,741 proteins                        |   0.6094 |
| Selected logistic configuration  | Post-selection out-of-fold diagnostic on the same development cohort |   0.6006 |

> [!WARNING]
> These scores are not interchangeable. The matched forest comparator in nested validation scored 0.5462. Cytoplasm
> recall remains weak, and the saved human predictors are uncalibrated. They predict one curated annotation, not exclusive
> or multiple biological localizations. Independent transfer is unestablished; these results do not evaluate DeepLoc.

In the separate **30-protein DeepLoc Fast signal-peptide experiment**, positive extracellular attribution was higher in signal regions but did not outperform the matched randomized-annotation control (**p = 0.0779**). One of three sensitivity checks was unstable; the remaining 27 proteins were unassessed. Broader biological validation and both-mode method comparison remain incomplete.

See [methods and evidence](docs/index.md) for selection, leakage controls, uncertainty and interpretation, and [reproduction instructions](docs/research-software.md) for commands and frozen evidence.

#### License

Code: [MIT](LICENSE). UniProt-derived sequences and annotations: UniProt Consortium, [CC BY 4.0](https://www.uniprot.org/help/license), with curation recorded alongside the data. DeepLoc assets remain subject to upstream terms and are not redistributed.

#### Contributing

Open an issue or submit a pull request. Include relevant tests and document changes that affect data selection, model behavior, explanation references or scientific interpretation.
