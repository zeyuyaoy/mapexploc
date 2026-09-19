### Explainable Subcellular Localization Predictor

MAP-ExPLoc is a model-agnostic framework for explaining sequence-based protein-localization predictions. It wraps pretrained predictors, generates local and cohort-level Shapley Additive exPlanations (SHAP), and maps contributions to engineered descriptors or sequence regions with biological annotation overlays.

The validated workflows use a human Random Forest and native **DeepLoc 2.1 Fast**. The repository also includes a compact logistic-regression model selected through grouped internal validation for human annotation prediction.

> [!NOTE]
> This project won the [2025 ISCB YBS Student Challenge](https://www.iscb.org/ybs2025/programme-agenda/student-challenge) at [ISMB/ECCB 2025](https://www.iscb.org/ismbeccb2025/home).

#### Features

- Public adapters for pretrained localization models, with explicit class, preprocessing and checkpoint metadata
- Random Forest predictions for five human annotation classes: Cytoplasm, Membrane, Mitochondrion, Nucleus and Secreted
- DeepLoc Fast predictions across ten native multilabel compartments
- Signed TreeSHAP feature contributions and black-box sequence-region KernelSHAP
- Biological overlays with exact sequence matching, coordinate checks and annotation provenance
- Python, command-line and React interfaces, batch FASTA input, and complete JSON/CSV/HTML reports
- A 63-feature multinomial logistic model, similarity-grouped validation and saved per-protein research results

> [!WARNING]
> Explanations describe model behavior under a specified reference distribution, not causal biological mechanisms. Region SHAP is not individual-residue attribution. Accurate/ProtT5 integration exists, but real inference and native parity remain unqualified after a memory-preflight failure.

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

   This is a general installation. Reproducing saved results requires the [pinned model-specific environments](docs/research-software.md#environments-and-assets). DeepLoc additionally requires its licensed standalone package, external weights and isolated native runtime.

3. Start the prediction API:

   ```bash
   python -m uvicorn mapexploc.api:app --host 127.0.0.1 --port 8000
   ```

   Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for interactive requests. Repository startup selects the supplied Random Forest; the research logistic model uses a [separate inference workflow](docs/research-software.md#reproduce-the-evidence).

4. Optionally start the web interface in another terminal:

   ```bash
   pnpm --dir ui install --frozen-lockfile
   pnpm --dir ui dev
   ```

   The viewer supports configured native-model prediction and completed report import. Expensive DeepLoc explanations run through the [analysis CLI](docs/research-software.md#generate-and-inspect-a-report).

#### Requirements

- Python ≥3.12 for the framework; separate Python 3.11.15 runtime for DeepLoc
- Core scientific libraries: scikit-learn, SHAP, pandas, NumPy, SciPy and Biopython
- Matplotlib for optional plots: `python -m pip install -e '.[plots]'`
- Node 24 and pnpm 11.18.0 for the optional web interface

Dependencies are declared in [pyproject.toml](pyproject.toml); reproducibility locks and asset instructions are in the [reproduction guide](docs/research-software.md).

#### Feature Engineering

The Random Forest uses **423 features**:

- Amino-acid composition: 20 frequencies
- Overlapping dipeptide composition: 400 frequencies
- Sequence length, GRAVY hydrophobicity and isoelectric point

The research logistic model uses **63 features**: 23 global descriptors excluding dipeptides, plus amino-acid composition in the first and last 50 residues. Length is log-transformed and inputs standardized using training data only.

DeepLoc uses protein-language-model representations. Its region explanations perturb complete sequence intervals and recompute native inference; embedding dimensions are not presented as biological features.

#### Results

The human predictor study uses curated UniProtKB/Swiss-Prot sequences with experimental localization evidence:

| Model or procedure               | Evaluation                                                           | Macro-F1 |
|----------------------------------|----------------------------------------------------------------------|---------:|
| Served Random Forest             | Original 364-protein holdout                                         |   0.5223 |
| Nested model-selection procedure | Grouped internal validation on 1,741 proteins                        |   0.6094 |
| Selected logistic configuration  | Post-selection out-of-fold diagnostic on the same development cohort |   0.6006 |
|----------------------------------|----------------------------------------------------------------------|---------:|

> [!WARNING]
> These scores are not interchangeable. The matched forest comparator in nested validation scored 0.5462. Cytoplasm recall remains weak, and the saved human predictors are uncalibrated. They predict one curated annotation, not exclusive or multiple biological localizations; transfer beyond the studied cohort is unestablished. These results do not evaluate DeepLoc.

In the separate **30-protein DeepLoc Fast signal-peptide experiment**, positive extracellular attribution was higher in signal regions, but did not outperform the matched randomized-annotation control (**p = 0.0779**). One of three sensitivity checks was unstable. Broader biological validation and both-mode method comparison remain incomplete.

See [methods, results and limitations](docs/index.md) for data selection, leakage controls, uncertainty and interpretation, and [reproduction instructions](docs/research-software.md) for commands and frozen evidence.

#### License

Code is licensed under the [MIT License](LICENSE). UniProt-derived sequences and annotations are attributed to the UniProt Consortium under [CC BY 4.0](https://www.uniprot.org/help/license), with MAP-ExPLoc curation and transformations recorded alongside the data. DeepLoc program/model assets remain subject to upstream terms and are not redistributed.

#### Contributing

Open an issue or submit a pull request against `main`. Include relevant tests and document changes to data selection, model behavior, explanation references or evaluation so their scientific effects can be assessed.
