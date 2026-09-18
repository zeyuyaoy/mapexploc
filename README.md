### Explainable Subcellular Localization Predictor

MAP-ExPLoc is a sequence-based machine-learning pipeline for predicting human protein subcellular-localization annotations. It combines classical feature engineering with 2 interpretable models: a Random Forest with Shapley Additive exPlanations (SHAP) for interactive prediction, and a compact logistic-regression model selected through grouped internal validation for research use.

> [!NOTE]
> This project won the [2025 ISCB YBS Student Challenge](https://www.iscb.org/ybs2025/programme-agenda/student-challenge) at [ISMB/ECCB 2025](https://www.iscb.org/ismbeccb2025/home).

#### Features

- Five subcellular-localization classes: Cytoplasm, Membrane, Mitochondrion, Nucleus and Secreted
- Curated human UniProtKB/Swiss-Prot sequences with experimental localization evidence
- Random Forest predictions with signed, per-feature SHAP contributions
- A 63-feature multinomial logistic model for computational research
- Python, command-line and web interfaces, including batch FASTA input and CSV/JSON exports
- Reproducible feature extraction, similarity-grouped validation and saved per-protein results

> [!WARNING]
> Explanations describe feature contributions, not causal sequence motifs.

#### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/zeyuyaoy/mapexploc.git
   cd mapexploc
   ```

2. Create and activate the Conda environment:

   ```bash
   conda env create --file environment.yml
   conda activate mapexploc
   ```

   Alternatively, install with Python 3.12 or newer:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install -e .
   ```

3. Start the prediction API:

   ```bash
   python -m uvicorn mapexploc.api:app --host 127.0.0.1 --port 8000
   ```

   Open `http://127.0.0.1:8000/docs` for interactive requests. Repository startup
   selects the supplied Random Forest; the research logistic model has a separate
   [inference workflow](docs/research-software.md#use-the-fitted-models).

#### Requirements

- Python 3.12 or newer
- Core scientific libraries: scikit-learn, SHAP, pandas, NumPy, SciPy and Biopython
- Matplotlib for optional plots: `python -m pip install -e '.[plots]'`
- Node 24 and pnpm 11.18.0 for the optional web interface

Installation resolves dependencies from [pyproject.toml](pyproject.toml). Note that reproducing recorded scientific results requires the [study-specific environments](docs/research-software.md).

#### Feature Engineering

The Random Forest uses **423 features**:

- Amino-acid composition: 20 frequencies
- Overlapping dipeptide composition: 400 frequencies
- Sequence length, GRAVY hydrophobicity and isoelectric point

The research logistic model uses **63 features**: the 23 global descriptors above, excluding dipeptides, plus amino-acid composition in the first and last 50 residues. Length is log-transformed and inputs standardized using training data only.

#### Results

| Model or procedure               | Evaluation                                                           | Macro-F1 |
|----------------------------------|----------------------------------------------------------------------|---------:|
| Served Random Forest             | Original 364-protein holdout                                         |   0.5223 |
| Nested model-selection procedure | Grouped internal validation on 1,741 proteins                        |   0.6094 |
| Selected logistic configuration  | Post-selection out-of-fold diagnostic on the same development cohort |   0.6006 |

> [!WARNING]
> Scores are not interchangeable. The matched forest comparator in nested validation scored 0.5462. Terminal composition contributes predictive information, but Cytoplasm recall remains weak and probabilities are uncalibrated.
> Additionally, the models predict one curated annotation, not exclusive or multiple biological localizations; transfer beyond the studied cohort is unestablished.

See [methods, results and limitations](docs/index.md) and [reproduction instructions](docs/research-software.md) for the data, leakage controls, uncertainty estimates and commands.

#### License

Code is licensed under the [MIT License](LICENSE). UniProt-derived sequences and annotations are attributed to the UniProt Consortium under [CC BY 4.0](https://www.uniprot.org/help/license).

#### Contributing

Open an issue for a bug report or proposed feature, or submit a pull request against `main`. Include relevant tests and document changes to data selection, features or evaluation so their scientific effects can be assessed.
