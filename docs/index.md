### Localizing the human protein

With MAP-ExPLoc, we ask whether sequence composition predicts one experimentally supported structured localization annotation for a canonical human protein. The target is **annotation prediction**, not exclusive localization, simultaneous localization in several compartments, or experimental confirmation. The completed study uses grouped internal validation; its scope does not include a required prospective validation phase.

See [Reproduction](research-software.md) for execution instructions.

#### Data and labels

Both source downloads contain the same 20,431 reviewed human UniProtKB/Swiss-Prot entries, release **2026_03**, retrieved 18 September 2026. Expansion of the dataset removed sampling caps; it did not add an independent release. Source SHA-256: `14be5581ec2412c7da60889232c2b9922b98373ef5a76d1350c460054447e967`.

Curation requires a canonical accession, a sequence containing only the 20 standard residues, one mapped structured class, and at least one `ECO:0000269` evidence item attached to a mapped location. Protein-existence evidence alone is insufficient.

The exact accepted location strings are:

| Class         | UniProt location strings       |
|---------------|--------------------------------|
| Cytoplasm     | Cytoplasm, Cytosol             |
| Membrane      | Cell membrane, Plasma membrane |
| Mitochondrion | Mitochondrion                  |
| Nucleus       | Nucleus                        |
| Secreted      | Secreted                       |

Any unmapped structured location, conflicting mapped class, or isoform/product-specific location comment excludes the record. Generic membrane is not plasma membrane. Identical sequences are deduplicated; conflicting duplicate labels are excluded. Missing localization is not a negative label and is not imputed.

Curation accepts **2,107 proteins (10.31%)**. First-failure exclusions are 9,462 unmapped locations, 3,763 without experimental localization evidence, 3,089 without location, 1,342 conflicting locations, 664 isoform/product-specific records, and four unsupported sequences. The latter are selenoproteins containing U, not invalid biology. Free-text notes do not determine acceptance: some accepted proteins have additional localizations described only in notes.

The original baseline caps each class at 500 using seed 42: 1,822 proteins, split into 1,458 training and 364 test proteins. The expanded cohort excludes all 364 historical test accessions and two additional related proteins, leaving **1,741 development proteins in 1,510 sequence groups**. No all-class independent confirmation set was available. Frozen CSVs retain accessions, sequences, labels, experimental evidence, groups and partitions; the expanded CSV also retains localization comments. The cohort audit found no missing required fields, nonfinite features, duplicate accessions/sequences or recorded partition overlaps.

#### Relatedness and validation

MMseqs2 **18-8cc5c** defines connected sequence groups at ≥30% identity and ≥80% coverage of both sequences. Expanded grouping uses the union of original and accepted expanded proteins. Historical-related groups are excluded from every new fit; bidirectional development–historical searches found zero qualifying hits. Curation-rejected source proteins were not bridge nodes. These controls do not exclude remote/domain homology or dependence from shared experiments.

The internal study uses three outer stratified group folds, repeated with seeds 20260918 and 20260919, and three inner folds per outer training set. Each fold has all classes and disjoint groups. Candidate selection, scaling and temperature fitting use only outer-training data. The original forest configuration is refitted on the same outer training rows as the comparator. The previously inspected 364-protein benchmark is never fresh confirmation, and prior familiarity with the development cohort limits even nested validation to internal evidence.

Sensitivity analyses repeat nested selection once after excluding localization-note records (1,428 proteins), and once joining sequence groups through shared supporting PubMed IDs (1,248 groups). There are 313 note-bearing development proteins; 67 supporting publications overlap development and historical-related records, and one publication supports 51 of 154 development mitochondrial labels. Publication grouping addresses only recorded links; absence of notes does not establish exclusive localization.

#### Features and model selection

Whitespace is removed and case normalized; unsupported residues, gaps and empty sequences are rejected. The 423-feature representation is length, 20 amino-acid frequencies, 400 overlapping dipeptide frequencies normalized by `max(length−1, 1)`, GRAVY and Biopython isoelectric point. The 23-feature subset omits dipeptides; the 63-feature subset adds amino-acid frequencies in the first and last 50 residues. Each terminal window uses the whole sequence if shorter; windows can overlap.

Fifteen candidates are compared: prior dummy; length-only logistic regression; L2 multinomial logistic regression with C ∈ {0.01, 0.1, 1} on each of the three feature sets; and four forests. Forest configurations are specified in `examples/experiments/research-revision/primary/protocol.json`. Linear models use `log1p(length)` and training-fitted standardization. There is no imputation, oversampling, augmentation or learned feature selection.

Candidates are ranked by pooled inner out-of-fold five-class macro-F1. Among those within 0.01 F1 of the leader, no more than 0.02 worse log loss and no class F1 more than 0.05 worse, select the least complex. The order is dummy, length-only, global/terminal/full logistic, global/terminal/original/expanded forest; remaining ties use smaller C then candidate ID. These tolerances are operational choices, not biological effect thresholds.

Full-development selection chooses **63-feature logistic regression, C=0.1**:L2, lbfgs, max_iter=3000, tol=1e-4, intercept, no class weighting, seed 20260918. The final model is fitted to all 1,741 development proteins. Stored scaling and softmax/argmax determine predictions; class order follows the table above. Convergence warnings fail fitting. Scalar temperature calibration (bounded 0.5–3)was fitted to inner out-of-fold predictions. Retention required a ≥0.02 outer log-loss reduction and a paired 97.5% interval excluding zero. An exploratory post-selection check for the exact logistic configuration failed this gate; **the final temperature is 1**.

#### Estimands, uncertainty and results

The two declared internal comparisons are adaptive selection minus refitted original forest in macro-F1, and calibrated minus raw adaptive selection in log loss. Metrics pool out-of-fold predictions across repeats. Paired percentile bootstrap resamples whole groups, retaining every protein and both repeat predictions: 2,000 draws, seed 20260918; 97.5% intervals for the two comparisons and descriptive 95% intervals for individual scores. There are 1,741 proteins, not 3,482 independent observations. Intervals condition on fitted predictions; they omit complete training/selection uncertainty and unrecorded dependence.

Macro-F1 always includes all five classes, with zero for zero-denominator F1; AUROC/AP are undefined without both outcomes. AP is average precision. Multiclass Brier is the mean sum of squared class errors (0–2); ECE uses ten equal-width bins of maximum probability against correctness. Neither low ECE nor proper scores alone establishes calibration.

On identical development proteins and outer folds:

| Procedure/configuration                       | Macro-F1 | Log loss |  Brier |
|-----------------------------------------------|---------:|---------:|-------:|
| Prior dummy                                   |   0.1120 |   1.4714 | 0.7420 |
| Original forest refitted                      |   0.5462 |   1.1676 | 0.5885 |
| Adaptive selection, raw                       |   0.6094 |   0.9684 | 0.4990 |
| Adaptive selection, temperature-scaled        |   0.6094 |   0.9089 | 0.4743 |
| Final logistic configuration, descriptive OOF |   0.6006 |   0.8964 | 0.4711 |

Adaptive macro-F1 is 0.6094 (95% interval 0.5847–0.6323); its paired gain is **0.0632 (97.5% interval 0.0357–0.0907)**. Selection chose terminal forests in five outer folds and terminal logistic in one. The final logistic row is a post-selection diagnostic, not an independent estimate for the final fitted artifact. Adaptive calibration reduced loss by 0.0595, whereas exact-logistic calibration reduced it by only 0.0020 (calibrated-minus-raw interval −0.0109 to 0.0062); the latter was rejected. Final calibration and stronger label-permutation checks were added after inspecting earlier results.

The adaptive gain persists with publication-linked grouping (0.5710 versus 0.4813) and without localization notes (0.6234 versus 0.5483), but these are sensitivity analyses, not additional confirmation.

| Final logistic class | Unique proteins | Descriptive OOF F1 | Recall |
|----------------------|----------------:|-------------------:|-------:|
| Cytoplasm            |             316 |             0.3698 | 0.3212 |
| Membrane             |             406 |             0.7098 | 0.7426 |
| Mitochondrion        |             154 |             0.6332 | 0.6136 |
| Nucleus              |             677 |             0.7477 | 0.7954 |
| Secreted             |             188 |             0.5428 | 0.4973 |

The logistic model is wrong in both repeats for 508 proteins, including 69 with mean maximum probability >0.8. No labels were repaired to improve these scores.

Separately, the served 128-tree forest scored **0.5223** on the original 364-protein holdout (prior dummy 0.0862). Its training-only three-fold grouped search selected depth 24, minimum leaf size 3 and balanced class weights. This historical score is not a matched comparator for the table above.

#### Biological interpretation and limits

Global and N-terminal composition carry predictive information in this selected cohort. Removing 25 N-terminal residues reduced the selected models' mean fold F1 from 0.6083 to 0.4667. This supports sensitivity to intact terminal sequence, not discovery of targeting motifs or a causal mechanism. Coefficients, permutation importance and tree SHAP are model associations; none supplies experimental localization evidence.

Structured annotations do not establish exclusive localization. Notes can identify additional compartments; the cohort lacks systematic gene, tissue, assay, condition and fragment-status metadata. Canonical precursor input may differ from the assayed mature protein. Selection for available evidence, class caps in the baseline, and study dependence prevent proteome-wide prevalence or accuracy claims. Cytoplasm recall and confident errors remain substantial weaknesses.

The model forces one of five classes and has no validated out-of-scope detection, abstention or multilabel capability. Other species, isoforms, fragments and other compartments are unsupported. Reported performance is internal to the curated cohort; transfer to independently sampled populations and experimental localization have not been established. These are limits on interpretation, not outstanding project milestones. Extensions should retain annotation uncertainty and sequence/assay dependence controls, and distinguish exploratory selection from held-out evaluation.
