# Scientific revision: frozen development protocol

Protocol prepared on 2026-09-18 before running the new comparisons. The starting
point is the existing, uncommitted `human-v2` study. Its files and served version 1
artifact are preserved. This revision uses only its development proteins. The
historical benchmark has already influenced earlier work and is not a new test.
Even nested cross-validation on these previously studied proteins is internal
validation, not independent confirmation.

## Prioritized changes and decision gates

| Priority    | Change / problem                                                                                                                  | Rationale and expected benefit                                                                    | Cost or failure mode                                                            | Test / decision gate                                                                                                                                            |
|-------------|-----------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Essential   | Add nested, similarity-grouped validation around selection; the earlier 48-configuration search reports selection scores          | Keep every outer validation group outside preprocessing, selection and calibration                | Smaller inner training sets; prior development familiarity remains              | Three outer folds, two fixed seeds, three inner folds; assert class coverage, disjoint groups and complete out-of-fold coverage; never read historical outcomes |
| Essential   | Use fixed-class, validated probability metrics; current macro-F1 silently changes its denominator when classes disappear          | Comparable subgroup/bootstrap metrics; fail on invalid probability matrices                       | Missing-class subgroup AUROC is undefined                                       | Hand-calculated metric tests, explicit nulls for undefined AUC, full five-class F1                                                                              |
| Essential   | Add simple learned comparators; earlier search only compares tree families                                                        | Determine whether nonlinear complexity is justified on identical folds                            | Linear misspecification; correlated compositional features                      | Dummy, length-only and regularized multinomial logistic models; deterministic selection rule below                                                              |
| Essential   | Audit cohort, exact duplicates, missingness and provenance                                                                        | Prevent technical errors and distinguish selected annotation availability from biological absence | Annotation bias cannot be repaired by imputation                                | Zero nonfinite features or accession/sequence duplication; no group/accession/sequence overlap with reserved partitions; retain source hashes                   |
| Essential   | Separate research selection from production promotion and uncertainty claims                                                      | Avoid deploying a selected winner as independently validated                                      | Stronger evidence may require new data                                          | No promotion without independent confirmation; require a positive paired group-bootstrap lower bound as well as existing effect-size/class/runtime gates        |
| Recommended | Compare scalar temperature calibration using inner out-of-fold predictions                                                        | Adjust probability sharpness without changing class decisions                                     | Inner model-selection reuse and training-size shift can bias temperature        | Evaluate on untouched outer folds; retain only if log-loss reduction >=0.02 and the paired 97.5% interval excludes zero; no claim of universal calibration      |
| Recommended | Ablate dipeptides; add simple terminal amino-acid composition                                                                     | Whole-sequence composition loses position; terminal signals are a plausible localization cue      | 50-residue windows are coarse and cannot identify a signal peptide or mechanism | Predeclare global 23, full 423, and global-plus-terminal 63 features; same folds and regularization grid; no window tuning                                      |
| Recommended | Quantify error, length/family subgroup performance, seed stability, partial-sequence sensitivity and permutation failure controls | Reveal artifacts and fragile improvements                                                         | Small subgroups and simulated truncation are not external validation            | Flag seed macro-F1 range >0.05, subgroup support <50, and subgroup decline >0.10; these trigger qualification, not retuning                                     |
| Recommended | Preserve predictions, split memberships, software/code hashes, calibration and fitted parameters                                  | Permit independent reanalysis and exact resumption                                                | Serialized models require a trusted matching environment                        | Reject modified completed results/configuration/source; reload parity check; repeat seeds and verify cached-result integrity                                    |
| Optional    | Embeddings, deep networks, broad hyperparameter search or ensembles                                                               | May capture remote sequence context                                                               | Compute cost, pretrained contamination, further selection bias                  | Defer until independent data and a clear failure hypothesis justify them                                                                                        |
| Optional    | Temporal/site/publication holdouts and prospective biological experiments                                                         | Test actual transportability and annotation artifacts                                             | Required metadata/data are incomplete                                           | Document unavailable evidence; do not invent patient, site or clinical claims                                                                                   |

## Locked model comparison

- Source: `examples/experiments/human-v2/dataset.csv`, rows explicitly marked
  `development`; all other partitions are reserved. Similarity groups are the
  existing union connected components (30% identity, 80% bidirectional coverage).
- Outer split seeds: 20260918 and 20260919; three stratified grouped folds each.
  Inner splits: three folds, seed `outer_seed + fold + 100`. Estimator seed is the
  outer split seed. Final development selection uses split seed 20261018 and
  estimator seed 20260918.
- Features: unchanged 423 features; global subset = length, 20 amino-acid
  frequencies, GRAVY and isoelectric point; terminal subset adds frequencies in
  the first and last 50 residues (whole sequence if shorter). Log-transform
  length using `log1p` for linear models; fit standardization inside each fold.
- Logistic candidates: L2 multinomial logistic regression, C in {0.01, 0.1, 1},
  unweighted classes, maximum 3000 iterations. Length-only uses C=1. Prior dummy
  estimates training priors. No synthetic oversampling.
- Tree candidates: original RF (128, depth 24, leaf 3, sqrt, balanced); previous
  RF candidate (512, unlimited depth, leaf 1, feature fraction 0.5, balanced);
  compact RF on global and terminal features (128, depth 24, leaf 3, feature
  fraction 0.5, balanced). Each estimator uses one worker.
- Score candidates using pooled inner out-of-fold five-class macro-F1. Among
  candidates within 0.01 of the best F1, within 0.02 log loss and no class F1
  decline >0.05 relative to that leader, choose the least complex: dummy,
  length, global logistic, terminal logistic, full logistic, compact global RF,
  compact terminal RF, original RF, expanded RF. Break ties by smaller C then ID.
  These are pragmatic predeclared tolerances, not established biological minima.
- Fit one bounded temperature (0.5 to 3) on the selected model's inner out-of-fold
  probabilities by log loss. Report both uncalibrated and calibrated outer
  predictions. Do not use outer labels to fit the temperature or select candidates.
- Fixed original RF is refitted on each identical outer training set for a fair
  comparator. Historical artifact scores remain provenance, not the comparator
  for the new out-of-fold scores.

## Statistical analysis and freezing

Audit addendum before any new model results: free-text notes contradict exclusive
localization for some records, and several mitochondrial labels share supporting
studies. Keep the original structured-label estimand explicit. Run two additional
one-repeat nested sensitivity analyses with the identical candidate/selection
procedure: (1) exclude any nonempty localization-note record, (2) form connected
groups using either sequence links or shared supporting PubMed IDs. These address
cohort ambiguity and study dependence; they are exploratory and cannot create
independent confirmation. Do not select the primary pipeline from their scores.

Primary comparisons are selected raw method versus the refitted original RF in
macro-F1 and calibrated versus raw selected method in log loss. Use paired whole
similarity-group percentile bootstrap (2000 draws, seed 20260918). Each resampled
group retains all its proteins and both repeat predictions; repeats are not
independent observations. Report 97.5% intervals for the two primary comparisons
(Bonferroni family coverage), and descriptive 95% intervals for individual scores.
Intervals condition on fitted out-of-fold predictions and do not include all
training/selection uncertainty. No naive t-test on overlapping folds.

Report pooled and per-repeat performance, class metrics, AUROC, average precision
(AP, not trapezoidal PR area), log loss, Brier, reliability-bin counts, and
group-equal-weight sensitivity. Other model comparisons and subgroups are
exploratory, without confirmatory p-values. Probability calibration cannot be
validated from ECE or proper scoring rules alone.

Freeze a research artifact after the declared development selection rule. Retain
calibration only under its gate. Production remains version 1 unless new,
independent confirmation meets the separate promotion protocol. No result in this
revision will justify general human-proteome, multilocalization, nonhuman or
clinical-performance claims.

The final calibration on/off choice uses the two predeclared outer-path results
and is therefore a development decision. Report both paths; the score of whichever
path is retained is not an unbiased evaluation of that additional selection step.
An independent test is still required for the frozen, complete final procedure.

## Experiment ledger

The execution ledger and complete audit/results/revised Methods are maintained in
[Scientific revision](../../../docs/research-revision.md). The protocol, exact folds and machine
readable outputs accompany the result archive. Deviations must be disclosed there.

## Methodological sources

- [Cawley and Talbot, 2010](https://jmlr.org/papers/v11/cawley10a.html): model
  selection itself can overfit an estimated validation criterion.
- [scikit-learn calibration documentation](https://scikit-learn.org/stable/modules/calibration.html):
  calibrators need predictions from data excluded from base-estimator fitting;
  proper scoring rules combine calibration and discrimination.
- [UniProt subcellular-location annotation guidance](https://www.uniprot.org/help/subcellular_location):
  locations can depend on isoform, processing, and biological context.
