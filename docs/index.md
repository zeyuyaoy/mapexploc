### Methods and evidence

MAP-ExPLoc studies which sequence properties drive localization predictions and whether attributions agree with annotated localization determinants. The human predictor study and DeepLoc attribution experiment have different targets and validation designs. Neither establishes a causal mechanism or experimentally confirms localization. [Reproduction instructions](research-software.md) identify the evidence bundles and executable workflows.

#### Human annotation prediction

Both source downloads contain the same 20,431 reviewed human UniProtKB/Swiss-Prot records, release 2026_03, retrieved 2026-09-18. Curation requires a canonical sequence, one mapped structured class and `ECO:0000269` attached to a mapped location; protein-existence evidence alone is insufficient.

| Class         | Accepted locations             | Development proteins |
|---------------|--------------------------------|---------------------:|
| Cytoplasm     | Cytoplasm, Cytosol             |                  316 |
| Membrane      | Cell membrane, Plasma membrane |                  406 |
| Mitochondrion | Mitochondrion                  |                  154 |
| Nucleus       | Nucleus                        |                  677 |
| Secreted      | Secreted                       |                  188 |

Unmapped/conflicting structured locations and isoform/product-specific comments exclude records. Identical sequences are deduplicated; conflicting duplicate labels are excluded. Missing localization is neither negative nor imputed. Free-text notes do not determine acceptance and may describe additional compartments. Thus the target is **one supported annotation, not exclusive localization**. Curation accepts 2,107 proteins (10.31%); record-level exclusions are in `examples/experiments/research-revision/audit/`.

The historical baseline capped classes at 500 (seed 42), producing 1,458 training and 364 holdout proteins. Expanded development excludes those 364 accessions and two related proteins: 1,741 proteins in 1,510 groups. MMseqs2 18-8cc5c connected components use ≥30% identity and ≥80% bidirectional coverage over original and accepted expanded sequences; rejected records are not bridge nodes. No qualifying development–historical hits remained. Remote/domain homology and shared experimental dependence remain possible; the expanded cohort is not an independent release.

#### Representation and validation

Inputs are case/whitespace-normalized; empty sequences, gaps and nonstandard residues are rejected, excluding selenoproteins. The historical 128-tree Random Forest (RF) uses 423 whole-sequence descriptors: length, 20 amino-acid frequencies, 400 overlapping dipeptide frequencies divided by `max(length−1, 1)`, GRAVY and Biopython isoelectric point. The 23-feature representation omits dipeptides; 63 features add first-/last-50-residue composition, using the whole sequence when shorter and allowing terminal overlap. Linear models use `log1p(length)` and training-fitted standardization.

Nested selection uses three outer stratified group folds, repeated with seeds 20260918/20260919, and three inner folds. Scaling, selection and temperature fitting remain inside training folds. Fifteen candidates include prior dummy, length-only logistic, L2 multinomial logistic on 23/63/423 features and four forests; the historical RF is refitted on identical outer rows. Selection chooses the least complex candidate within 0.01 macro-F1, 0.02 log loss and 0.05 per-class F1 of the leader; ties prefer smaller C, then candidate ID. Candidate definitions/ranks are in `primary/protocol.json` within the research-revision bundle.

Primary estimands are adaptive-selection minus matched-RF macro-F1, and calibrated minus raw adaptive-selection log loss. Metrics pool out-of-fold predictions across repeats. Macro-F1 includes all five classes with undefined F1 set to zero. Paired whole-group bootstrap retains both repeats: 2,000 draws, seed 20260918, 97.5% intervals for two primary comparisons. Intervals condition on fitted predictions and omit full training/selection uncertainty; repeated predictions are not independent samples.

| Procedure                                     | Macro-F1 | Log loss |
|-----------------------------------------------|---------:|---------:|
| Prior dummy                                   |   0.1120 |   1.4714 |
| Matched RF                                    |   0.5462 |   1.1676 |
| Adaptive selection, raw                       |   0.6094 |   0.9684 |
| Adaptive selection, temperature-scaled        |   0.6094 |   0.9089 |
| Final logistic configuration, descriptive OOF |   0.6006 |   0.8964 |

Adaptive macro-F1 gain: **0.0632 (97.5% interval 0.0357–0.0907)**. Calibration changes log loss by −0.0595 (−0.0760 to −0.0430). Five outer folds selected terminal forests; one selected terminal logistic. Full-development selection chose 63-feature logistic, C=0.1. Its OOF row is post-selection description, not independent validation. An exploratory exact-model calibration check failed its ≥0.02 improvement/interval-excluding-zero gate; the final artifact has temperature 1. This check and stronger label permutations were added after inspecting earlier results. The served RF is a separate, uncalibrated artifact with original holdout macro-F1 0.5223; that score is not a matched comparison with nested selection.

The gain persisted under publication-linked grouping and exclusion of localization-note records; neither is independent confirmation. Development contains 313 proteins with such notes; 67 supporting publications overlap development and historical-related records. Publication grouping controls only recorded links. Final logistic Cytoplasm recall is 0.3212, and confident errors remain. Experimental-annotation selection and class caps preclude proteome-wide prevalence/accuracy claims. Tissue, condition, assay and fragment metadata are incomplete; assayed mature proteins may differ from canonical precursors. Multilabel, out-of-scope and abstention behavior, other species, isoforms, fragments and independent population transfer remain unvalidated.

#### Attribution methods

TreeSHAP explains feature-model class probabilities using fitted tree-path counts as reference; signed contributions reconstruct probabilities within `1e-6`. Whole-sequence descriptors, including dipeptide frequencies, cannot localize contributions to individual occurrences. Unsupported transformations/output types are rejected.

DeepLoc 2.1 has ten multilabel outputs, unlike the RF's five single-label classes. Reports preserve native class order, mode-specific thresholds, strict threshold comparison and four-decimal nearest-threshold fallback when none passes; scores are not renormalized. Fast uses ESM1b (10–1,022 residues); Accurate uses ProtT5-XL-UniRef50 (10–4,000). Longer sequences are rejected rather than middle-truncated. Accurate requires single-sequence batches because native pooling includes padding. Native attention/sorting-signal outputs are not SHAP.

Default region KernelSHAP partitions the first/last 50 residues into ten-residue windows and up to six interior regions; proteins shorter than 100 residues use contiguous ten-residue windows. Partitions are annotation-independent. Four deterministic whole-sequence shuffles provide fixed references. Absent regions are replaced from each reference, native inference is recomputed, and probabilities are averaged. The identity-link game uses no feature-selection regularization and 512 nontrivial coalitions, enumerating all when fewer exist. Shuffles preserve length/composition, but hybrid coalitions may change composition; both can be biologically implausible. Values attribute **regions, not residues**. Reconstruction tolerance is `1e-5`; additivity does not establish convergence or faithfulness.

Reports preserve references, seeds, coordinates, bases and all class contributions. Sensitivity diagnostics distinguish coalition sampling, reference sampling and interventions. Warning cutoffs—absolute change >0.02, magnitude-rank correlation <0.8, sign agreement <0.9 above magnitude 0.005, top-20% overlap <0.6—are operational, not biological confidence levels; constant explanations are uninformative.

Global summaries are class-specific mean signed/absolute contributions. Regional categories are averaged within protein, then across proteins; shared coordinates do not imply alignment. Schema 3 adds equal-group summaries, medians, 10% trimmed means, 1,000-replicate group-bootstrap intervals and explicit attribution density. Fewer than 20 groups triggers a warning. These summaries are exploratory without a prespecified estimand.

Overlays require exact sequence/hash, accession/isoform, source/version, evidence and original coordinates. Internal intervals are zero-based half-open; display is one-based inclusive. Overlaps remain separate; uncertain boundaries are excluded from exact-coordinate statistics. Mismatches are rejected without inferred alignment. Missing annotation does not establish absence.

#### DeepLoc biological evidence

The `examples/validation/v2/` experiment selected 30 accession-ordered sequence-group representatives from 730 eligible proteins in a 9,450-record reviewed eukaryotic UniProt snapshot (2026-09-18). Eligibility required 100–250 residues and a precisely bounded, experimentally supported signal peptide from residue 1 to 20–40; evidence had to support the positional feature. Grouping used ≥30% identity/≥80% bidirectional coverage.

The Extracellular estimand was positive SHAP per residue in complete signal-contained windows minus matched nonsignal windows within the first 50 residues, averaged equally across groups. All 30 had controls. The contrast was **0.009415 probability units/residue (95% group-bootstrap interval 0.006857–0.012581)**. However, 1,000 annotation–attribution randomizations within length/terminal strata gave null mean 0.009174 and **one-sided p=0.077922**: no support beyond that matched null. Twenty-eight signals occupied the same first two complete windows, limiting discrimination. The contrast interval does not test this null.

One of three sensitivity checks exceeded 0.02; 27 proteins were unassessed. Ten proteins exactly matched available localization/sorting-signal supervision; homologous and foundation-model exposure were not excluded. Definitions, null draws and overlap audits are in `protocol.json`, `biological-validation.json` and `supervision-overlap.json`.
