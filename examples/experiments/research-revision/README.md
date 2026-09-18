# Frozen scientific revision

Completed 2026-09-18. Full [audit, ledger, results, Methods, limitations and
reproduction instructions](../../../docs/research-revision.md); the
[predeclared plan](../../../docs/research-plan.md) records priorities and gates.

**Final candidate:** 63-feature multinomial logistic regression, L2, C=0.1,
log1p length and training-fitted standardization. No retained calibration.
Trained on 1,741 development proteins. This is internal development evidence;
there is no independent confirmation and the served default is unchanged.

Nested adaptive procedure macro-F1: 0.6094 versus refitted original RF 0.5462;
paired whole-group 97.5% difference interval +0.0357 to +0.0907. The fixed final
logistic configuration has descriptive development macro-F1 0.6006. These are
different estimands. The final candidate's Cytoplasm recall remains only 0.3212.

- `dataset.csv`: exact input, including reserved partition rows used only for
  integrity checks; fitting/evaluation in this revision uses development rows.
- `source-preparation.json`: original source/release/checksums and curation
  provenance. Its historical limitation wording is superseded by the audit.
- `primary/`, `study-groups/`, `note-free/`: readable protocols, designs and results.
  Each `run.tar.gz` contains that entire original run, including models, all
  fold records, per-protein probabilities and completion hashes. Unpack a run
  before applying its internal completion-file manifest.
- `final/model.joblib`: final **uncalibrated** research candidate; use
  `mapexploc.research.predict_research`, not the service artifact loader.
- `final/finalization.json`: exact-model calibration failure and final decision.
- `final/calibration-predictions.npz`: paired raw/temperature probabilities for
  the post-selection diagnostic, with truth and groups.
- `diagnostics/`: exact-refit checks, complete negative controls, feature
  permutation records, coefficients and error analyses.
- `audit/`: provenance/cohort audit, stored localization notes and exact original
  baseline selection reproduction.
- `figures/`: PNG/PDF, all model comparison metrics and the final-model error table.
- `execution-code.tar.gz`: executed package, tests, scripts, protocol and report.
- `scientific-environment.txt`, `execution.json`: environment and execution details.
- `checksums.json`: SHA-256 for every other archive file.

The primary nested run initially retained calibration for its adaptive selection
procedure, which selected forests in five of six outer folds. Full-development
selection chose logistic regression. An additional explicitly exploratory check
found no material calibration improvement for that exact configuration, so the
final artifact removes it. Both decisions and their probabilities are preserved.
The working finalization was repeated after formatting cleanup; the archive's
`final/` comes from `artifacts/research-revision/final-reviewed`, with identical
numerical results to the initial check and matching executed-script hashes.

Verify top-level hashes from this directory:

```python
import hashlib
import json
from pathlib import Path

root = Path(".")
for name, expected in json.loads((root / "checksums.json").read_text()).items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
```

Use a new directory for reruns. Input/source paths in frozen protocol JSON record
the original execution; use the copied dataset and a new run path on another
machine. Exact matching environments reduce numerical drift. Pickle artifacts
must come from a trusted source. The original historical benchmark was not used
to tune or confirm this revision.

Data: UniProt Consortium, UniProtKB/Swiss-Prot release 2026_03,
[CC BY 4.0](https://www.uniprot.org/help/license). Code remains under the repository
MIT license. The raw downloaded snapshot remains available locally under
`artifacts/human-v2/source.json`; this archive preserves all curated sequences and
structured annotation evidence used in the experiment, rather than redistributing
every rejected source record.
