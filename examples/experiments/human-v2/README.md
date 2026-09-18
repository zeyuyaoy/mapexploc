# Frozen human CPU comparison

Run `human-v2-fcd650fe52bf`, executed 2026-09-18 on Apple M2 Max CPU.

The 512-tree Random Forest candidate scored 0.5761 in grouped development
validation and 0.6002 on the previously inspected historical benchmark (version 1:
0.5223). **It was not promoted:** independent confirmation was unavailable and
prediction/SHAP latency exceeded the 2× gate. Version 1 remains unchanged.

See the [methods, results and candidate model card](../../../docs/model-improvement.md).

- `dataset.csv`: curated sequences, structured evidence, groups and partitions.
- `historical.csv`: original evaluation proteins with recomputed union groups.
- `preparation.json`: source/release, exclusions, confirmation shortfalls and audits.
- `run.json`, `design.json`: frozen inputs, environment, parameters and folds.
- `training.json`: all candidate rankings, resource use and artifact checks.
- `folds.tar.gz`: 441 original per-fold JSON score records, including class metrics.
- `candidate.joblib`: frozen development-only artifact; SHA-256
  `28802e97d55d19e9864a2388e6e11e32e9ed46117db5caf8bf26002fbf8ae505`.
- `evaluation.json`, `predictions.csv`: historical diagnostic metrics, paired group
  bootstrap intervals, per-protein probabilities and matched runtime measurements.
- `promotion.json`: actual rejection reasons; `*.complete.json`: original stage hashes.
- `scientific-environment.txt`, `execution-environment.json`: exact dependencies,
  hardware, MMseqs2 version and executed commands.
- `checksums.json`: archive file hashes; individual fold hashes are also recorded
  in `train.complete.json` and can be verified after unpacking the fold archive.
- `render-results.py`: regenerate the comparison figure from frozen reports.

This is a reference archive, not a writable experiment directory. The original
working run and downloaded 26 MB JSON snapshot remain under ignored
`artifacts/human-v2`; the archive records the snapshot's URL, release and checksum
and preserves structured evidence for every curated protein. New runs use new
directories. Do not modify version 1 or refit on the historical benchmark.

Data: UniProt Consortium, UniProtKB/Swiss-Prot release 2026_03,
[CC BY 4.0](https://www.uniprot.org/help/license). Human canonical, single-compartment
research scope only; probabilities are uncalibrated. No biological-performance
claim is based on the separate synthetic test fixtures.
