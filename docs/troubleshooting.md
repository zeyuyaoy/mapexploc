# Troubleshooting

| Symptom                                  | Action                                                                                                                                                                      |
|------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Model unavailable / HTTP 503             | Set `MAPEXPLOC_MODEL_PATH` to a trusted compatible artifact and restart the API. `/health` loads and validates the model before reporting ready.                            |
| Explanation unavailable / HTTP 501       | Prediction-only adapters do not support tree SHAP. Predictions and sequence descriptors remain usable.                                                                      |
| Unsupported residue / HTTP 422           | Correct the named FASTA record. X, B, Z, stop symbols and gaps are rejected rather than silently removed.                                                                   |
| Service unreachable                      | Start the API on port 8000 and check the Vite proxy or production origin configuration.                                                                                     |
| Model loads with version warnings        | Match the model card's Python-library versions or retrain the frozen dataset. Pickled estimators are not portable across arbitrary library versions.                        |
| `mapexploc` cannot be imported           | Activate the intended virtual environment and run `python -m pip install -e ".[dev,docs]"`. Use `python -m pytest` to avoid stale script shebangs from another interpreter. |
| pnpm install cannot resolve the registry | Check network/DNS access. Keep the lockfile; an infrastructure failure is not a frontend build defect.                                                                      |
| Existing baseline output                 | Choose a new output path or snapshot directory. Baseline commands refuse to replace existing scientific artifacts.                                                          |
| Too few curated classes or leaked split  | Inspect the manifest/source annotations. Do not loosen evidence or grouping rules merely to obtain a higher score.                                                          |

`/features` works without a model. HTTP clients cannot select server artifact paths. The API does not persist sequences; operators should also avoid logging request bodies in proxies or instrumentation.

Source startup resolves `config/default-model.json` beside the package's source checkout. A missing file, invalid checksum or mismatched model ID prevents readiness. Restore the trusted artifact/manifest pair rather than replacing its checksum to silence an unexplained mismatch. `MAPEXPLOC_MODEL_PATH` is an explicit operator override and takes priority; even an invalid override prevents fallback. Installed wheels do not include research artifacts and need that override.

Experiment stages reject changed snapshots, reference files, completed outputs, software versions or implementation hashes. Resume with the original inputs and environment, or use a new run directory. Insufficient independent confirmation is a scientific outcome: development training can finish, but promotion remains disabled. See the [executed comparison](model-improvement.md) for counts and evidence.
