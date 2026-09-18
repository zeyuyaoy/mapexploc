# REST response schema

`POST /predict` returns class labels and one result per input:

```json
{
  "model_classes": [
    "cytosol",
    "membrane",
    "secreted"
  ],
  "results": [
    {
      "index": 0,
      "sequence_length": 24,
      "prediction": "cytosol",
      "confidence": 0.72,
      "probabilities": [
        {
          "label": "cytosol",
          "probability": 0.72
        },
        {
          "label": "membrane",
          "probability": 0.18
        },
        {
          "label": "secreted",
          "probability": 0.10
        }
      ]
    }
  ]
}
```

`POST /explain` adds `base_value` and ranked contributions to each result:

```json
{
  "base_value": 0.33,
  "feature_contributions": [
    {
      "feature": "gravy",
      "value": 0.24,
      "contribution": 0.08
    },
    {
      "feature": "aa_K",
      "value": 0.17,
      "contribution": -0.03
    }
  ]
}
```

The API does not echo sequences. Validation errors use FastAPI's standard 422 format; missing or incompatible server models return 503; unsupported explainers return 501.

## Model-independent descriptors

`POST /features` accepts the same bounded `sequences` list and returns:

```json
{
  "results": [
    {
      "index": 0,
      "sequence_length": 4,
      "composition": {
        "A": 1.0,
        "C": 0.0
      },
      "gravy": 1.8,
      "isoelectric_point": 5.57
    }
  ]
}
```

The composition object contains all 20 standard residues (abbreviated above); fractions sum to one. Descriptor numbers shown here are illustrative. This route works when no model is configured and shares the Python feature implementation.

## Model provenance and readiness

`GET /model` returns `model_classes`, `feature_count`, `metadata_available` and `metadata`. Public metadata may include model ID/name, model family, evaluation status, experiment ID, scope, source URL/release, data hashes, counts, split policy, evaluation, timings, software versions, attribution and limitations. Filesystem paths and arbitrary metadata keys are omitted. Legacy models return an empty metadata object and `metadata_available: false`. Missing/incompatible models return 503.

Evaluation status distinguishes `development_only`, `historical_diagnostic`, `historical_holdout` and `independent_confirmation`. A higher development score is not independent confirmation. The immutable version 1 artifact's family and historical holdout status are derived at load time without changing its bytes.

`GET /health` retains `status`, `model_available` and `model_loaded`. A present but corrupt artifact is unavailable; the server attempts loading before reporting ready.

All sequence endpoints enforce at most 100 records, 100,000 residues per record and 1,000,000 residues per batch. Input normalization does not remove invalid biological symbols. Probabilities are not asserted to be calibrated.
