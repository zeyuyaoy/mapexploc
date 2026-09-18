"""Stronger software negative control after partial group shuffling retained signal."""

import argparse
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from mapexploc.baseline import checksum
from mapexploc.experiments import atomic_json, read_json
from mapexploc.research import (
    SEEDS,
    fit_checked,
    load_development,
    make_model,
    read_record,
    research_features,
    score,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = read_json(args.run / "protocol.json")
    source = Path(protocol["source"])
    if checksum(source) != protocol["source_sha256"]:
        raise ValueError("Input changed")
    frame, _ = load_development(source)
    x, y = research_features(frame.sequence), frame.label.to_numpy()
    configs = {c["id"]: c for c in protocol["candidates"]}
    design = read_json(args.run / "design.json")
    arrays = {seed: np.zeros((len(frame), 5)) for seed in range(SEEDS[0], SEEDS[0] + 5)}
    exchangeability = []
    with threadpool_limits(limits=1):
        for fold in design["folds"]:
            if fold["seed"] != SEEDS[0]:
                continue
            train, valid = np.array(fold["train"]), np.array(fold["validation"])
            groups = frame.group.iloc[train]
            sizes = groups.value_counts()
            no_partner = sizes.index[sizes.map(sizes.value_counts()) == 1]
            exchangeability.append(
                {
                    "fold": fold["fold"],
                    "training_rows": len(train),
                    "rows_without_same_size_group_partner": int(
                        groups.isin(no_partner).sum()
                    ),
                }
            )
            record = read_record(
                args.run / "folds" / f'{fold["seed"]}-{fold["fold"]}.json'
            )
            for seed, output in arrays.items():
                shuffled = np.random.default_rng(seed).permutation(y[train])
                model = fit_checked(
                    make_model(configs[record["selected"]], fold["seed"]),
                    x.iloc[train],
                    shuffled,
                )
                output[valid] = model.predict_proba(x.iloc[valid])
    result = {
        "purpose": "Exploratory falsification; not a valid group permutation p-value",
        "method": (
            "Completely permute training labels; "
            "validation labels and split groups unchanged"
        ),
        "source_sha256": checksum(source),
        "script_sha256": checksum(Path(__file__)),
        "group_shuffle_exchangeability": exchangeability,
        "metrics": {str(seed): score(y, p) for seed, p in arrays.items()},
    }
    atomic_json(args.output, result)
    print({s: round(m["macro_f1"], 4) for s, m in result["metrics"].items()})


if __name__ == "__main__":
    main()
