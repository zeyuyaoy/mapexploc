"""Resume a frozen real-checkpoint cohort; each cached report is contract-checked.

Run from an installed MAP-ExPLoc environment. Licensed assets remain external.
"""

import argparse
import json
from pathlib import Path

from mapexploc import AnalysisReport, Protein, run_analysis, write_report
from mapexploc.adapter import registered_adapter
from mapexploc.biological_validation import validate_signal_concordance
from mapexploc.report_v2 import AnalysisConfiguration, aggregate_cohort


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--adapter-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = json.loads((args.inputs / "uniprot-records.json").read_text())
    manifest = json.loads((args.inputs / "cohort-manifest.json").read_text())
    members = {r["protein_id"]: r for r in manifest["members"]}
    annotations = json.loads((args.inputs / "annotations.json").read_text())
    base_config = AnalysisConfiguration(
        cohort_id=manifest["cohort_id"],
        selection_criteria=manifest["selection_criteria"],
    )
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    with registered_adapter(
        "deeploc2", json.loads(args.adapter_config.read_text())
    ) as adapter:
        for index, record in enumerate(records):
            accession = record["primaryAccession"]
            protein = Protein(
                protein_id=accession,
                sequence=record["sequence"]["value"],
                group=members[accession]["group"],
                split=members[accession]["split"],
            )
            configuration = base_config.model_copy(
                update={"stability_checks": index < 3}
            )
            directory = args.output / "proteins" / accession
            cached = directory / "report.json"
            intervals = [a for a in annotations if a["protein_id"] == accession]
            if cached.exists():
                report = AnalysisReport.model_validate_json(cached.read_text())
                if (
                    report.model != adapter.descriptor
                    or report.configuration != configuration
                    or report.results[0].protein != protein
                    or [a.model_dump() for a in report.results[0].annotations]
                    != intervals
                ):
                    raise ValueError(
                        (
                            "Cached report differs from frozen "
                            "model/protein/configuration/annotations"
                        )
                    )
            else:
                print(
                    f"Explaining {index + 1}/{len(records)}: {accession} "
                    f"({len(protein.sequence)} aa)",
                    flush=True,
                )
                report = run_analysis(adapter, [protein], configuration, intervals)
                write_report(report, directory)
            results.append(report.results[0])
            print(
                f"Completed {accession}: "
                f'{report.results[0].explainer["stability"]["status"]}',
                flush=True,
            )
        complete = AnalysisReport(
            model=adapter.descriptor,
            configuration=base_config,
            results=results,
            cohort=aggregate_cohort(results, base_config),
            software=report.software,
            created_at=report.created_at,
            warnings=[
                (
                    "Frozen biological model-behavior cohort; see "
                    "supervision-overlap audit. First three accession-ordered "
                    "proteins include seed/budget sensitivity checks."
                )
            ],
        )
        write_report(complete, args.output)
        validation = validate_signal_concordance(complete)
        (args.output / "biological-validation.json").write_text(
            json.dumps(validation, indent=2) + "\n"
        )
        print(
            validation["conclusion"],
            validation["effect"],
            validation["one_sided_randomization_p"],
            flush=True,
        )


if __name__ == "__main__":
    main()
