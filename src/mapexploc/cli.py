"""Command-line workflows for training, predicting, and explaining."""

from __future__ import annotations

import logging
from importlib.resources import files
from pathlib import Path
from typing import Any, NoReturn

import json
import typer

from mapexploc.artifacts import load_model_artifact, save_model_artifact
from mapexploc.config import load_config
from mapexploc.data import load_example_dataset
from mapexploc.explainers.shap import ShapExplainer
from mapexploc.features import build_feature_matrix
from mapexploc.models.rf import rf_predict, rf_predict_proba, train_random_forest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("shap").setLevel(logging.WARNING)

app = typer.Typer(
    help="MAP-ExPLoc: explainable protein subcellular localization",
    no_args_is_help=True,
    invoke_without_command=True,
)


@app.callback()
def callback(
    version: bool = typer.Option(
        False, "--version", help="Show the installed version and exit", is_eager=True
    ),
) -> None:
    """Run MAP-ExPLoc workflows."""
    if version:
        from mapexploc import __version__

        typer.echo(__version__)
        raise typer.Exit()


def _fail(message: str) -> NoReturn:
    typer.echo(message, err=True)
    raise typer.Exit(code=1)


@app.command()
def train(
    config: Path = typer.Option(..., exists=True, dir_okay=False),
    data_path: Path | None = typer.Option(
        None,
        "--data-path",
        "--data",
        help="CSV with sequence and label columns",
    ),
    output_model: Path = typer.Option(
        Path("model.pkl"), "--output-model", "--output", help="Artifact output path"
    ),
) -> None:
    """Train and save a Random Forest localization model."""
    try:
        cfg = load_config(config)
        if data_path is None:
            data_path = Path(str(files("mapexploc").joinpath("examples/smoke.csv")))
        frame = load_example_dataset(data_path)
        features = build_feature_matrix(frame["sequence"])
        targets = frame["label"].astype(str)
        param_grid: dict[str, list[Any]] = {
            "rf__n_estimators": [cfg.model.n_estimators],
            "rf__max_depth": [cfg.model.max_depth],
        }
        result = train_random_forest(
            features,
            targets,
            param_grid,
            random_state=cfg.seed,
            groups=frame["group"] if "group" in frame else None,
        )
        save_model_artifact(
            result["model"],
            output_model,
            metadata={
                "sample_count": len(frame),
                "best_params": result["best_params"],
                "best_cv_score": result["best_cv_score"],
                "evaluation_status": "development_selection_only",
                "grouped_cv": "group" in frame,
            },
        )
    except (OSError, ValueError) as exc:
        _fail(f"Training failed: {exc}")
    typer.echo(f"Saved model artifact to {output_model}")
    if result["best_cv_score"] is None:
        typer.echo(
            "Warning: the dataset was too small for cross-validation; this artifact "
            "is suitable for workflow testing, not scientific use.",
            err=True,
        )


@app.command()
def predict(
    sequence: str = typer.Argument(..., help="Unambiguous protein sequence"),
    model_path: Path = typer.Option(
        Path("model.pkl"), exists=True, dir_okay=False, help="Trusted model artifact"
    ),
) -> None:
    """Predict localization and confidence for one protein sequence."""
    try:
        model = load_model_artifact(model_path).model
        features = build_feature_matrix([sequence])
        prediction = str(rf_predict(model, features)[0])
        probabilities = rf_predict_proba(model, features)[0]
    except (OSError, ValueError, TypeError) as exc:
        _fail(f"Prediction failed: {exc}")
    typer.echo(f"Prediction: {prediction}")
    typer.echo(f"Confidence: {float(probabilities.max()):.3f}")


@app.command()
def explain(
    sequence: str = typer.Argument(..., help="Unambiguous protein sequence"),
    model_path: Path = typer.Option(
        Path("model.pkl"), exists=True, dir_okay=False, help="Trusted model artifact"
    ),
    output_dir: Path = typer.Option(
        Path("results/shap"), help="Directory for explanation.json"
    ),
    top_n: int = typer.Option(12, min=1, max=25, help="Contributions to return"),
) -> None:
    """Write a local SHAP feature explanation as structured JSON."""
    try:
        model = load_model_artifact(model_path).model
        features = build_feature_matrix([sequence])
        report = ShapExplainer(model, output_dir=str(output_dir)).explain_predictions(
            features, top_n=top_n
        )[0]
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "explanation.json"
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, TypeError, ImportError) as exc:
        _fail(f"Explanation failed: {exc}")
    typer.echo(f"Wrote explanation to {output_path}")


@app.command("baseline-download")
def baseline_download(
    directory: Path = typer.Option(Path("artifacts/human-baseline")),
) -> None:
    """Explicitly download a new public reviewed-human UniProt snapshot."""
    from mapexploc.baseline import download_snapshot

    try:
        download_snapshot(directory)
    except (OSError, ValueError) as exc:
        _fail(str(exc))
    typer.echo(f"Downloaded snapshot to {directory}")


@app.command("baseline-prepare")
def baseline_prepare(
    directory: Path = typer.Option(Path("artifacts/human-baseline")),
    cap: int = typer.Option(500, min=10),
    threads: int = typer.Option(4, min=1),
) -> None:
    """Curate evidence, group related sequences, and audit an 80/20 split."""
    from mapexploc.baseline import prepare_baseline

    try:
        manifest = prepare_baseline(directory, cap=cap, threads=threads)
    except (OSError, ValueError) as exc:
        _fail(str(exc))
    typer.echo(f"Prepared {sum(manifest['class_counts'].values())} curated proteins")


@app.command("baseline-train")
def baseline_train(
    directory: Path = typer.Option(Path("artifacts/human-baseline")),
    output_model: Path = typer.Option(Path("artifacts/human-baseline/model.joblib")),
    jobs: int = typer.Option(4, min=1),
) -> None:
    """Reproduce grouped training and the already-inspected historical holdout."""
    from mapexploc.baseline import train_baseline

    try:
        report = train_baseline(directory, output_model, jobs=jobs)
    except (OSError, ValueError) as exc:
        _fail(str(exc))
    typer.echo(f"Held-out macro-F1: {report['evaluation']['macro_f1']:.3f}")
    typer.echo(f"Saved schema-checked research artifact to {output_model}")


@app.command("experiment-prepare")
def experiment_prepare(
    directory: Path = typer.Option(Path("artifacts/human-v2")),
    reference_root: Path = typer.Option(Path(".")),
    cap: int = typer.Option(1000, min=20),
    threads: int = typer.Option(4, min=1, max=4),
) -> None:
    """Freeze a refreshed experiment and its independent confirmation eligibility."""
    from mapexploc.experiments import prepare_experiment

    try:
        result = prepare_experiment(directory, reference_root, cap, threads)
    except (OSError, ValueError) as exc:
        _fail(str(exc))
    typer.echo(json.dumps(result, indent=2))


@app.command("experiment-train")
def experiment_train(
    directory: Path = typer.Option(Path("artifacts/human-v2")),
    jobs: int = typer.Option(4, min=1, max=4),
) -> None:
    """Run the fixed 441-fit grouped RF/Extra Trees comparison, with resume."""
    from mapexploc.experiments import train_experiment

    try:
        result = train_experiment(directory, jobs)
    except (OSError, ValueError) as exc:
        _fail(str(exc))
    typer.echo(json.dumps(result["winner"], indent=2))


@app.command("experiment-evaluate")
def experiment_evaluate(
    directory: Path = typer.Option(Path("artifacts/human-v2")),
) -> None:
    """Evaluate the frozen winner and record independent or historical status."""
    from mapexploc.experiments import evaluate_experiment

    try:
        result = evaluate_experiment(directory)
    except (OSError, ValueError) as exc:
        _fail(str(exc))
    typer.echo(
        json.dumps(
            {
                "status": result["evaluation_status"],
                "macro_f1": result["evaluation"]["macro_f1"],
                "promotion": result["promotion"],
            },
            indent=2,
        )
    )


@app.command("model-promote")
def model_promote(
    directory: Path = typer.Option(Path("artifacts/human-v2")),
    repository: Path = typer.Option(Path(".")),
) -> None:
    """Update the default only when all independent-confirmation gates pass."""
    from mapexploc.experiments import promote_experiment

    try:
        result = promote_experiment(directory, repository)
    except (OSError, ValueError) as exc:
        _fail(str(exc))
    typer.echo(json.dumps(result, indent=2))


@app.command()
def analyze(
    fasta: Path = typer.Option(..., exists=True, dir_okay=False),
    adapter: str = typer.Option(..., help="Installed adapter registry identifier"),
    adapter_config: Path = typer.Option(..., exists=True, dir_okay=False),
    configuration: Path | None = typer.Option(None, exists=True, dir_okay=False),
    annotations: Path | None = typer.Option(None, exists=True, dir_okay=False),
    cohort_manifest: Path | None = typer.Option(None, exists=True, dir_okay=False),
    output_dir: Path = typer.Option(Path("results/analysis")),
    model_mode: str | None = typer.Option(None, help="DeepLoc mode: fast or accurate"),
    cache_dir: Path | None = typer.Option(
        None, help="Bounded persistent probability cache"
    ),
    restart_dir: Path | None = typer.Option(
        None, help="Save/resume completed proteins with a validated manifest"
    ),
    run_spec: Path | None = typer.Option(
        None,
        exists=True,
        dir_okay=False,
        help="Portable mode/identity/method specification exported by the viewer",
    ),
) -> None:
    """Complete all-class analysis to portable JSON, CSV and standalone HTML."""
    from Bio import SeqIO

    from mapexploc.adapter import registered_adapter
    from mapexploc.analysis import run_analysis
    from mapexploc.execution import ExecutionOptions
    from mapexploc.provenance import sequence_sha256
    from mapexploc.report_v2 import Protein, write_report

    loaded = None
    try:
        proteins = [
            Protein(protein_id=r.id, sequence=str(r.seq))
            for r in SeqIO.parse(fasta, "fasta")  # type: ignore[no-untyped-call]
        ]
        config = json.loads(configuration.read_text()) if configuration else {}
        portable = json.loads(run_spec.read_text()) if run_spec else None
        if portable:
            if (
                portable.get("schema_version") != 1
                or portable.get("adapter") != adapter
            ):
                raise ValueError(
                    "Run specification schema or adapter disagrees with CLI"
                )
            if configuration and config != portable["configuration"]:
                raise ValueError(
                    "Run specification and analysis configuration disagree"
                )
            config = portable["configuration"]
            if model_mode and model_mode != portable.get("mode"):
                raise ValueError("Run specification mode disagrees with CLI")
            model_mode = portable.get("mode")
        if cohort_manifest:
            manifest = json.loads(cohort_manifest.read_text())
            expected = {p.protein_id: sequence_sha256(p.sequence) for p in proteins}
            members = manifest["members"]
            if (
                len(members) != len(expected)
                or {m["protein_id"]: m["sequence_sha256"] for m in members} != expected
            ):
                raise ValueError(
                    "Cohort manifest membership/checksums differ from FASTA"
                )
            config.update(
                cohort_id=manifest["cohort_id"],
                selection_criteria=manifest["selection_criteria"],
            )
            by_id = {m["protein_id"]: m for m in members}
            proteins = [
                Protein(
                    **{
                        **p.model_dump(),
                        **{
                            k: by_id[p.protein_id][k]
                            for k in ("group", "split", "evaluation_labels")
                            if k in by_id[p.protein_id]
                        },
                    }
                )
                for p in proteins
            ]
        native_configuration = json.loads(adapter_config.read_text())
        if model_mode is not None:
            if adapter != "deeploc2" or model_mode not in {"fast", "accurate"}:
                raise ValueError("--model-mode requires deeploc2 and fast|accurate")
            if (
                "mode" in native_configuration
                and native_configuration["mode"] != model_mode
            ):
                raise ValueError(
                    "CLI mode disagrees with the trusted adapter configuration"
                )
            native_configuration["mode"] = model_mode
        loaded = registered_adapter(adapter, native_configuration)
        if portable and (
            portable.get("expected_model_id") != loaded.descriptor.model_id
            or portable.get("expected_checkpoint_sha256")
            != loaded.descriptor.checkpoint_sha256
        ):
            raise ValueError(
                "Run specification differs from the actual model/checkpoint"
            )
        intervals = json.loads(annotations.read_text()) if annotations else []
        report = run_analysis(
            loaded,
            proteins,
            config,
            intervals,
            execution=ExecutionOptions(
                cache_directory=cache_dir,
                restart_directory=restart_dir,
                progress=lambda event: typer.echo(json.dumps(event), err=True),
            ),
        )
        paths = write_report(report, output_dir)
    except (OSError, ValueError, TypeError, RuntimeError, ImportError) as exc:
        _fail(f"Analysis failed: {exc}")
    finally:
        close = getattr(loaded, "close", None)
        if callable(close):
            close()
    typer.echo(f"Completed {len(report.results)} proteins: {paths['report.json']}")


@app.command("compare")
def compare_command(
    left: Path = typer.Option(..., exists=True, dir_okay=False),
    right: Path = typer.Option(..., exists=True, dir_okay=False),
    output: Path = typer.Option(Path("comparison.json")),
) -> None:
    """Compare completed reports without recomputing predictions or explanations."""
    from .comparison import compare_reports
    from .execution import atomic_json
    from .report_v3 import load_report

    try:
        atomic_json(output, compare_reports(load_report(left), load_report(right)))
    except (OSError, ValueError, TypeError) as exc:
        _fail(str(exc))
    typer.echo(str(output))


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    main()
