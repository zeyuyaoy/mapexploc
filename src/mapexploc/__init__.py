"""MAP-ExPLoc: Explainable Subcellular Localization Predictor.

This package provides utilities for preprocessing protein sequences,
training classical machine-learning models, and interpreting
predictions with SHAP.
"""

from importlib.metadata import PackageNotFoundError, version

from .adapter import BaseModelAdapter, FeatureModelAdapter, load_adapter
from .analysis import run_analysis
from .api import create_app
from .artifacts import load_model_artifact, save_model_artifact
from .config import ModelConfig, Settings, load_config
from .contracts import AdapterDescriptor
from .data import iter_sequences, load_example_dataset
from .execution import ExecutionOptions
from .explainers.shap import ShapExplainer
from .features import build_feature_matrix, normalize_protein_sequence
from .methods import MethodConfiguration
from .models import (
    evaluate_knn,
    evaluate_rf,
    knn_predict,
    knn_predict_proba,
    rf_predict,
    rf_predict_proba,
    train_knn,
    train_random_forest,
)
from .preprocessing import ALLOWED_LOCS, _clean_and_primary, extract_protein_data
from .report_v2 import AnalysisConfiguration, AnalysisReport, Protein, write_report
from .report_v3 import AnalysisReportV3, load_report

try:
    __version__ = version("mapexploc")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    "AdapterDescriptor",
    "AnalysisConfiguration",
    "AnalysisReport",
    "AnalysisReportV3",
    "MethodConfiguration",
    "ExecutionOptions",
    "load_report",
    "Protein",
    "run_analysis",
    "write_report",
    "BaseModelAdapter",
    "FeatureModelAdapter",
    "load_adapter",
    "load_model_artifact",
    "save_model_artifact",
    "ShapExplainer",
    "create_app",
    "Settings",
    "ModelConfig",
    "load_config",
    "load_example_dataset",
    "iter_sequences",
    "build_feature_matrix",
    "normalize_protein_sequence",
    "extract_protein_data",
    "_clean_and_primary",
    "ALLOWED_LOCS",
    "train_knn",
    "knn_predict",
    "knn_predict_proba",
    "evaluate_knn",
    "train_random_forest",
    "rf_predict",
    "rf_predict_proba",
    "evaluate_rf",
]
