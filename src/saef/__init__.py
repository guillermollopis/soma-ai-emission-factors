"""SOMA AI Emission Factors (SAEF): estimated carbon and water factors for AI inference."""
from ._core import (
    DATASET_VERSION,
    Estimate,
    SaefError,
    citation,
    estimate,
    factor,
    grids,
    model_class_for,
    regions,
)

__version__ = "3.2.0"

__all__ = [
    "DATASET_VERSION",
    "Estimate",
    "SaefError",
    "__version__",
    "citation",
    "estimate",
    "factor",
    "grids",
    "model_class_for",
    "regions",
]
