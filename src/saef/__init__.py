"""SOMA AI Emission Factors (SAEF): estimated carbon and water factors for AI inference."""
from ._core import (
    DATASET_VERSION,
    Estimate,
    SaefError,
    citation,
    estimate,
    factor,
    model_class_for,
    regions,
)

__version__ = DATASET_VERSION

__all__ = [
    "DATASET_VERSION",
    "Estimate",
    "SaefError",
    "__version__",
    "citation",
    "estimate",
    "factor",
    "model_class_for",
    "regions",
]
