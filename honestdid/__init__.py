"""
HonestDiD: Robust Inference in Difference-in-Differences and Event Study Designs

A Python implementation of the methods from Rambachan & Roth (2023),
"A More Credible Approach to Parallel Trends".

This package provides tools for sensitivity analysis in difference-in-differences
designs, allowing researchers to relax the parallel trends assumption.
"""

__version__ = "0.1.1"

from .core import (
    # Utility functions
    basis_vector,

    # Main estimation functions
    find_optimal_flci,
    constructOriginalCS,

    # Sensitivity analysis - DeltaSD (smoothness)
    createSensitivityResults,
    computeConditionalCS_DeltaSD,
    DeltaSD_upperBound_Mpre,
    DeltaSD_lowerBound_Mpre,

    # Sensitivity analysis - DeltaRM (relative magnitudes)
    createSensitivityResults_relativeMagnitudes,
    computeConditionalCS_DeltaRM,

    # Combined restrictions
    computeConditionalCS_DeltaSDB,
    computeConditionalCS_DeltaSDM,
    computeConditionalCS_DeltaRMB,
    computeConditionalCS_DeltaRMM,
    computeConditionalCS_DeltaSDRM,
    computeConditionalCS_DeltaSDRMB,
    computeConditionalCS_DeltaSDRMM,
)

__all__ = [
    # Version
    "__version__",

    # Utility functions
    "basis_vector",

    # Main estimation functions
    "find_optimal_flci",
    "constructOriginalCS",

    # Sensitivity analysis - DeltaSD (smoothness)
    "createSensitivityResults",
    "computeConditionalCS_DeltaSD",
    "DeltaSD_upperBound_Mpre",
    "DeltaSD_lowerBound_Mpre",

    # Sensitivity analysis - DeltaRM (relative magnitudes)
    "createSensitivityResults_relativeMagnitudes",
    "computeConditionalCS_DeltaRM",

    # Combined restrictions
    "computeConditionalCS_DeltaSDB",
    "computeConditionalCS_DeltaSDM",
    "computeConditionalCS_DeltaRMB",
    "computeConditionalCS_DeltaRMM",
    "computeConditionalCS_DeltaSDRM",
    "computeConditionalCS_DeltaSDRMB",
    "computeConditionalCS_DeltaSDRMM",
]
