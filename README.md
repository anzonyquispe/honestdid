# HonestDiD (Python)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Robust inference for Difference-in-Differences and event study designs.

This is a Python implementation of the methods proposed in [Rambachan and Roth (2023)](https://asheshrambachan.github.io/assets/files/hpt-draft.pdf), "A More Credible Approach to Parallel Trends."

## Overview

The `honestdid` package implements methods for constructing robust confidence intervals in difference-in-differences and event study designs. These methods allow researchers to relax the parallel trends assumption and perform sensitivity analysis to evaluate how violations of parallel trends affect their conclusions.

The key idea is that **pre-trends are informative about violations of parallel trends**. The package formalizes this intuition through two main approaches:

### Smoothness Restrictions (Delta^{SD})

Restrictions that bound how the slope of the underlying trend can change between consecutive periods. Setting $M = 0$ implies the counterfactual trend is exactly linear, while larger values of $M$ allow for more non-linearity.

### Relative Magnitudes Restrictions (Delta^{RM})

Restrictions that bound the post-treatment violation of parallel trends relative to the maximum pre-treatment violation. Setting $\bar{M} = 1$ means post-treatment violations can be no larger than the maximum pre-treatment violation.

## Installation

### Install from GitHub (recommended)

```bash
pip install git+https://github.com/anzonyquispe/honestdid.git
```

### Install from source

```bash
git clone https://github.com/anzonyquispe/honestdid.git
cd honestdid
pip install -e .
```

### Dependencies

- Python >= 3.8
- PyTorch >= 1.9.0
- NumPy >= 1.19.0
- SciPy >= 1.6.0
- CVXPY >= 1.1.0
- Pandas >= 1.2.0

## Basic Usage

```python
import torch
import honestdid as hd

# Load your event study estimates (betahat) and variance-covariance matrix (sigma)
# betahat: coefficient estimates for pre and post periods (excluding reference period)
# sigma: variance-covariance matrix

# Example: 4 pre-treatment periods + 4 post-treatment periods
numPrePeriods = 4
numPostPeriods = 4

# Target parameter: effect in first post-treatment period
l_vec = hd.basis_vector(index=1, size=numPostPeriods)

# Original confidence set (assuming parallel trends holds exactly)
original_cs = hd.constructOriginalCS(
    betahat=betahat,
    sigma=sigma,
    numPrePeriods=numPrePeriods,
    numPostPeriods=numPostPeriods,
    l_vec=l_vec,
    alpha=0.05
)

# Sensitivity analysis using smoothness restrictions (DeltaSD)
sensitivity_results = hd.createSensitivityResults(
    betahat=betahat,
    sigma=sigma,
    numPrePeriods=numPrePeriods,
    numPostPeriods=numPostPeriods,
    l_vec=l_vec,
    Mvec=[0, 0.01, 0.02, 0.03],  # different values of M
    alpha=0.05
)
print(sensitivity_results)

# Sensitivity analysis using relative magnitudes (DeltaRM)
sensitivity_rm = hd.createSensitivityResults_relativeMagnitudes(
    betahat=betahat,
    sigma=sigma,
    numPrePeriods=numPrePeriods,
    numPostPeriods=numPostPeriods,
    l_vec=l_vec,
    Mbarvec=[0.5, 1.0, 1.5, 2.0],  # different values of Mbar
    alpha=0.05
)
print(sensitivity_rm)
```

## Main Functions

| Function | Description |
|----------|-------------|
| `constructOriginalCS` | Constructs the original confidence set assuming parallel trends holds |
| `find_optimal_flci` | Finds the optimal fixed-length confidence interval |
| `createSensitivityResults` | Sensitivity analysis under smoothness restrictions (DeltaSD) |
| `createSensitivityResults_relativeMagnitudes` | Sensitivity analysis under relative magnitudes (DeltaRM) |
| `computeConditionalCS_DeltaSD` | Conditional confidence set for DeltaSD |
| `computeConditionalCS_DeltaRM` | Conditional confidence set for DeltaRM |
| `DeltaSD_upperBound_Mpre` | Upper bound for M from pre-treatment periods |
| `DeltaSD_lowerBound_Mpre` | Lower bound for M from pre-treatment periods |

## Advanced Usage

### Fixed-Length Confidence Intervals (FLCI)

```python
flci = hd.find_optimal_flci(
    betahat=betahat,
    sigma=sigma,
    numPrePeriods=numPrePeriods,
    numPostPeriods=numPostPeriods,
    l_vec=l_vec,
    M=0,  # smoothness parameter
    alpha=0.05
)
print(f"FLCI: [{flci['FLCI'][0]:.4f}, {flci['FLCI'][1]:.4f}]")
```

### Breakdown Values

Calculate the value of M at which the confidence set includes zero:

```python
# Get upper bound for M from pre-treatment data
M_upper = hd.DeltaSD_upperBound_Mpre(
    betahat=betahat,
    sigma=sigma,
    numPrePeriods=numPrePeriods,
    alpha=0.05
)
print(f"Upper bound for M: {M_upper:.4f}")
```

### Combined Restrictions

The package also supports combining multiple restrictions:

- `computeConditionalCS_DeltaSDB`: Smoothness + sign restrictions
- `computeConditionalCS_DeltaSDM`: Smoothness + monotonicity
- `computeConditionalCS_DeltaRMB`: Relative magnitudes + sign restrictions
- `computeConditionalCS_DeltaRMM`: Relative magnitudes + monotonicity
- `computeConditionalCS_DeltaSDRM`: Smoothness + relative magnitudes

## Related Implementations

- **R Package**: [HonestDiD](https://github.com/asheshrambachan/HonestDiD) (original implementation)
- **Stata Package**: [honestdid](https://github.com/mcaceresb/stata-honestdid)

## Citation

If you use this package, please cite:

```bibtex
@article{rambachan2023more,
  title={A More Credible Approach to Parallel Trends},
  author={Rambachan, Ashesh and Roth, Jonathan},
  journal={Review of Economic Studies},
  volume={90},
  number={5},
  pages={2555--2591},
  year={2023},
  publisher={Oxford University Press}
}
```

## References

- Rambachan, Ashesh and Jonathan Roth. "A More Credible Approach to Parallel Trends." *Review of Economic Studies* 90.5 (2023): 2555-2591. [[Paper]](https://asheshrambachan.github.io/assets/files/hpt-draft.pdf)
- [Video presentation](https://www.youtube.com/watch?v=6-NkiA2jN7U) by the authors
- [Interactive Shiny application](https://ccfang2.shinyapps.io/HonestDiDSenAnlys/)

## License

MIT License
