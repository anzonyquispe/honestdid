"""
Example: HonestDiD Python Package - Basic Usage
Uses BCdata_EventStudy from Rambachan & Roth (2023)

Run after installing the package:
    pip install -e .
    python examples/basic_usage.py
"""
import os
import torch
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import honestdid as hd

# =============================================================================
# Step 1: Get data from R (run this in R first to generate the JSON)
# =============================================================================
# In R:
#   library(HonestDiD); library(jsonlite)
#   data(BCdata_EventStudy)
#   out <- list(betahat = as.numeric(BCdata_EventStudy$betahat),
#               sigma = as.matrix(BCdata_EventStudy$sigma))
#   write_json(out, 'bcdata.json', digits=16)
#
# Then load here:
# import json
# with open('bcdata.json') as f:
#     data = json.load(f)
# betahat = torch.tensor(data['betahat'], dtype=torch.float64)
# sigma = torch.tensor(data['sigma'], dtype=torch.float64)

print("=" * 60)
print("HonestDiD Python Package - Example Usage")
print("=" * 60)

# --- Option A: Load from JSON (recommended) ---
# Look for test data in the examples directory or parent directory
script_dir = os.path.dirname(os.path.abspath(__file__))
data_paths = [
    os.path.join(script_dir, 'test_bcdata.json'),
    os.path.join(os.path.dirname(script_dir), 'test_bcdata.json'),
]

data_loaded = False
for data_path in data_paths:
    try:
        import json
        with open(data_path) as f:
            data = json.load(f)
        betahat = torch.tensor(data['betahat'], dtype=torch.float64)
        sigma = torch.tensor(data['sigma'], dtype=torch.float64)
        numPre = 4
        numPost = 4
        print(f"\nLoaded BCdata: betahat length={len(data['betahat'])}, sigma shape={sigma.shape}")
        data_loaded = True
        break
    except FileNotFoundError:
        continue

if not data_loaded:
    print("\nNo test_bcdata.json found. Generating synthetic data for demo...")
    # --- Option B: Synthetic data for demo ---
    np.random.seed(42)
    numPre = 4
    numPost = 3
    n = numPre + numPost
    betahat_np = np.concatenate([
        np.random.normal(0, 0.01, numPre),    # pre-treatment (near zero)
        np.random.normal(0.05, 0.01, numPost)  # post-treatment (positive effect)
    ])
    # Create a positive definite covariance matrix
    A = np.random.randn(n, n) * 0.01
    sigma_np = A @ A.T + np.eye(n) * 0.001
    betahat = torch.tensor(betahat_np, dtype=torch.float64)
    sigma = torch.tensor(sigma_np, dtype=torch.float64)
    print(f"  Synthetic betahat: {betahat_np.round(4)}")

l_vec = hd.basis_vector(1, numPost)  # effect for first post-treatment period
alpha = 0.05

# =============================================================================
# Test 1: Original Confidence Set
# =============================================================================
print("\n--- 1. Original Confidence Set ---")
orig = hd.constructOriginalCS(
    betahat=betahat, sigma=sigma,
    numPrePeriods=numPre, numPostPeriods=numPost,
    l_vec=l_vec, alpha=alpha
)
print(f"  Original CS: [{float(orig['lb'].iloc[0]):.6f}, {float(orig['ub'].iloc[0]):.6f}]")

# =============================================================================
# Test 2: FLCI (Fixed-Length Confidence Interval)
# =============================================================================
print("\n--- 2. Optimal FLCI ---")
flci = hd.find_optimal_flci(
    betahat=betahat, sigma=sigma,
    numPrePeriods=numPre, numPostPeriods=numPost,
    l_vec=l_vec, alpha=alpha
)
print(f"  Half-length: {flci['optimalHalfLength']:.6f}")
print(f"  FLCI: [{flci['FLCI'][0]:.6f}, {flci['FLCI'][1]:.6f}]")

# =============================================================================
# Test 3: Sensitivity Analysis - Smoothness (DeltaSD)
# =============================================================================
print("\n--- 3. Sensitivity Results (DeltaSD - Smoothness) ---")
sens_sd = hd.createSensitivityResults(
    betahat=betahat, sigma=sigma,
    numPrePeriods=numPre, numPostPeriods=numPost,
    l_vec=l_vec, alpha=alpha,
    Mvec=[0, 0.01, 0.02, 0.03]
)
print(sens_sd.to_string(index=False))

# =============================================================================
# Test 4: Sensitivity Analysis - Relative Magnitudes (DeltaRM)
# =============================================================================
print("\n--- 4. Sensitivity Results (DeltaRM - Relative Magnitudes) ---")
sens_rm = hd.createSensitivityResults_relativeMagnitudes(
    betahat=betahat, sigma=sigma,
    numPrePeriods=numPre, numPostPeriods=numPost,
    l_vec=l_vec, alpha=alpha,
    Mbarvec=[0.5, 1.0, 1.5, 2.0]
)
print(sens_rm.to_string(index=False))

# =============================================================================
# Test 5: Upper Bound for M
# =============================================================================
print("\n--- 5. DeltaSD Upper Bound for M ---")
ub_m = hd.DeltaSD_upperBound_Mpre(
    betahat=betahat, sigma=sigma,
    numPrePeriods=numPre, alpha=alpha
)
print(f"  Upper bound M: {ub_m:.6f}")

# =============================================================================
# Test 6: Conditional CS with DeltaSD
# =============================================================================
print("\n--- 6. Conditional CS - DeltaSD (M=0) ---")
cs_sd = hd.computeConditionalCS_DeltaSD(
    betahat=betahat, sigma=sigma,
    numPrePeriods=numPre, numPostPeriods=numPost,
    l_vec=l_vec, alpha=alpha, M=0,
    hybrid_flag="FLCI"
)
accepted = cs_sd[cs_sd['accept'] == 1]['grid']
if len(accepted) > 0:
    print(f"  CS: [{accepted.min():.6f}, {accepted.max():.6f}]")
else:
    print("  Empty set")

print("\n" + "=" * 60)
print("All tests completed!")
print("=" * 60)
