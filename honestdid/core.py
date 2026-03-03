"""
HonestDiD: Robust Inference in Difference-in-Differences and Event Study Designs

Python port (with PyTorch) of the R package by Ashesh Rambachan & Jonathan Roth.
Implements methods from Rambachan & Roth (2023) "A More Credible Approach to Parallel Trends".

Dependencies: torch, numpy, scipy, cvxpy, sympy, pandas
"""

import warnings
import math
import numpy as np
import torch
import pandas as pd
import cvxpy as cp
from scipy import optimize as sciopt
from scipy import stats as scistats

# Use float64 throughout for numerical precision matching R
torch.set_default_dtype(torch.float64)

# =========================================================================
# PHASE 1: UTILITIES + DELTA UTILITIES
# =========================================================================

# --- Utility Functions (from utilities.R) ---

def basis_vector(index=1, size=1):
    """Create a basis vector with 1 at position `index` (1-indexed, matching R)."""
    v = torch.zeros(size, 1)
    v[index - 1] = 1.0
    return v


def _selection_mat(selection, size, select="columns"):
    """
    Create a selection matrix.
    `selection` is a list of 1-indexed positions.
    """
    if isinstance(selection, (int, np.integer)):
        selection = [selection]
    k = len(selection)
    if select == "rows":
        m = torch.zeros(k, size)
        for i, s in enumerate(selection):
            m[i, s - 1] = 1.0
    else:
        m = torch.zeros(size, k)
        for i, s in enumerate(selection):
            m[s - 1, i] = 1.0
    return m


def _lee_cfn(eta, Sigma):
    """Compute c = Sigma @ eta / (eta' @ Sigma @ eta)."""
    num = Sigma @ eta
    denom = (eta.T @ Sigma @ eta).item()
    return num / denom


def _vlo_vup_fn(eta, Sigma, A, b, z):
    """Compute VLo and VUp truncation bounds for the ARP test."""
    c = _lee_cfn(eta, Sigma).reshape(-1)
    Ac = (A @ c).reshape(-1)
    objective = (b.reshape(-1) - (A @ z).reshape(-1)) / Ac

    neg_idx = (Ac < 0).squeeze()
    pos_idx = (Ac > 0).squeeze()

    if neg_idx.sum() == 0:
        VLo = -float('inf')
    else:
        VLo = objective[neg_idx].max().item()

    if pos_idx.sum() == 0:
        VUp = float('inf')
    else:
        VUp = objective[pos_idx].min().item()

    return VLo, VUp


def _warn_if_not_symm_psd(sigma):
    """Validate sigma is symmetric PSD."""
    sigma_np = sigma.numpy() if isinstance(sigma, torch.Tensor) else sigma
    asym = np.max(np.abs(sigma_np - sigma_np.T))
    if asym > 1e-12:
        warnings.warn(f"matrix sigma not exactly symmetric (largest asymmetry was {asym:g})")
    eigvals = np.linalg.eigvalsh(sigma_np)
    if np.any(eigvals < 0):
        warnings.warn(f"matrix sigma not numerically positive semi-definite (smallest eigenvalue was {np.min(eigvals):g})")


def _stop_if_not_conformable(betahat, sigma, numPrePeriods, numPostPeriods, l_vec):
    """Validate input dimensions."""
    betaL = betahat.numel()
    if sigma.shape[0] != sigma.shape[1]:
        raise ValueError(f"expected a square matrix but sigma was {sigma.shape[0]} by {sigma.shape[1]}")
    if sigma.shape[0] != betaL:
        raise ValueError(f"betahat ({betaL}) and sigma ({sigma.shape[0]} by {sigma.shape[1]}) were non-conformable")
    if numPrePeriods + numPostPeriods != betaL:
        raise ValueError(f"betahat ({betaL}) and pre + post periods ({numPrePeriods} + {numPostPeriods}) were non-conformable")
    if l_vec.numel() != numPostPeriods:
        raise ValueError(f"l_vec (length {l_vec.numel()}) and post periods ({numPostPeriods}) were non-conformable")


def _ensure_tensor(x):
    """Convert numpy array or list to torch tensor if needed."""
    if isinstance(x, torch.Tensor):
        return x.double()
    return torch.tensor(np.asarray(x), dtype=torch.float64)


def _to_col(x):
    """Ensure x is a column vector (2D)."""
    x = _ensure_tensor(x)
    if x.dim() == 1:
        return x.unsqueeze(1)
    return x


# --- Delta Utility Functions (from delta_utility_functions.R) ---

def _create_A_M(numPrePeriods, numPostPeriods, monotonicityDirection, postPeriodMomentsOnly=False):
    """
    Create monotonicity constraint matrix. A @ delta <= 0 implies delta is
    increasing/decreasing depending on direction.
    """
    n = numPrePeriods + numPostPeriods
    A_M = torch.zeros(n, n)
    # Pre-period rows: delta_r - delta_{r+1} <= 0 (increasing)
    for r in range(numPrePeriods - 1):
        A_M[r, r] = 1.0
        A_M[r, r + 1] = -1.0
    # Row at numPrePeriods-1: delta_{numPre-1} <= 0
    A_M[numPrePeriods - 1, numPrePeriods - 1] = 1.0
    # Post-period rows
    if numPostPeriods > 0:
        A_M[numPrePeriods, numPrePeriods] = -1.0
        if numPostPeriods > 1:
            for r in range(numPrePeriods + 1, n):
                A_M[r, r - 1] = 1.0
                A_M[r, r] = -1.0

    # If postPeriodMomentsOnly, exclude rows that only involve pre-periods
    if postPeriodMomentsOnly:
        post_cols = list(range(numPrePeriods, n))
        keep = []
        for i in range(A_M.shape[0]):
            if A_M[i, post_cols].abs().sum() > 0:
                keep.append(i)
        A_M = A_M[keep, :]

    if monotonicityDirection == "decreasing":
        A_M = -A_M
    elif monotonicityDirection != "increasing":
        raise ValueError("direction must be 'increasing' or 'decreasing'")
    return A_M


def _create_A_B(numPrePeriods, numPostPeriods, biasDirection):
    """Create sign restriction matrix for bias direction."""
    n = numPrePeriods + numPostPeriods
    A_B = -torch.eye(n)
    A_B = A_B[numPrePeriods:, :]  # Keep only post-period rows

    if biasDirection == "negative":
        A_B = -A_B
    elif biasDirection != "positive":
        raise ValueError("biasDirection must be 'positive' or 'negative'")
    return A_B


# =========================================================================
# PHASE 2a: FLCI FUNCTIONS (from flci.R)
# =========================================================================

def _qfoldednormal(p, mu=0.0, sd=1.0, numSims=10**6, seed=0):
    """Compute pth quantile of folded normal distribution. Vectorized over mu."""
    rng = np.random.RandomState(seed)
    draws = rng.normal(scale=sd, size=numSims)
    if isinstance(mu, (list, np.ndarray, torch.Tensor)):
        mu_arr = np.asarray(mu).ravel()
        return np.array([np.quantile(np.abs(draws + m), p) for m in mu_arr])
    else:
        return np.quantile(np.abs(draws + mu), p)


def _w_to_l_fn(w):
    """Convert vector from w space to l space."""
    K = len(w)
    W2L = torch.eye(K)
    if K > 1:
        for col in range(K - 1):
            W2L[col + 1, col] = -1.0
    return (W2L @ _to_col(w)).squeeze()


def _l_to_w_fn(l_vec):
    """Convert vector from l space to w space."""
    K = l_vec.numel()
    L2W = torch.eye(K)
    if K > 1:
        for col in range(K - 1):
            L2W[col + 1, col] = 1.0
    return (L2W @ _to_col(l_vec)).squeeze()


def _create_matrices_for_variance_from_w(sigma, numPrePeriods, l_vec):
    """Construct matrices for variance computation from weights w."""
    sigma = _ensure_tensor(sigma)
    l_vec = _to_col(_ensure_tensor(l_vec))
    pre_idx = list(range(numPrePeriods))
    post_idx = list(range(numPrePeriods, sigma.shape[0]))

    SigmaPre = sigma[pre_idx][:, pre_idx]
    SigmaPrePost = sigma[pre_idx][:, post_idx]
    SigmaPost = l_vec.T @ sigma[post_idx][:, post_idx] @ l_vec

    K = numPrePeriods
    WtoLPreMat = torch.eye(K)
    if K > 1:
        for col in range(K - 1):
            WtoLPreMat[col + 1, col] = -1.0

    zeros_left = torch.zeros(K, K)
    UstackWtoLPreMat = torch.cat([zeros_left, WtoLPreMat], dim=1)

    A_quad = UstackWtoLPreMat.T @ SigmaPre @ UstackWtoLPreMat
    A_lin = 2.0 * UstackWtoLPreMat.T @ SigmaPrePost @ l_vec
    A_const = SigmaPost

    return A_quad, A_lin, A_const


def _compute_sigma_l_from_w(w, sigma, numPrePeriods, numPostPeriods, l_vec):
    """Compute variance of affine estimator for choice of weights w."""
    w = _ensure_tensor(w).reshape(-1)
    A_quad, A_lin, A_const = _create_matrices_for_variance_from_w(sigma, numPrePeriods, l_vec)
    Uw = torch.cat([torch.zeros(len(w)), w]).unsqueeze(1)
    var_l = (Uw.T @ A_quad @ Uw + A_lin.T @ Uw + A_const).item()
    return var_l


def _find_worst_case_bias_given_h(h, sigma, numPrePeriods, numPostPeriods, l_vec, M=1.0, returnDF=False):
    """Minimize worst-case bias over Delta^SD(M) subject to SD(estimator) <= h."""
    sigma = _ensure_tensor(sigma)
    l_vec = _to_col(_ensure_tensor(l_vec))
    K = numPrePeriods
    n_post = numPostPeriods
    dim = 2 * K  # [U; W]

    # cvxpy variable
    UW = cp.Variable(dim)

    # Absolute value constraints: |cumsum(w)| <= U
    lowerTriMat = np.tril(np.ones((K, K)))
    I_K = np.eye(K)
    A_abs = np.vstack([
        np.hstack([-I_K, lowerTriMat]),
        np.hstack([-I_K, -lowerTriMat])
    ])
    abs_constraint = A_abs @ UW <= 0

    # Sum-of-weights constraint
    sum_coef = np.zeros(dim)
    sum_coef[K:] = 1.0
    l_np = l_vec.numpy().ravel()
    sum_target = float(np.arange(1, n_post + 1) @ l_np)
    sum_constraint = sum_coef @ UW == sum_target

    # Variance constraint (quadratic): UW' A_quad UW + A_lin' UW + A_const <= h^2
    A_quad, A_lin, A_const = _create_matrices_for_variance_from_w(sigma, numPrePeriods, l_vec)
    A_quad_np = A_quad.numpy()
    A_lin_np = A_lin.numpy().ravel()
    A_const_val = A_const.item()

    # Make A_quad_np symmetric PSD for cvxpy
    A_quad_np = (A_quad_np + A_quad_np.T) / 2.0
    eigvals = np.linalg.eigvalsh(A_quad_np)
    if np.min(eigvals) < -1e-10:
        A_quad_np += (-np.min(eigvals) + 1e-10) * np.eye(dim)
    elif np.min(eigvals) < 0:
        A_quad_np += 1e-10 * np.eye(dim)

    quad_constraint = cp.quad_form(UW, A_quad_np) + A_lin_np @ UW + A_const_val <= h ** 2

    # Objective: minimize bias
    # constant + sum(U_i)
    constant = 0.0
    for s in range(1, n_post + 1):
        constant += abs(float(np.arange(1, s + 1) @ l_np[(n_post - s):]))
    constant -= float(np.arange(1, n_post + 1) @ l_np)

    obj_coef = np.zeros(dim)
    obj_coef[:K] = 1.0
    objective = cp.Minimize(constant + obj_coef @ UW)

    prob = cp.Problem(objective, [abs_constraint, sum_constraint, quad_constraint])
    for _solver in [cp.CLARABEL, cp.ECOS, cp.SCS]:
        try:
            prob.solve(solver=_solver)
            if prob.status in ("optimal", "optimal_inaccurate"):
                break
        except (cp.SolverError, Exception):
            continue

    status = prob.status
    if status in ("optimal", "optimal_inaccurate"):
        val = prob.value * M
        opt_x = UW.value
        opt_w = opt_x[K:]
        opt_l = _w_to_l_fn(torch.tensor(opt_w)).numpy()
    else:
        val = float('inf')
        opt_x = np.zeros(dim)
        opt_w = np.zeros(K)
        opt_l = np.zeros(K)

    return {
        'status': status,
        'value': val / M if M != 0 else val,  # store un-scaled bias
        'value_scaled': val,
        'optimal_x': opt_x,
        'optimal_w': opt_w,
        'optimal_l': opt_l
    }


def _find_lowest_h(sigma, numPrePeriods, numPostPeriods, l_vec, sigmascale=10, maxscale=10):
    """Find the minimum variance affine estimator (lowest h)."""
    sigma = _ensure_tensor(sigma)
    l_vec = _to_col(_ensure_tensor(l_vec))
    K = numPrePeriods
    dim = 2 * K

    def _try_solve(sig):
        UW = cp.Variable(dim)
        # Absolute value constraints
        lowerTriMat = np.tril(np.ones((K, K)))
        I_K = np.eye(K)
        A_abs = np.vstack([np.hstack([-I_K, lowerTriMat]), np.hstack([-I_K, -lowerTriMat])])
        abs_constr = A_abs @ UW <= 0
        # Sum constraint
        sum_coef = np.zeros(dim)
        sum_coef[K:] = 1.0
        l_np = l_vec.numpy().ravel()
        sum_target = float(np.arange(1, numPostPeriods + 1) @ l_np)
        sum_constr = sum_coef @ UW == sum_target
        # Objective: minimize variance
        A_quad, A_lin, A_const = _create_matrices_for_variance_from_w(sig, numPrePeriods, l_vec)
        A_quad_np = A_quad.numpy()
        A_quad_np = (A_quad_np + A_quad_np.T) / 2.0
        eigv = np.linalg.eigvalsh(A_quad_np)
        if np.min(eigv) < 0:
            A_quad_np += (-np.min(eigv) + 1e-10) * np.eye(dim)
        A_lin_np = A_lin.numpy().ravel()
        A_const_val = A_const.item()
        obj = cp.Minimize(cp.quad_form(UW, A_quad_np) + A_lin_np @ UW + A_const_val)
        prob = cp.Problem(obj, [abs_constr, sum_constr])
        try:
            prob.solve(solver=cp.ECOS)
        except cp.SolverError:
            prob.solve(solver=cp.SCS)
        return prob

    prob = _try_solve(sigma)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        for iscale in range(1, maxscale + 1):
            if iscale > math.ceil(maxscale / 2):
                scaled = sigmascale ** (math.ceil(maxscale / 2) - iscale)
            else:
                scaled = sigmascale ** iscale
            prob = _try_solve(scaled * sigma)
            if prob.status in ("optimal", "optimal_inaccurate"):
                minVar = prob.value / (sigmascale ** iscale)
                return math.sqrt(max(minVar, 0))
        warnings.warn("Error in optimization for h0")
        return 0.0
    else:
        return math.sqrt(max(prob.value, 0))


def _find_h_for_minimum_bias(sigma, numPrePeriods, numPostPeriods, l_vec):
    """Compute h when w sets all weight on the sum constraint."""
    l_vec = _ensure_tensor(l_vec).reshape(-1)
    w = torch.zeros(numPrePeriods)
    sum_val = float(torch.arange(1, numPostPeriods + 1, dtype=torch.float64) @ l_vec)
    w[-1] = sum_val
    hsq = _compute_sigma_l_from_w(w, sigma, numPrePeriods, numPostPeriods, l_vec)
    return math.sqrt(max(hsq, 0))


def _find_optimal_ci_derivative_bisection(a, b, M, numPoints, alpha, sigma, numPrePeriods, numPostPeriods, l_vec, returnDF, seed=0):
    """Bisection search for optimal h minimizing CI half-length."""
    def f(h):
        res = _find_worst_case_bias_given_h(h, sigma, numPrePeriods, numPostPeriods, l_vec, M=1.0, returnDF=False)
        maxBias = M * res['value']
        if res['value'] < float('inf') and res['status'] in ("optimal", "optimal_inaccurate"):
            return float(_qfoldednormal(1 - alpha, mu=maxBias / h, seed=seed) * h)
        else:
            return float('nan')

    eps = np.finfo(float).eps
    failtol = max(eps ** 0.5, 2.0)  # tolerant for solver noise in finite differences
    dif = (b - a) / numPoints  # use wider steps for stable finite differences
    fa = f(a)
    fb = f(b)
    if math.isnan(fa) or math.isnan(fb):
        return float('nan')
    fpa = (f(a + dif) - fa) / dif
    fpb = (f(b - dif) - fb) / (-dif)
    maxiter = int(10 * math.ceil(math.log(max(abs(b - a) / max(dif, 1e-15), 1)) / math.log(2)))
    if math.isnan(fpa) or math.isnan(fpb):
        return float('nan')
    if fpa > fpb:
        return float('nan')
    if fpb < 0:
        return b
    if fpa > 0:
        return a

    failed = False
    it = 1
    while not failed and abs(b - a) > dif:
        it += 1
        x = (a + b) / 2
        fx_plus = f(x + dif)
        fx_minus = f(x - dif)
        if math.isnan(fx_plus) or math.isnan(fx_minus):
            failed = True
            break
        fpx = (fx_plus - fx_minus) / (2 * dif)
        failed = (fpx > fpb + failtol) or (fpx + failtol < fpa) or it > maxiter
        if fpx > 0:
            b = x
        else:
            a = x
    hstar = (a + b) / 2
    return float('nan') if failed else hstar


def _find_optimal_flci_helper(sigma, M, numPrePeriods, numPostPeriods, l_vec, numPoints=100, alpha=0.05, seed=0):
    """Core FLCI computation helper."""
    sigma = _ensure_tensor(sigma)
    l_vec = _ensure_tensor(l_vec).reshape(-1)

    h0 = _find_h_for_minimum_bias(sigma, numPrePeriods, numPostPeriods, l_vec)
    hMin = _find_lowest_h(sigma, numPrePeriods, numPostPeriods, l_vec)
    hstar = _find_optimal_ci_derivative_bisection(hMin, h0, M, numPoints, alpha, sigma, numPrePeriods, numPostPeriods, l_vec, True, seed=seed)

    if math.isnan(hstar):
        # Fall back to grid search with finer resolution
        hGrid = np.linspace(hMin, h0, max(numPoints, 1000))
        best_hl = float('inf')
        best_res = None
        best_h = hMin
        for h in hGrid:
            res = _find_worst_case_bias_given_h(h, sigma, numPrePeriods, numPostPeriods, l_vec, M=1.0)
            if res['status'] in ("optimal", "optimal_inaccurate") and res['value'] < float('inf'):
                maxBias = M * res['value']
                hl = float(_qfoldednormal(1 - alpha, mu=maxBias / max(h, 1e-15), seed=seed) * h)
                if hl < best_hl:
                    best_hl = hl
                    best_res = res
                    best_h = h
        if best_res is None:
            best_res = _find_worst_case_bias_given_h(hMin, sigma, numPrePeriods, numPostPeriods, l_vec, M=1.0)
            best_hl = float(_qfoldednormal(1 - alpha, mu=0.0, seed=seed) * hMin)
        optimalVec = np.concatenate([best_res['optimal_l'], l_vec.numpy()])
        return {
            'optimalVec': optimalVec,
            'optimalPrePeriodVec': best_res['optimal_l'],
            'optimalHalfLength': best_hl,
            'M': M,
            'status': best_res['status']
        }
    else:
        res = _find_worst_case_bias_given_h(hstar, sigma, numPrePeriods, numPostPeriods, l_vec, M=1.0, returnDF=True)
        maxBias = M * res['value']
        hl = float(_qfoldednormal(1 - alpha, mu=maxBias / max(hstar, 1e-15), seed=seed) * hstar)
        optimalVec = np.concatenate([res['optimal_l'], l_vec.numpy()])
        return {
            'optimalVec': optimalVec,
            'optimalPrePeriodVec': res['optimal_l'],
            'optimalHalfLength': hl,
            'M': M,
            'status': res['status']
        }


def find_optimal_flci(betahat, sigma, M=0.0, numPrePeriods=None, numPostPeriods=None,
                      l_vec=None, numPoints=100, alpha=0.05, seed=0):
    """
    Compute optimal Fixed-Length Confidence Interval for Delta^SD(M).

    Returns dict with keys: FLCI (tuple), optimalVec, optimalHalfLength, M, status.
    """
    betahat = _ensure_tensor(betahat).reshape(-1)
    sigma = _ensure_tensor(sigma)
    if l_vec is None:
        l_vec = basis_vector(index=1, size=numPostPeriods).squeeze()
    else:
        l_vec = _ensure_tensor(l_vec).reshape(-1)

    res = _find_optimal_flci_helper(sigma, M, numPrePeriods, numPostPeriods, l_vec, numPoints, alpha, seed)
    optVec = torch.tensor(res['optimalVec'])
    pe = (optVec @ betahat).item()
    FLCI = (pe - res['optimalHalfLength'], pe + res['optimalHalfLength'])

    return {
        'FLCI': FLCI,
        'optimalVec': res['optimalVec'],
        'optimalHalfLength': res['optimalHalfLength'],
        'M': res['M'],
        'status': res['status']
    }


# =========================================================================
# PHASE 2b: ARP TESTS (from arp-nuisance.R and arp-nonuisance.R)
# =========================================================================

# --- ARP Helper Functions ---

def _norminvp_generalized(p, l, u, mu=0.0, sd=1.0):
    """Generalized truncated normal quantile function."""
    sd = float(sd)
    mu = float(mu)
    l = float(l)
    u = float(u)
    if sd <= 0:
        return mu
    lnorm = (l - mu) / sd
    unorm = (u - mu) / sd
    if lnorm >= unorm:
        return mu + lnorm * sd
    if np.isinf(lnorm) and lnorm < 0 and np.isinf(unorm) and unorm > 0:
        return float(scistats.norm.ppf(p) * sd + mu)
    qnorm = scistats.truncnorm.ppf(p, a=lnorm, b=unorm)
    return float(mu + qnorm * sd)


def _roundeps(x, eps=None):
    """Round values close to zero to zero."""
    if eps is None:
        eps = np.finfo(float).eps ** 0.75
    return 0.0 if abs(x) < eps else float(x)


def _max_program(s_T, gamma_tilde, sigma, W_T, c_val):
    """LP for dual bisection: max f'x s.t. W_T'x = beq, x >= 0."""
    s_T = np.asarray(s_T, dtype=float).ravel()
    gamma_tilde = np.asarray(gamma_tilde, dtype=float).ravel()
    sigma_np = np.asarray(sigma, dtype=float)
    W_T_np = np.asarray(W_T, dtype=float)

    gt_sig_gt = float(gamma_tilde @ sigma_np @ gamma_tilde)
    if abs(gt_sig_gt) < 1e-30:
        f = s_T.copy()
    else:
        f = s_T + (1.0 / gt_sig_gt) * (sigma_np @ gamma_tilde) * c_val

    n = len(f)
    Aeq = W_T_np.T
    beq = np.zeros(Aeq.shape[0])
    beq[0] = 1.0

    result = sciopt.linprog(c=-f, A_eq=Aeq, b_eq=beq,
                            bounds=[(0, None)] * n, method='highs')
    return {
        'optimum': -result.fun if result.success else float('nan'),
        'solution': result.x if result.success else np.zeros(n),
        'success': result.success
    }


def _check_if_solution_helper(c_val, tol, s_T, gamma_tilde, sigma, W_T):
    """Check if c_val is a valid solution to the max program."""
    lp = _max_program(s_T, gamma_tilde, sigma, W_T, c_val)
    lp['honestsolution'] = lp['success'] and (abs(c_val - lp['optimum']) <= tol)
    return lp


def _vlo_vup_dual_fn(eta, s_T, gamma_tilde, sigma, W_T):
    """Compute vlo, vup for dual LP using bisection (Algorithm 1, ARP Appendix D)."""
    sigma_np = np.asarray(sigma, dtype=float)
    gamma_np = np.asarray(gamma_tilde, dtype=float).ravel()
    s_T_np = np.asarray(s_T, dtype=float).ravel()
    W_T_np = np.asarray(W_T, dtype=float)

    tol_c = 1e-6
    tol_equality = 1e-6
    gt_sig_gt = float(gamma_np @ sigma_np @ gamma_np)
    if abs(gt_sig_gt) < 1e-30:
        return {'vlo': -float('inf'), 'vup': float('inf')}
    sigma_B = float(np.sqrt(gt_sig_gt))
    low_initial = min(-100.0, eta - 20 * sigma_B)
    high_initial = max(100.0, eta + 20 * sigma_B)
    maxiters = 10000
    switchiters = 10

    checksol = _check_if_solution_helper(eta, tol_equality, s_T_np, gamma_np, sigma_np, W_T_np)
    if not checksol['honestsolution']:
        return {'vlo': eta, 'vup': float('inf')}

    b_vec = (1.0 / gt_sig_gt) * (sigma_np @ gamma_np)

    # --- Compute vup ---
    lp = _check_if_solution_helper(high_initial, tol_equality, s_T_np, gamma_np, sigma_np, W_T_np)
    if lp['honestsolution']:
        vup = float('inf')
    else:
        dif = 0.0
        iters = 1
        denom = 1.0 - float(lp['solution'] @ b_vec)
        mid = _roundeps(float(lp['solution'] @ s_T_np)) / denom if abs(denom) > 1e-15 else high_initial

        lp = _check_if_solution_helper(mid, tol_equality, s_T_np, gamma_np, sigma_np, W_T_np)
        while not lp['honestsolution'] and iters < maxiters:
            iters += 1
            if iters >= switchiters:
                dif = tol_c + 1
                break
            denom = 1.0 - float(lp['solution'] @ b_vec)
            mid = _roundeps(float(lp['solution'] @ s_T_np)) / denom if abs(denom) > 1e-15 else mid
            lp = _check_if_solution_helper(mid, tol_equality, s_T_np, gamma_np, sigma_np, W_T_np)

        low = eta
        high = mid
        while dif > tol_c and iters < maxiters:
            iters += 1
            mid = (high + low) / 2
            if _check_if_solution_helper(mid, tol_equality, s_T_np, gamma_np, sigma_np, W_T_np)['honestsolution']:
                low = mid
            else:
                high = mid
            dif = high - low
        vup = mid

    # --- Compute vlo ---
    lp = _check_if_solution_helper(low_initial, tol_equality, s_T_np, gamma_np, sigma_np, W_T_np)
    if lp['honestsolution']:
        vlo = float('-inf')
    else:
        dif = 0.0
        iters = 1
        denom = 1.0 - float(lp['solution'] @ b_vec)
        mid = _roundeps(float(lp['solution'] @ s_T_np)) / denom if abs(denom) > 1e-15 else low_initial

        lp = _check_if_solution_helper(mid, tol_equality, s_T_np, gamma_np, sigma_np, W_T_np)
        while not lp['honestsolution'] and iters < maxiters:
            iters += 1
            if iters >= switchiters:
                dif = tol_c + 1
                break
            denom = 1.0 - float(lp['solution'] @ b_vec)
            mid = _roundeps(float(lp['solution'] @ s_T_np)) / denom if abs(denom) > 1e-15 else mid
            lp = _check_if_solution_helper(mid, tol_equality, s_T_np, gamma_np, sigma_np, W_T_np)

        low_v = mid
        high_v = eta
        while dif > tol_c and iters < maxiters:
            mid = (low_v + high_v) / 2
            iters += 1
            if _check_if_solution_helper(mid, tol_equality, s_T_np, gamma_np, sigma_np, W_T_np)['honestsolution']:
                high_v = mid
            else:
                low_v = mid
            dif = high_v - low_v
        vlo = mid

    return {'vlo': vlo, 'vup': vup}


# --- LP Solver and Dual Functions ---

def _test_delta_lp_fn(y_T, X_T, sigma):
    """
    Solve min eta s.t. y_T - X_T*delta <= eta * sqrt(diag(sigma)).
    Returns eta_star, delta_star, lambda (dual), error_flag.
    """
    y_T = np.asarray(y_T, dtype=float).ravel()
    sigma_np = np.asarray(sigma, dtype=float)
    X_T = np.asarray(X_T, dtype=float)
    if X_T.ndim == 1:
        X_T = X_T.reshape(-1, 1)

    dimDelta = X_T.shape[1]
    sdVec = np.sqrt(np.diag(sigma_np))
    M = len(y_T)

    # Objective: min [1, 0, ..., 0]' @ [eta, delta]
    f = np.zeros(1 + dimDelta)
    f[0] = 1.0

    # Constraints: -sdVec*eta - X_T*delta <= -y_T
    C = np.column_stack([-sdVec, -X_T])
    b = -y_T

    result = sciopt.linprog(c=f, A_ub=C, b_ub=b,
                            bounds=[(None, None)] * (1 + dimDelta),
                            method='highs')

    if result.success:
        eta_star = result.x[0]
        delta_star = result.x[1:]
        # Dual variables: negate scipy marginals to get non-negative λ
        dual = -result.ineqlin.marginals if hasattr(result, 'ineqlin') else np.zeros(M)
        error_flag = 0
    else:
        eta_star = float('nan')
        delta_star = np.zeros(dimDelta)
        dual = np.zeros(M)
        error_flag = 1

    return {
        'eta_star': eta_star,
        'delta_star': delta_star,
        'lambda': dual,
        'error_flag': error_flag
    }


def _lp_dual_fn(y_T, X_T, eta, gamma_tilde, sigma):
    """Wrapper to compute vlo, vup using dual bisection approach."""
    y_T = np.asarray(y_T, dtype=float).ravel()
    sigma_np = np.asarray(sigma, dtype=float)
    X_T = np.asarray(X_T, dtype=float)
    if X_T.ndim == 1:
        X_T = X_T.reshape(-1, 1)
    gamma_tilde = np.asarray(gamma_tilde, dtype=float).ravel()

    sdVec = np.sqrt(np.diag(sigma_np))
    W_T = np.column_stack([sdVec, X_T])
    gt_sig_gt = float(gamma_tilde @ sigma_np @ gamma_tilde)
    if abs(gt_sig_gt) < 1e-30:
        s_T = y_T.copy()
    else:
        proj = (sigma_np @ (np.outer(gamma_tilde, gamma_tilde))) / gt_sig_gt
        s_T = (np.eye(len(y_T)) - proj) @ y_T

    vList = _vlo_vup_dual_fn(eta=eta, s_T=s_T, gamma_tilde=gamma_tilde,
                             sigma=sigma_np, W_T=W_T)
    return {'vlo': vList['vlo'], 'vup': vList['vup'],
            'eta': eta, 'gamma_tilde': gamma_tilde}


def _construct_Gamma(l_vec):
    """
    Construct invertible matrix Gamma with l' as first row, using RREF.
    Uses sympy for reduced row echelon form.
    """
    import sympy
    l = np.asarray(l_vec, dtype=float).ravel()
    barT = len(l)
    B = np.column_stack([l.reshape(-1, 1), np.eye(barT)])
    B_sym = sympy.Matrix(B)
    rref_B, pivots = B_sym.rref()
    pivots = list(pivots)
    Gamma = B[:, pivots].T
    if abs(np.linalg.det(Gamma)) < 1e-14:
        raise ValueError("Something went wrong in RREF algorithm.")
    return Gamma


def _max_bias_fn(v, A, d):
    """Compute maximum bias max v'delta s.t. A*delta <= d."""
    v = np.asarray(v, dtype=float).ravel()
    A_np = np.asarray(A, dtype=float)
    d_np = np.asarray(d, dtype=float).ravel()
    n = A_np.shape[1]
    delta = cp.Variable(n)
    objective = cp.Maximize(v @ delta)
    prob = cp.Problem(objective, [A_np @ delta <= d_np])
    prob.solve(solver=cp.ECOS)
    if prob.status in ('infeasible', 'unbounded'):
        prob.solve(solver=cp.SCS)
    return {'value': prob.value, 'status': prob.status}


def _min_bias_fn(v, A, d):
    """Compute minimum bias min v'delta s.t. A*delta <= d."""
    v = np.asarray(v, dtype=float).ravel()
    A_np = np.asarray(A, dtype=float)
    d_np = np.asarray(d, dtype=float).ravel()
    n = A_np.shape[1]
    delta = cp.Variable(n)
    objective = cp.Minimize(v @ delta)
    prob = cp.Problem(objective, [A_np @ delta <= d_np])
    prob.solve(solver=cp.ECOS)
    if prob.status in ('infeasible', 'unbounded'):
        prob.solve(solver=cp.SCS)
    return {'value': prob.value, 'status': prob.status}


def _create_v_linear_no_intercept(l_vec, t_vec, referencePeriod=0):
    """Create vector that extrapolates a linear trend to pre-period."""
    l_vec = np.asarray(l_vec, dtype=float).ravel()
    t_vec = np.asarray(t_vec, dtype=float).ravel()
    relativeTVec = t_vec - referencePeriod
    tPre = relativeTVec[relativeTVec < 0]
    tPost = relativeTVec[relativeTVec > 0]
    slopePre = (tPre / (tPre @ tPre)).reshape(-1, 1)
    l_trend_NI = np.concatenate([
        -float(l_vec @ tPost) * slopePre.ravel(),
        l_vec
    ])
    return l_trend_NI


def _create_constraints_linear_trend(A, d, l_vec, t_vec, referencePeriod=0):
    """Create constraints related to max/min bias of linear estimator using pre-period."""
    v_trend = _create_v_linear_no_intercept(l_vec, t_vec, referencePeriod)
    maxBias = _max_bias_fn(v_trend, A, d)['value']
    minBias = _min_bias_fn(v_trend, A, d)['value']
    A_trend = np.vstack([v_trend, -v_trend])
    d_trend = np.array([maxBias, -minBias])
    return {'A_trend': A_trend, 'd_trend': d_trend}


# --- Hybrid Helper Functions ---

def _compute_least_favorable_cv(X_T, sigma, hybrid_kappa, sims=1000,
                                rowsForARP=None, seed=0):
    """Compute least favorable critical value (Section 6.2, ARP 2019)."""
    sigma_np = np.asarray(sigma, dtype=float)
    if rowsForARP is not None:
        idx = np.asarray(rowsForARP)
        if X_T is not None:
            X_T_np = np.asarray(X_T, dtype=float)
            if X_T_np.ndim == 1:
                X_T_np = X_T_np[idx].reshape(-1, 1)
            else:
                X_T_np = X_T_np[idx, :]
        else:
            X_T_np = None
        sigma_np = sigma_np[np.ix_(idx, idx)]
    else:
        X_T_np = np.asarray(X_T, dtype=float) if X_T is not None else None
        if X_T_np is not None and X_T_np.ndim == 1:
            X_T_np = X_T_np.reshape(-1, 1)

    rng = np.random.RandomState(seed)

    if X_T_np is None:
        # No nuisance parameter case
        xi_draws = rng.multivariate_normal(np.zeros(sigma_np.shape[0]), sigma_np, size=sims)
        sdVec = np.sqrt(np.diag(sigma_np))
        xi_draws = xi_draws / sdVec[np.newaxis, :]
        eta_vec = xi_draws.max(axis=1)
        return float(np.quantile(eta_vec, 1 - hybrid_kappa))
    else:
        # Nuisance parameter case
        sdVec = np.sqrt(np.diag(sigma_np))
        dimDelta = X_T_np.shape[1]
        f = np.zeros(1 + dimDelta)
        f[0] = 1.0
        C = np.column_stack([-sdVec, -X_T_np])
        xi_draws = rng.multivariate_normal(np.zeros(sigma_np.shape[0]), sigma_np, size=sims)

        eta_vec = []
        for i in range(sims):
            b = -xi_draws[i]
            result = sciopt.linprog(c=f, A_ub=C, b_ub=b,
                                    bounds=[(None, None)] * (1 + dimDelta),
                                    method='highs',
                                    options={'time_limit': 10})
            if result.success:
                eta_vec.append(result.x[0])
        if len(eta_vec) == 0:
            return float('inf')
        return float(np.quantile(eta_vec, 1 - hybrid_kappa))


def _FLCI_compute_vlo_vup(vbar, dbar, S, c_vec):
    """Compute Vlo, Vup modified for the FLCI hybrid."""
    vbar = np.asarray(vbar, dtype=float).ravel()
    dbar = np.asarray(dbar, dtype=float).ravel()
    S = np.asarray(S, dtype=float).ravel()
    c_vec = np.asarray(c_vec, dtype=float).ravel()

    VbarMat = np.vstack([vbar, -vbar])
    max_or_min = (dbar - VbarMat @ S) / (VbarMat @ c_vec)
    neg_idx = (VbarMat @ c_vec) < 0
    pos_idx = (VbarMat @ c_vec) > 0
    vlo = float(max_or_min[neg_idx].max()) if neg_idx.any() else float('-inf')
    vup = float(max_or_min[pos_idx].min()) if pos_idx.any() else float('inf')
    return {'vlo': vlo, 'vup': vup}


# --- Main ARP Test with Nuisance Parameters ---

def _lp_conditional_test_fn(theta, y_T, X_T, sigma, alpha,
                            hybrid_flag, hybrid_list, rowsForARP=None):
    """
    ARP test of moment inequality E[y_T - X_T*delta] <= 0.
    Returns reject indicator, eta, delta, lambda.
    """
    y_T = np.asarray(y_T, dtype=float).ravel()
    sigma_full = np.asarray(sigma, dtype=float)
    X_T_full = np.asarray(X_T, dtype=float)
    if X_T_full.ndim == 1:
        X_T_full = X_T_full.reshape(-1, 1)

    if rowsForARP is None:
        rowsForARP = np.arange(len(y_T))
    else:
        rowsForARP = np.asarray(rowsForARP)

    y_T_ARP = y_T[rowsForARP]
    X_T_ARP = X_T_full[rowsForARP, :]
    if X_T_ARP.ndim == 1:
        X_T_ARP = X_T_ARP.reshape(-1, 1)
    sigma_ARP = sigma_full[np.ix_(rowsForARP, rowsForARP)]

    M_dim = sigma_ARP.shape[0]
    k = X_T_ARP.shape[1]

    # Compute eta and argmin delta
    linSoln = _test_delta_lp_fn(y_T_ARP, X_T_ARP, sigma_ARP)

    if linSoln['error_flag'] > 0:
        warnings.warn("LP for eta did not converge. Not rejecting.")
        return {'reject': 0, 'eta': linSoln['eta_star'],
                'delta': linSoln['delta_star'], 'lambda': linSoln['lambda']}

    # HYBRID first-stage test
    if hybrid_flag == "LF":
        mod_size = (alpha - hybrid_list['hybrid_kappa']) / (1 - hybrid_list['hybrid_kappa'])
        if linSoln['eta_star'] > hybrid_list['lf_cv']:
            return {'reject': 1, 'eta': linSoln['eta_star'],
                    'delta': linSoln['delta_star'], 'lambda': linSoln['lambda']}
    elif hybrid_flag == "FLCI":
        mod_size = (alpha - hybrid_list['hybrid_kappa']) / (1 - hybrid_list['hybrid_kappa'])
        VbarMat = np.vstack([hybrid_list['vbar'], -hybrid_list['vbar']])
        if float(np.max(VbarMat @ y_T - hybrid_list['dbar'])) > 0:
            return {'reject': 1, 'eta': linSoln['eta_star'],
                    'delta': linSoln['delta_star'], 'lambda': linSoln['lambda']}
    elif hybrid_flag == "ARP":
        mod_size = alpha
    else:
        raise ValueError("hybrid_flag must be 'LF', 'FLCI', or 'ARP'")

    # Check degeneracy
    tol_lambda = 1e-6
    lam = linSoln['lambda']
    degenerate_flag = (np.sum(lam > tol_lambda) != (k + 1))
    B_index = lam > tol_lambda
    Bc_index = ~B_index

    X_TB = X_T_ARP[B_index, :]
    if X_TB.ndim == 1:
        X_TB = X_TB.reshape(-1, 1)
    Xdim = min(X_TB.shape)

    if X_TB.size == 0 or Xdim == 0:
        fullRank_flag = False
    else:
        fullRank_flag = (np.linalg.matrix_rank(X_TB) == Xdim)

    if not fullRank_flag or degenerate_flag:
        # --- Dual approach ---
        lpDualSoln = _lp_dual_fn(y_T_ARP, X_T_ARP, linSoln['eta_star'],
                                 linSoln['lambda'], sigma_ARP)
        gt = np.asarray(lpDualSoln['gamma_tilde']).ravel()
        sigma_B_dual2 = float(gt @ sigma_ARP @ gt)

        if abs(sigma_B_dual2) < np.finfo(float).eps:
            return {'reject': int(linSoln['eta_star'] > 0),
                    'eta': linSoln['eta_star'], 'delta': linSoln['delta_star'],
                    'lambda': linSoln['lambda']}
        elif sigma_B_dual2 < 0:
            raise ValueError("Negative variance in dual approach")

        sigma_B_dual = math.sqrt(sigma_B_dual2)
        maxstat = lpDualSoln['eta'] / sigma_B_dual

        # HYBRID: Modify vlo, vup
        if hybrid_flag == "LF":
            zlo_dual = lpDualSoln['vlo'] / sigma_B_dual
            zup_dual = min(lpDualSoln['vup'], hybrid_list['lf_cv']) / sigma_B_dual
        elif hybrid_flag == "FLCI":
            gamma_full = np.zeros(len(y_T))
            gamma_full[rowsForARP] = gt
            gf = gamma_full.reshape(-1, 1)
            sigma_gamma = (sigma_full @ gf) / float(gf.T @ sigma_full @ gf)
            S = y_T - (sigma_gamma @ (gf.T @ y_T)).ravel()
            vFLCI = _FLCI_compute_vlo_vup(hybrid_list['vbar'], hybrid_list['dbar'],
                                          S, sigma_gamma.ravel())
            zlo_dual = max(lpDualSoln['vlo'], vFLCI['vlo']) / sigma_B_dual
            zup_dual = min(lpDualSoln['vup'], vFLCI['vup']) / sigma_B_dual
        else:  # ARP
            zlo_dual = lpDualSoln['vlo'] / sigma_B_dual
            zup_dual = lpDualSoln['vup'] / sigma_B_dual

        if not (zlo_dual <= maxstat <= zup_dual):
            return {'reject': 0, 'eta': linSoln['eta_star'],
                    'delta': linSoln['delta_star'], 'lambda': linSoln['lambda']}
        else:
            cval = max(0.0, _norminvp_generalized(1 - mod_size, zlo_dual, zup_dual))
            reject = int(maxstat > cval)
            return {'reject': reject, 'eta': linSoln['eta_star'],
                    'delta': linSoln['delta_star'], 'lambda': linSoln['lambda']}
    else:
        # --- Primal approach ---
        size_B = int(np.sum(B_index))
        sdVec = np.sqrt(np.diag(sigma_ARP))
        sdVec_B = sdVec[B_index]
        sdVec_Bc = sdVec[Bc_index]
        X_TBc = X_T_ARP[Bc_index, :]
        S_B = np.eye(M_dim)[B_index, :]
        S_Bc = np.eye(M_dim)[Bc_index, :]

        cbind_B = np.column_stack([sdVec_B, X_TB])
        cbind_Bc = np.column_stack([sdVec_Bc, X_TBc])
        Gamma_B = cbind_Bc @ np.linalg.solve(cbind_B, S_B) - S_Bc
        e1 = np.zeros(size_B)
        e1[0] = 1.0
        v_B = (e1 @ np.linalg.solve(cbind_B, S_B)).reshape(-1, 1)
        sigma2_B = float(v_B.T @ sigma_ARP @ v_B)
        sigma_B = math.sqrt(sigma2_B)
        rho = (Gamma_B @ sigma_ARP @ v_B / sigma2_B).ravel()
        maximand = (-(Gamma_B @ y_T_ARP) / rho) + float(v_B.T @ y_T_ARP)

        # Truncation values
        vlo = float(maximand[rho > 0].max()) if np.any(rho > 0) else float('-inf')
        vup = float(maximand[rho < 0].min()) if np.any(rho < 0) else float('inf')

        # HYBRID: Modify vlo, vup
        if hybrid_flag == "LF":
            zlo = vlo / sigma_B
            zup = min(vup, hybrid_list['lf_cv']) / sigma_B
        elif hybrid_flag == "FLCI":
            gamma_full = np.zeros(len(y_T))
            gamma_full[rowsForARP] = v_B.ravel()
            gf = gamma_full.reshape(-1, 1)
            sigma_gamma = (sigma_full @ gf) / float(gf.T @ sigma_full @ gf)
            S = y_T - (sigma_gamma @ (gf.T @ y_T)).ravel()
            vFLCI = _FLCI_compute_vlo_vup(hybrid_list['vbar'], hybrid_list['dbar'],
                                          S, sigma_gamma.ravel())
            zlo = max(vlo, vFLCI['vlo']) / sigma_B
            zup = min(vup, vFLCI['vup']) / sigma_B
        else:  # ARP
            zlo = vlo / sigma_B
            zup = vup / sigma_B

        maxstat = linSoln['eta_star'] / sigma_B

        if not (zlo <= maxstat <= zup):
            return {'reject': 0, 'eta': linSoln['eta_star'],
                    'delta': linSoln['delta_star'], 'lambda': linSoln['lambda']}
        else:
            cval = max(0.0, _norminvp_generalized(1 - mod_size, zlo, zup))
            reject = int(maxstat > cval)
            return {'reject': reject, 'eta': linSoln['eta_star'],
                    'delta': linSoln['delta_star'], 'lambda': linSoln['lambda']}


def _ARP_computeCI(betahat, sigma, numPrePeriods, numPostPeriods,
                   A, d, l_vec, alpha, hybrid_flag, hybrid_list,
                   returnLength, grid_lb, grid_ub, gridPoints, rowsForARP=None):
    """Compute ARP confidence interval for Delta = {A*delta <= d} (nuisance parameter case)."""
    betahat = np.asarray(betahat, dtype=float).ravel()
    sigma_np = np.asarray(sigma, dtype=float)
    A_np = np.asarray(A, dtype=float)
    d_np = np.asarray(d, dtype=float).ravel()
    l_vec_np = np.asarray(l_vec, dtype=float).ravel()

    thetaGrid = np.linspace(grid_lb, grid_ub, gridPoints)

    # Construct Gamma and A*(Gamma)^{-1}
    Gamma = _construct_Gamma(l_vec_np)
    post_cols = np.arange(numPrePeriods, numPrePeriods + numPostPeriods)
    AGammaInv = A_np[:, post_cols] @ np.linalg.inv(Gamma)
    AGammaInv_one = AGammaInv[:, 0]
    AGammaInv_minusOne = AGammaInv[:, 1:]

    # Compute Y = A*betahat - d and its variance
    Y = A_np @ betahat - d_np
    sigmaY = A_np @ sigma_np @ A_np.T

    # HYBRID: least favorable CV
    if hybrid_flag == "LF":
        hybrid_list['lf_cv'] = _compute_least_favorable_cv(
            AGammaInv_minusOne, sigmaY, hybrid_list['hybrid_kappa'],
            rowsForARP=rowsForARP)

    def testTheta(theta):
        hl = dict(hybrid_list)
        if hybrid_flag == "FLCI":
            vbar_row = np.asarray(hl['vbar']).ravel()
            vbar_d = float(vbar_row @ d_np)
            vbar_AG1 = float(vbar_row @ AGammaInv_one)
            hl['dbar'] = np.array([
                hl['flci_halflength'] - vbar_d + (1 - vbar_AG1) * theta,
                hl['flci_halflength'] + vbar_d - (1 - vbar_AG1) * theta
            ])
        result = _lp_conditional_test_fn(
            theta=theta, y_T=Y - AGammaInv_one * theta,
            X_T=AGammaInv_minusOne, sigma=sigmaY, alpha=alpha,
            hybrid_flag=hybrid_flag, hybrid_list=hl, rowsForARP=rowsForARP)
        return 1 - result['reject']

    testResults = np.array([testTheta(th) for th in thetaGrid])

    if (testResults[0] == 1 or testResults[-1] == 1) and hybrid_flag != "FLCI":
        warnings.warn("CI is open at one of the endpoints; CI length may not be accurate")

    if returnLength:
        gridLength = 0.5 * (np.concatenate([[0], np.diff(thetaGrid)]) +
                            np.concatenate([np.diff(thetaGrid), [0]]))
        return float(np.sum(testResults * gridLength))
    else:
        return pd.DataFrame({'grid': thetaGrid, 'accept': testResults})


# --- ARP Tests Without Nuisance Parameters ---

def _test_in_identified_set(y, sigma, A, d, alpha,
                            Abar_additional=None, dbar_additional=None):
    """APR test of E[A*Y] - d <= 0 without nuisance parameters."""
    y = np.asarray(y, dtype=float).ravel()
    sigma_np = np.asarray(sigma, dtype=float)
    A_np = np.asarray(A, dtype=float)
    d_np = np.asarray(d, dtype=float).ravel()

    sigmaTilde = np.sqrt(np.diag(A_np @ sigma_np @ A_np.T))
    Atilde = np.diag(1.0 / sigmaTilde) @ A_np
    dtilde = (1.0 / sigmaTilde) * d_np

    normalizedMoments = Atilde @ y - dtilde
    maxLocation = int(np.argmax(normalizedMoments))
    maxMoment = normalizedMoments[maxLocation]

    nrows = Atilde.shape[0]
    T_B = np.zeros((1, nrows))
    T_B[0, maxLocation] = 1.0
    iota = np.ones((nrows, 1))

    gamma = (T_B @ Atilde).T  # column vector
    Abar = Atilde - iota @ (T_B @ Atilde)
    dbar = (np.eye(nrows) - iota @ T_B) @ dtilde

    if Abar_additional is not None:
        Abar = np.vstack([Abar, np.asarray(Abar_additional, dtype=float)])
        dbar = np.concatenate([dbar, np.asarray(dbar_additional, dtype=float).ravel()])

    sigmabar = float(np.sqrt(gamma.T @ sigma_np @ gamma))
    c_vec = (sigma_np @ gamma) / float(gamma.T @ sigma_np @ gamma)
    z = (np.eye(len(y)) - c_vec @ gamma.T) @ y
    VLoVUpVec = _vlo_vup_fn(
        torch.tensor(gamma), torch.tensor(sigma_np),
        torch.tensor(Abar), torch.tensor(dbar), torch.tensor(z))
    VLo, VUp = VLoVUpVec

    mu_val = float(T_B @ dtilde)
    criticalVal = max(0.0, _norminvp_generalized(1 - alpha, float(VLo), float(VUp),
                                                  mu=mu_val, sd=sigmabar))
    reject = (maxMoment + mu_val) > criticalVal
    return bool(reject)


def _test_in_identified_set_FLCI_hybrid(y, sigma, A, d, alpha, hybrid_list):
    """FLCI hybrid test: first test |l'y| > halflength, then conditional test."""
    y = np.asarray(y, dtype=float).ravel()
    flci_l = np.asarray(hybrid_list['flci_l'], dtype=float).ravel()
    A_fs = np.vstack([flci_l, -flci_l])
    d_fs = np.array([hybrid_list['flci_halflength'], hybrid_list['flci_halflength']])

    if float(np.max(A_fs @ y - d_fs)) > 0:
        return True
    else:
        alphatilde = (alpha - hybrid_list['hybrid_kappa']) / (1 - hybrid_list['hybrid_kappa'])
        return _test_in_identified_set(y, sigma, A, d, alpha=alphatilde,
                                       Abar_additional=A_fs, dbar_additional=d_fs)


def _test_in_identified_set_LF_hybrid(y, sigma, A, d, alpha, hybrid_list):
    """Least-favorable hybrid test."""
    y = np.asarray(y, dtype=float).ravel()
    sigma_np = np.asarray(sigma, dtype=float)
    A_np = np.asarray(A, dtype=float)
    d_np = np.asarray(d, dtype=float).ravel()

    sigmaTilde = np.sqrt(np.diag(A_np @ sigma_np @ A_np.T))
    Atilde = np.diag(1.0 / sigmaTilde) @ A_np
    dtilde = (1.0 / sigmaTilde) * d_np

    normalizedMoments = Atilde @ y - dtilde
    maxLocation = int(np.argmax(normalizedMoments))
    maxMoment = normalizedMoments[maxLocation]

    if maxMoment > hybrid_list['lf_cv']:
        return True

    nrows = Atilde.shape[0]
    T_B = np.zeros((1, nrows))
    T_B[0, maxLocation] = 1.0
    iota = np.ones((nrows, 1))

    gamma = (T_B @ Atilde).T
    Abar = Atilde - iota @ (T_B @ Atilde)
    dbar = (np.eye(nrows) - iota @ T_B) @ dtilde

    sigmabar = float(np.sqrt(gamma.T @ sigma_np @ gamma))
    c_vec = (sigma_np @ gamma) / float(gamma.T @ sigma_np @ gamma)
    z = (np.eye(len(y)) - c_vec @ gamma.T) @ y
    VLoVUpVec = _vlo_vup_fn(
        torch.tensor(gamma), torch.tensor(sigma_np),
        torch.tensor(Abar), torch.tensor(dbar), torch.tensor(z))
    VLo, VUp = VLoVUpVec

    alphatilde = (alpha - hybrid_list['hybrid_kappa']) / (1 - hybrid_list['hybrid_kappa'])
    mu_val = float(T_B @ dtilde)
    criticalVal = max(0.0, _norminvp_generalized(1 - alphatilde, float(VLo), float(VUp),
                                                  mu=mu_val, sd=sigmabar))
    reject = (maxMoment + mu_val) > criticalVal
    return bool(reject)


def _test_over_theta_grid(betahat, sigma, A, d, thetaGrid, numPrePeriods,
                          alpha, testFn=None, **kwargs):
    """Test whether values in a grid lie in the identified set."""
    if testFn is None:
        testFn = _test_in_identified_set
    betahat = np.asarray(betahat, dtype=float).ravel()
    sigma_np = np.asarray(sigma, dtype=float)
    A_np = np.asarray(A, dtype=float)
    d_np = np.asarray(d, dtype=float).ravel()

    results = []
    e1 = np.zeros(len(betahat))
    e1[numPrePeriods] = 1.0  # basis_vector(numPrePeriods+1, len(betahat)), 0-indexed
    for theta in thetaGrid:
        y = betahat - e1 * theta
        reject = testFn(y=y, sigma=sigma_np, A=A_np, d=d_np, alpha=alpha, **kwargs)
        results.append(0 if reject else 1)
    return np.column_stack([thetaGrid, np.array(results)])


def _APR_computeCI_NoNuis(betahat, sigma, A, d, numPrePeriods, numPostPeriods,
                          l_vec, alpha, returnLength, hybrid_flag, hybrid_list,
                          grid_ub, grid_lb, gridPoints, postPeriodMomentsOnly):
    """Compute APR confidence interval without nuisance parameters."""
    thetaGrid = np.linspace(grid_lb, grid_ub, gridPoints)

    if hybrid_flag == "ARP":
        resultsGrid = _test_over_theta_grid(
            betahat, sigma, A, d, thetaGrid, numPrePeriods, alpha)
    elif hybrid_flag == "FLCI":
        resultsGrid = _test_over_theta_grid(
            betahat, sigma, A, d, thetaGrid, numPrePeriods, alpha,
            testFn=_test_in_identified_set_FLCI_hybrid, hybrid_list=hybrid_list)
    elif hybrid_flag == "LF":
        resultsGrid = _test_over_theta_grid(
            betahat, sigma, A, d, thetaGrid, numPrePeriods, alpha,
            testFn=_test_in_identified_set_LF_hybrid, hybrid_list=hybrid_list)
    else:
        raise ValueError("hybrid_flag must be 'ARP', 'FLCI', or 'LF'")

    if resultsGrid[0, 1] == 1 or resultsGrid[-1, 1] == 1:
        warnings.warn("CI is open at one of the endpoints; CI length may not be accurate")

    if returnLength:
        gridLength = 0.5 * (np.concatenate([[0], np.diff(thetaGrid)]) +
                            np.concatenate([np.diff(thetaGrid), [0]]))
        return float(np.sum(resultsGrid[:, 1] * gridLength))
    else:
        return pd.DataFrame({'grid': resultsGrid[:, 0], 'accept': resultsGrid[:, 1]})


# =========================================================================
# PHASE 3: DELTA SD AND DELTA RM
# =========================================================================

# --- DeltaSD Functions (from deltasd.R) ---

def _create_A_SD(numPrePeriods, numPostPeriods, postPeriodMomentsOnly=False):
    """Create constraint matrix A for Delta^SD(M) (second-difference constraints)."""
    totalPeriods = numPrePeriods + numPostPeriods
    # Atilde: second-differences including t=0, then drop t=0 column
    Atilde = np.zeros((totalPeriods - 1, totalPeriods + 1))
    for r in range(totalPeriods - 1):
        Atilde[r, r] = 1.0
        Atilde[r, r + 1] = -2.0
        Atilde[r, r + 2] = 1.0
    # Drop column for t=0 (0-indexed: column numPrePeriods)
    Atilde = np.delete(Atilde, numPrePeriods, axis=1)

    if postPeriodMomentsOnly:
        postPeriodIndices = np.arange(numPrePeriods, Atilde.shape[1])
        prePeriodOnlyRows = np.where(np.sum(Atilde[:, postPeriodIndices] != 0, axis=1) == 0)[0]
        if len(prePeriodOnlyRows) > 0:
            Atilde = np.delete(Atilde, prePeriodOnlyRows, axis=0)

    A = np.vstack([Atilde, -Atilde])
    return A


def _create_d_SD(numPrePeriods, numPostPeriods, M, postPeriodMomentsOnly=False):
    """Create d vector for Delta^SD(M) constraints."""
    A_SD = _create_A_SD(numPrePeriods, numPostPeriods, postPeriodMomentsOnly)
    return np.full(A_SD.shape[0], M)


def _compute_IDset_DeltaSD(M, trueBeta, l_vec, numPrePeriods, numPostPeriods):
    """Compute identified set bounds for Delta^SD(M)."""
    trueBeta = np.asarray(trueBeta, dtype=float).ravel()
    l_vec = np.asarray(l_vec, dtype=float).ravel()
    totalPeriods = numPrePeriods + numPostPeriods

    fDelta = np.concatenate([np.zeros(numPrePeriods), l_vec])
    A_SD = _create_A_SD(numPrePeriods, numPostPeriods)
    d_SD = _create_d_SD(numPrePeriods, numPostPeriods, M)

    # Pre-period equality constraints
    A_eq = np.hstack([np.eye(numPrePeriods), np.zeros((numPrePeriods, numPostPeriods))])
    b_eq = trueBeta[:numPrePeriods]

    bounds = [(None, None)] * totalPeriods

    # Max
    res_max = sciopt.linprog(-fDelta, A_ub=A_SD, b_ub=d_SD, A_eq=A_eq, b_eq=b_eq,
                             bounds=bounds, method='highs')
    # Min
    res_min = sciopt.linprog(fDelta, A_ub=A_SD, b_ub=d_SD, A_eq=A_eq, b_eq=b_eq,
                             bounds=bounds, method='highs')

    l_beta_post = float(l_vec @ trueBeta[numPrePeriods:numPrePeriods + numPostPeriods])
    if not res_max.success and not res_min.success:
        warnings.warn("Solver did not find an optimum")
        return {'id_lb': l_beta_post, 'id_ub': l_beta_post}

    id_ub = l_beta_post - res_min.fun if res_min.success else l_beta_post
    id_lb = l_beta_post - (-res_max.fun) if res_max.success else l_beta_post
    return {'id_lb': id_lb, 'id_ub': id_ub}


def computeConditionalCS_DeltaSD(betahat, sigma, numPrePeriods, numPostPeriods,
                                  l_vec=None, M=0, alpha=0.05, hybrid_flag="FLCI",
                                  hybrid_kappa=None, returnLength=False,
                                  postPeriodMomentsOnly=True,
                                  gridPoints=1000, grid_ub=None, grid_lb=None, seed=0):
    """
    Compute ARP confidence set for Delta^SD(M).

    Args:
        betahat: estimated event study coefficients
        sigma: covariance matrix
        numPrePeriods: number of pre-periods
        numPostPeriods: number of post-periods
        l_vec: vector defining parameter of interest (default: first post-period)
        M: smoothness parameter (default: 0)
        alpha: significance level (default: 0.05)
        hybrid_flag: 'FLCI', 'LF', or 'ARP'
        hybrid_kappa: size of first-stage test (default: alpha/10)
        returnLength: if True, return CI length only
        postPeriodMomentsOnly: exclude pre-period-only moments
        gridPoints: number of grid points
        grid_ub, grid_lb: grid bounds (auto-computed if None)
        seed: random seed

    Returns:
        DataFrame with 'grid' and 'accept' columns, or CI length (float).
    """
    betahat = np.asarray(betahat, dtype=float).ravel()
    sigma_np = np.asarray(sigma, dtype=float)
    if l_vec is None:
        l_vec = np.zeros(numPostPeriods)
        l_vec[0] = 1.0
    else:
        l_vec = np.asarray(l_vec, dtype=float).ravel()
    if hybrid_kappa is None:
        hybrid_kappa = alpha / 10

    # Construct A_SD, d_SD (always with postPeriodMomentsOnly=False for full constraints)
    A_SD = _create_A_SD(numPrePeriods, numPostPeriods, postPeriodMomentsOnly=False)
    d_SD = _create_d_SD(numPrePeriods, numPostPeriods, M, postPeriodMomentsOnly=False)

    # Determine rowsForARP
    if postPeriodMomentsOnly and numPostPeriods > 1:
        postPeriodIndices = np.arange(numPrePeriods, A_SD.shape[1])
        postPeriodRows = np.where(np.sum(A_SD[:, postPeriodIndices] != 0, axis=1) > 0)[0]
        rowsForARP = postPeriodRows
    else:
        rowsForARP = np.arange(A_SD.shape[0])

    hybrid_list = {'hybrid_kappa': hybrid_kappa}

    if numPostPeriods == 1:
        # --- No nuisance parameter case ---
        if hybrid_flag == "FLCI":
            flci = _find_optimal_flci_helper(sigma_np, M, numPrePeriods, numPostPeriods, l_vec, alpha=hybrid_kappa)
            hybrid_list['flci_l'] = flci['optimalVec']
            hybrid_list['flci_halflength'] = flci['optimalHalfLength']
            if grid_ub is None:
                grid_ub = float(flci['optimalVec'] @ betahat) + flci['optimalHalfLength']
            if grid_lb is None:
                grid_lb = float(flci['optimalVec'] @ betahat) - flci['optimalHalfLength']
        elif hybrid_flag == "LF":
            lf_cv = _compute_least_favorable_cv(None, A_SD @ sigma_np @ A_SD.T, hybrid_kappa, seed=seed)
            hybrid_list['lf_cv'] = lf_cv
            if grid_ub is None or grid_lb is None:
                IDset = _compute_IDset_DeltaSD(M, np.zeros(numPrePeriods + numPostPeriods), l_vec, numPrePeriods, numPostPeriods)
                post_sigma = sigma_np[numPrePeriods:numPrePeriods + numPostPeriods, numPrePeriods:numPrePeriods + numPostPeriods]
                sdTheta = float(np.sqrt(l_vec @ post_sigma @ l_vec))
                if grid_ub is None:
                    grid_ub = IDset['id_ub'] + 20 * sdTheta
                if grid_lb is None:
                    grid_lb = IDset['id_lb'] - 20 * sdTheta
        else:  # ARP
            if grid_ub is None or grid_lb is None:
                IDset = _compute_IDset_DeltaSD(M, np.zeros(numPrePeriods + numPostPeriods), l_vec, numPrePeriods, numPostPeriods)
                post_sigma = sigma_np[numPrePeriods:numPrePeriods + numPostPeriods, numPrePeriods:numPrePeriods + numPostPeriods]
                sdTheta = float(np.sqrt(l_vec @ post_sigma @ l_vec))
                if grid_ub is None:
                    grid_ub = IDset['id_ub'] + 20 * sdTheta
                if grid_lb is None:
                    grid_lb = IDset['id_lb'] - 20 * sdTheta

        return _APR_computeCI_NoNuis(betahat, sigma_np, A_SD, d_SD,
                                     numPrePeriods, numPostPeriods, l_vec, alpha,
                                     returnLength, hybrid_flag, hybrid_list,
                                     grid_ub, grid_lb, gridPoints, postPeriodMomentsOnly)
    else:
        # --- Nuisance parameter case (numPostPeriods > 1) ---
        if hybrid_flag == "FLCI":
            flci = _find_optimal_flci_helper(sigma_np, M, numPrePeriods, numPostPeriods, l_vec, alpha=hybrid_kappa)
            hybrid_list['flci_l'] = flci['optimalVec']
            hybrid_list['flci_halflength'] = flci['optimalHalfLength']
            # Compute vbar via projection
            optVec = flci['optimalVec']
            nA = A_SD.shape[0]
            vbar = cp.Variable(nA)
            obj = cp.Minimize(cp.sum_squares(optVec - A_SD.T @ vbar))
            prob = cp.Problem(obj)
            prob.solve(solver=cp.ECOS)
            hybrid_list['vbar'] = vbar.value
            if grid_ub is None:
                grid_ub = float(optVec @ betahat) + flci['optimalHalfLength']
            if grid_lb is None:
                grid_lb = float(optVec @ betahat) - flci['optimalHalfLength']
        else:
            IDset = _compute_IDset_DeltaSD(M, np.zeros(numPrePeriods + numPostPeriods), l_vec, numPrePeriods, numPostPeriods)
            post_sigma = sigma_np[numPrePeriods:numPrePeriods + numPostPeriods, numPrePeriods:numPrePeriods + numPostPeriods]
            sdTheta = float(np.sqrt(l_vec @ post_sigma @ l_vec))
            if grid_ub is None:
                grid_ub = IDset['id_ub'] + 20 * sdTheta
            if grid_lb is None:
                grid_lb = IDset['id_lb'] - 20 * sdTheta

        return _ARP_computeCI(betahat, sigma_np, numPrePeriods, numPostPeriods,
                              A_SD, d_SD, l_vec, alpha, hybrid_flag, hybrid_list,
                              returnLength, grid_lb, grid_ub, gridPoints,
                              rowsForARP=rowsForARP)


# --- DeltaRM Functions (from deltarm.R) ---

def _create_A_RM(numPrePeriods, numPostPeriods, Mbar=1, s=0, max_positive=True, dropZero=True):
    """Create constraint matrix A for Delta^RM_{s,(.)}(Mbar)."""
    totalPeriods = numPrePeriods + numPostPeriods
    # First-difference matrix including t=0: totalPeriods x (totalPeriods+1)
    Atilde = np.zeros((totalPeriods, totalPeriods + 1))
    for r in range(totalPeriods):
        Atilde[r, r] = -1.0
        Atilde[r, r + 1] = 1.0

    # v_max_dif: extracts the first difference at period s
    # R uses 1-indexed: v_max_dif[(numPrePeriods+s):(numPrePeriods+1+s)]
    # In 0-indexed: positions (numPrePeriods+s-1) and (numPrePeriods+s)
    v_max_dif = np.zeros((1, totalPeriods + 1))
    idx = numPrePeriods + s - 1  # convert from R's 1-indexing
    v_max_dif[0, idx] = -1.0
    v_max_dif[0, idx + 1] = 1.0

    if not max_positive:
        v_max_dif = -v_max_dif

    # Bounds: 1*v_max_dif for pre-periods, Mbar*v_max_dif for post-periods
    A_UB = np.vstack([
        np.tile(v_max_dif, (numPrePeriods, 1)),
        np.tile(Mbar * v_max_dif, (numPostPeriods, 1))
    ])

    # |Atilde * delta| <= A_UB * delta
    A = np.vstack([Atilde - A_UB, -Atilde - A_UB])

    # Remove zero rows
    row_norms = np.sum(A ** 2, axis=1)
    A = A[row_norms > 1e-10, :]

    # Drop t=0 column
    if dropZero:
        A = np.delete(A, numPrePeriods, axis=1)

    return A


def _create_d_RM(numPrePeriods, numPostPeriods, Mbar=0, s=0, max_positive=True, dropZero=True):
    """Create d vector for Delta^RM constraints (all zeros)."""
    A_RM = _create_A_RM(numPrePeriods, numPostPeriods, Mbar=Mbar, s=s,
                        max_positive=max_positive, dropZero=dropZero)
    return np.zeros(A_RM.shape[0])


def _compute_IDset_DeltaRM_fixedS(s, Mbar, max_positive, trueBeta, l_vec,
                                   numPrePeriods, numPostPeriods):
    """Compute identified set for Delta^RM for fixed s and (+)/(-)."""
    trueBeta = np.asarray(trueBeta, dtype=float).ravel()
    l_vec = np.asarray(l_vec, dtype=float).ravel()
    totalPeriods = numPrePeriods + numPostPeriods

    fDelta = np.concatenate([np.zeros(numPrePeriods), l_vec])
    A_RM_s = _create_A_RM(numPrePeriods, numPostPeriods, Mbar=Mbar, s=s, max_positive=max_positive)
    d_RM = _create_d_RM(numPrePeriods, numPostPeriods, Mbar=Mbar, s=s, max_positive=max_positive)

    A_eq = np.hstack([np.eye(numPrePeriods), np.zeros((numPrePeriods, numPostPeriods))])
    b_eq = trueBeta[:numPrePeriods]
    bounds = [(None, None)] * totalPeriods

    res_max = sciopt.linprog(-fDelta, A_ub=A_RM_s, b_ub=d_RM, A_eq=A_eq, b_eq=b_eq,
                             bounds=bounds, method='highs')
    res_min = sciopt.linprog(fDelta, A_ub=A_RM_s, b_ub=d_RM, A_eq=A_eq, b_eq=b_eq,
                             bounds=bounds, method='highs')

    l_beta_post = float(l_vec @ trueBeta[numPrePeriods:numPrePeriods + numPostPeriods])
    if not res_max.success and not res_min.success:
        return {'id_lb': l_beta_post, 'id_ub': l_beta_post}
    id_ub = l_beta_post - res_min.fun if res_min.success else l_beta_post
    id_lb = l_beta_post - (-res_max.fun) if res_max.success else l_beta_post
    return {'id_lb': id_lb, 'id_ub': id_ub}


def _compute_IDset_DeltaRM(Mbar, trueBeta, l_vec, numPrePeriods, numPostPeriods):
    """Compute identified set for Delta^RM (union over s and +/-)."""
    min_s = -(numPrePeriods - 1)
    all_lbs, all_ubs = [], []
    for s in range(min_s, 1):
        for mp in [True, False]:
            res = _compute_IDset_DeltaRM_fixedS(s, Mbar, mp, trueBeta, l_vec,
                                                numPrePeriods, numPostPeriods)
            all_lbs.append(res['id_lb'])
            all_ubs.append(res['id_ub'])
    return {'id_lb': min(all_lbs), 'id_ub': max(all_ubs)}


def _computeConditionalCS_DeltaRM_fixedS(s, max_positive, Mbar, betahat, sigma,
                                          numPrePeriods, numPostPeriods, l_vec,
                                          alpha, hybrid_flag, hybrid_kappa,
                                          postPeriodMomentsOnly, gridPoints,
                                          grid_ub, grid_lb, seed=0):
    """Compute ARP CI for Delta^RM for a fixed s and (+)/(-)."""
    if hybrid_flag not in ("LF", "ARP"):
        raise ValueError("For DeltaRM, hybrid_flag must be 'LF' or 'ARP'")

    betahat = np.asarray(betahat, dtype=float).ravel()
    sigma_np = np.asarray(sigma, dtype=float)
    l_vec = np.asarray(l_vec, dtype=float).ravel()
    hybrid_list = {'hybrid_kappa': hybrid_kappa}

    A_RM_s = _create_A_RM(numPrePeriods, numPostPeriods, Mbar=Mbar, s=s, max_positive=max_positive)
    d_RM = _create_d_RM(numPrePeriods, numPostPeriods, Mbar=Mbar, s=s, max_positive=max_positive)

    if postPeriodMomentsOnly:
        if numPostPeriods > 1:
            postPeriodIndices = np.arange(numPrePeriods, A_RM_s.shape[1])
            postPeriodRows = np.where(np.sum(A_RM_s[:, postPeriodIndices] != 0, axis=1) > 0)[0]
            rowsForARP = postPeriodRows
        else:
            postPeriodRows = np.where(A_RM_s[:, -1] != 0)[0]
            A_RM_s = A_RM_s[postPeriodRows, :]
            d_RM = d_RM[postPeriodRows]
            rowsForARP = None
    else:
        rowsForARP = np.arange(A_RM_s.shape[0]) if numPostPeriods > 1 else None

    if numPostPeriods == 1:
        if hybrid_flag == "LF":
            lf_cv = _compute_least_favorable_cv(None, A_RM_s @ sigma_np @ A_RM_s.T,
                                                hybrid_kappa, seed=seed)
            hybrid_list['lf_cv'] = lf_cv
        return _APR_computeCI_NoNuis(betahat, sigma_np, A_RM_s, d_RM,
                                     numPrePeriods, numPostPeriods, l_vec, alpha,
                                     False, hybrid_flag, hybrid_list,
                                     grid_ub, grid_lb, gridPoints, postPeriodMomentsOnly)
    else:
        return _ARP_computeCI(betahat, sigma_np, numPrePeriods, numPostPeriods,
                              A_RM_s, d_RM, l_vec, alpha, hybrid_flag, hybrid_list,
                              False, grid_lb, grid_ub, gridPoints,
                              rowsForARP=rowsForARP)


def computeConditionalCS_DeltaRM(betahat, sigma, numPrePeriods, numPostPeriods,
                                  l_vec=None, Mbar=0, alpha=0.05, hybrid_flag="LF",
                                  hybrid_kappa=None, returnLength=False,
                                  postPeriodMomentsOnly=True,
                                  gridPoints=1000, grid_ub=None, grid_lb=None, seed=0):
    """
    Compute ARP confidence set for Delta^RM(Mbar).
    Takes union of CIs over all s in {-(numPrePeriods-1), ..., 0} and (+)/(-).
    """
    betahat = np.asarray(betahat, dtype=float).ravel()
    sigma_np = np.asarray(sigma, dtype=float)
    if l_vec is None:
        l_vec = np.zeros(numPostPeriods)
        l_vec[0] = 1.0
    else:
        l_vec = np.asarray(l_vec, dtype=float).ravel()
    if hybrid_kappa is None:
        hybrid_kappa = alpha / 10

    min_s = -(numPrePeriods - 1)
    s_indices = list(range(min_s, 1))

    # Auto-compute grid bounds
    post_sigma = sigma_np[numPrePeriods:numPrePeriods + numPostPeriods,
                          numPrePeriods:numPrePeriods + numPostPeriods]
    sdTheta = float(np.sqrt(l_vec @ post_sigma @ l_vec))
    if grid_ub is None:
        grid_ub = 20 * sdTheta
    if grid_lb is None:
        grid_lb = -20 * sdTheta

    # Loop over s and (+)/(-)
    CIs_plus = np.zeros((gridPoints, len(s_indices)))
    CIs_minus = np.zeros((gridPoints, len(s_indices)))
    for i, s in enumerate(s_indices):
        CI_plus = _computeConditionalCS_DeltaRM_fixedS(
            s, True, Mbar, betahat, sigma_np, numPrePeriods, numPostPeriods,
            l_vec, alpha, hybrid_flag, hybrid_kappa, postPeriodMomentsOnly,
            gridPoints, grid_ub, grid_lb, seed)
        CIs_plus[:, i] = CI_plus['accept'].values

        CI_minus = _computeConditionalCS_DeltaRM_fixedS(
            s, False, Mbar, betahat, sigma_np, numPrePeriods, numPostPeriods,
            l_vec, alpha, hybrid_flag, hybrid_kappa, postPeriodMomentsOnly,
            gridPoints, grid_ub, grid_lb, seed)
        CIs_minus[:, i] = CI_minus['accept'].values

    # Union: max over s for each (+), (-), then max between (+) and (-)
    accept = np.maximum(CIs_plus.max(axis=1), CIs_minus.max(axis=1))
    thetaGrid = np.linspace(grid_lb, grid_ub, gridPoints)
    CI_RM = pd.DataFrame({'grid': thetaGrid, 'accept': accept})

    if returnLength:
        gridLength = 0.5 * (np.concatenate([[0], np.diff(thetaGrid)]) +
                            np.concatenate([np.diff(thetaGrid), [0]]))
        return float(np.sum(accept * gridLength))
    else:
        return CI_RM


# =========================================================================
# PHASE 4: COMPOSITE DELTA VARIANTS
# =========================================================================

# --- Generic Helpers for SD-like and RM-like CI computation ---

def _generic_IDset(A_fn, d_fn, trueBeta, l_vec, numPrePeriods, numPostPeriods, **kwargs):
    """Generic identified set computation via LP."""
    trueBeta = np.asarray(trueBeta, dtype=float).ravel()
    l_vec = np.asarray(l_vec, dtype=float).ravel()
    totalPeriods = numPrePeriods + numPostPeriods
    fDelta = np.concatenate([np.zeros(numPrePeriods), l_vec])
    A = A_fn(**kwargs)
    d = d_fn(**kwargs)
    A_eq = np.hstack([np.eye(numPrePeriods), np.zeros((numPrePeriods, numPostPeriods))])
    b_eq = trueBeta[:numPrePeriods]
    bounds = [(None, None)] * totalPeriods
    res_max = sciopt.linprog(-fDelta, A_ub=A, b_ub=d, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    res_min = sciopt.linprog(fDelta, A_ub=A, b_ub=d, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    l_beta_post = float(l_vec @ trueBeta[numPrePeriods:numPrePeriods + numPostPeriods])
    id_ub = l_beta_post - res_min.fun if res_min.success else l_beta_post
    id_lb = l_beta_post - (-res_max.fun) if res_max.success else l_beta_post
    return {'id_lb': id_lb, 'id_ub': id_ub}


def _generic_cs_sd_type(betahat, sigma, numPrePeriods, numPostPeriods, l_vec,
                         A, d, alpha, hybrid_flag, hybrid_kappa, returnLength,
                         postPeriodMomentsOnly, gridPoints, grid_ub, grid_lb,
                         seed, IDset_fn, IDset_kwargs):
    """Generic CI computation for SD-like variants (single constraint matrix)."""
    betahat = np.asarray(betahat, dtype=float).ravel()
    sigma_np = np.asarray(sigma, dtype=float)
    l_vec = np.asarray(l_vec, dtype=float).ravel()

    if postPeriodMomentsOnly and numPostPeriods > 1:
        postPeriodIndices = np.arange(numPrePeriods, A.shape[1])
        rowsForARP = np.where(np.sum(A[:, postPeriodIndices] != 0, axis=1) > 0)[0]
    else:
        rowsForARP = np.arange(A.shape[0])

    hybrid_list = {'hybrid_kappa': hybrid_kappa}

    if numPostPeriods == 1:
        if hybrid_flag == "FLCI":
            flci = _find_optimal_flci_helper(sigma_np, IDset_kwargs.get('M', 0),
                                             numPrePeriods, numPostPeriods, l_vec, alpha=hybrid_kappa)
            hybrid_list['flci_l'] = flci['optimalVec']
            hybrid_list['flci_halflength'] = flci['optimalHalfLength']
            if grid_ub is None:
                grid_ub = float(flci['optimalVec'] @ betahat) + flci['optimalHalfLength']
            if grid_lb is None:
                grid_lb = float(flci['optimalVec'] @ betahat) - flci['optimalHalfLength']
        elif hybrid_flag == "LF":
            lf_cv = _compute_least_favorable_cv(None, A @ sigma_np @ A.T, hybrid_kappa, seed=seed)
            hybrid_list['lf_cv'] = lf_cv
        if grid_ub is None or grid_lb is None:
            IDset = IDset_fn(**IDset_kwargs)
            post_sigma = sigma_np[numPrePeriods:numPrePeriods + numPostPeriods,
                                  numPrePeriods:numPrePeriods + numPostPeriods]
            sdTheta = float(np.sqrt(l_vec @ post_sigma @ l_vec))
            if grid_ub is None:
                grid_ub = IDset['id_ub'] + 20 * sdTheta
            if grid_lb is None:
                grid_lb = IDset['id_lb'] - 20 * sdTheta
        return _APR_computeCI_NoNuis(betahat, sigma_np, A, d, numPrePeriods, numPostPeriods,
                                     l_vec, alpha, returnLength, hybrid_flag, hybrid_list,
                                     grid_ub, grid_lb, gridPoints, postPeriodMomentsOnly)
    else:
        if hybrid_flag == "FLCI":
            flci = _find_optimal_flci_helper(sigma_np, IDset_kwargs.get('M', 0),
                                             numPrePeriods, numPostPeriods, l_vec, alpha=hybrid_kappa)
            hybrid_list['flci_l'] = flci['optimalVec']
            hybrid_list['flci_halflength'] = flci['optimalHalfLength']
            nA = A.shape[0]
            vbar = cp.Variable(nA)
            obj = cp.Minimize(cp.sum_squares(flci['optimalVec'] - A.T @ vbar))
            cp.Problem(obj).solve(solver=cp.ECOS)
            hybrid_list['vbar'] = vbar.value
            if grid_ub is None:
                grid_ub = float(flci['optimalVec'] @ betahat) + flci['optimalHalfLength']
            if grid_lb is None:
                grid_lb = float(flci['optimalVec'] @ betahat) - flci['optimalHalfLength']
        else:
            if grid_ub is None or grid_lb is None:
                IDset = IDset_fn(**IDset_kwargs)
                post_sigma = sigma_np[numPrePeriods:numPrePeriods + numPostPeriods,
                                      numPrePeriods:numPrePeriods + numPostPeriods]
                sdTheta = float(np.sqrt(l_vec @ post_sigma @ l_vec))
                if grid_ub is None:
                    grid_ub = IDset['id_ub'] + 20 * sdTheta
                if grid_lb is None:
                    grid_lb = IDset['id_lb'] - 20 * sdTheta
        return _ARP_computeCI(betahat, sigma_np, numPrePeriods, numPostPeriods,
                              A, d, l_vec, alpha, hybrid_flag, hybrid_list,
                              returnLength, grid_lb, grid_ub, gridPoints, rowsForARP=rowsForARP)


def _generic_cs_rm_type(betahat, sigma, numPrePeriods, numPostPeriods, l_vec,
                         create_A_fn, create_d_fn, A_kwargs_fn, d_kwargs=None,
                         min_s=0, alpha=0.05, hybrid_flag="LF", hybrid_kappa=None,
                         returnLength=False, postPeriodMomentsOnly=True,
                         gridPoints=1000, grid_ub=None, grid_lb=None, seed=0,
                         d_kwargs_fn=None):
    """Generic CI computation for RM-like variants (loop over s and +/-)."""
    betahat = np.asarray(betahat, dtype=float).ravel()
    sigma_np = np.asarray(sigma, dtype=float)
    l_vec = np.asarray(l_vec, dtype=float).ravel()

    if hybrid_flag not in ("LF", "ARP"):
        raise ValueError("For RM-type variants, hybrid_flag must be 'LF' or 'ARP'")

    post_sigma = sigma_np[numPrePeriods:numPrePeriods + numPostPeriods,
                          numPrePeriods:numPrePeriods + numPostPeriods]
    sdTheta = float(np.sqrt(l_vec @ post_sigma @ l_vec))
    if grid_ub is None:
        grid_ub = 20 * sdTheta
    if grid_lb is None:
        grid_lb = -20 * sdTheta

    s_indices = list(range(min_s, 1))
    CIs_plus = np.zeros((gridPoints, len(s_indices)))
    CIs_minus = np.zeros((gridPoints, len(s_indices)))

    for i, s in enumerate(s_indices):
        for mp_flag, storage in [(True, CIs_plus), (False, CIs_minus)]:
            ak = A_kwargs_fn(s, mp_flag)
            A_s = create_A_fn(**ak)
            if d_kwargs_fn is not None:
                d_s = create_d_fn(**d_kwargs_fn(s, mp_flag))
            else:
                d_s = create_d_fn(**d_kwargs)
            hybrid_list = {'hybrid_kappa': hybrid_kappa}

            if postPeriodMomentsOnly:
                if numPostPeriods > 1:
                    postPeriodIndices = np.arange(numPrePeriods, A_s.shape[1])
                    rowsForARP = np.where(np.sum(A_s[:, postPeriodIndices] != 0, axis=1) > 0)[0]
                else:
                    postPeriodRows = np.where(A_s[:, -1] != 0)[0]
                    A_s = A_s[postPeriodRows, :]
                    d_s = d_s[postPeriodRows]
                    rowsForARP = None
            else:
                rowsForARP = np.arange(A_s.shape[0]) if numPostPeriods > 1 else None

            if numPostPeriods == 1:
                if hybrid_flag == "LF":
                    lf_cv = _compute_least_favorable_cv(None, A_s @ sigma_np @ A_s.T, hybrid_kappa, seed=seed)
                    hybrid_list['lf_cv'] = lf_cv
                CI = _APR_computeCI_NoNuis(betahat, sigma_np, A_s, d_s, numPrePeriods, numPostPeriods,
                                           l_vec, alpha, False, hybrid_flag, hybrid_list,
                                           grid_ub, grid_lb, gridPoints, postPeriodMomentsOnly)
            else:
                CI = _ARP_computeCI(betahat, sigma_np, numPrePeriods, numPostPeriods,
                                    A_s, d_s, l_vec, alpha, hybrid_flag, hybrid_list,
                                    False, grid_lb, grid_ub, gridPoints, rowsForARP=rowsForARP)
            storage[:, i] = CI['accept'].values

    accept = np.maximum(CIs_plus.max(axis=1), CIs_minus.max(axis=1))
    thetaGrid = np.linspace(grid_lb, grid_ub, gridPoints)

    if returnLength:
        gridLength = 0.5 * (np.concatenate([[0], np.diff(thetaGrid)]) +
                            np.concatenate([np.diff(thetaGrid), [0]]))
        return float(np.sum(accept * gridLength))
    else:
        return pd.DataFrame({'grid': thetaGrid, 'accept': accept})


# --- DeltaSDB: SD + Bias Direction ---

def _create_A_SDB(numPrePeriods, numPostPeriods, biasDirection="positive", postPeriodMomentsOnly=False):
    A_SD = _create_A_SD(numPrePeriods, numPostPeriods, postPeriodMomentsOnly)
    A_B = _create_A_B(numPrePeriods, numPostPeriods, biasDirection)
    return np.vstack([A_SD, A_B])

def _create_d_SDB(numPrePeriods, numPostPeriods, M, postPeriodMomentsOnly=False):
    d_SD = _create_d_SD(numPrePeriods, numPostPeriods, M, postPeriodMomentsOnly)
    d_B = np.zeros(numPostPeriods)
    return np.concatenate([d_SD, d_B])

def computeConditionalCS_DeltaSDB(betahat, sigma, numPrePeriods, numPostPeriods,
                                   l_vec=None, M=0, alpha=0.05, hybrid_flag="FLCI",
                                   hybrid_kappa=None, returnLength=False,
                                   biasDirection="positive", postPeriodMomentsOnly=True,
                                   gridPoints=1000, grid_ub=None, grid_lb=None, seed=0):
    """Compute confidence set for Delta^SDB(M) = SD + bias direction."""
    if l_vec is None:
        l_vec = np.zeros(numPostPeriods); l_vec[0] = 1.0
    if hybrid_kappa is None:
        hybrid_kappa = alpha / 10
    A = _create_A_SDB(numPrePeriods, numPostPeriods, biasDirection, postPeriodMomentsOnly=False)
    d = _create_d_SDB(numPrePeriods, numPostPeriods, M, postPeriodMomentsOnly=False)
    IDset_fn = lambda **kw: _generic_IDset(
        lambda **_: _create_A_SDB(numPrePeriods, numPostPeriods, biasDirection),
        lambda **_: _create_d_SDB(numPrePeriods, numPostPeriods, M),
        np.zeros(numPrePeriods + numPostPeriods), l_vec, numPrePeriods, numPostPeriods)
    return _generic_cs_sd_type(betahat, sigma, numPrePeriods, numPostPeriods, l_vec,
                                A, d, alpha, hybrid_flag, hybrid_kappa, returnLength,
                                postPeriodMomentsOnly, gridPoints, grid_ub, grid_lb, seed,
                                IDset_fn, {'M': M})


# --- DeltaSDM: SD + Monotonicity ---

def _create_A_SDM(numPrePeriods, numPostPeriods, monotonicityDirection="increasing", postPeriodMomentsOnly=False):
    A_SD = _create_A_SD(numPrePeriods, numPostPeriods, postPeriodMomentsOnly)
    A_M = _create_A_M(numPrePeriods, numPostPeriods, monotonicityDirection, postPeriodMomentsOnly)
    return np.vstack([A_SD, A_M])

def _create_d_SDM(numPrePeriods, numPostPeriods, M, postPeriodMomentsOnly=False):
    d_SD = _create_d_SD(numPrePeriods, numPostPeriods, M, postPeriodMomentsOnly)
    nM = numPostPeriods if postPeriodMomentsOnly else (numPrePeriods + numPostPeriods)
    d_M = np.zeros(nM)
    return np.concatenate([d_SD, d_M])

def computeConditionalCS_DeltaSDM(betahat, sigma, numPrePeriods, numPostPeriods,
                                   l_vec=None, M=0, alpha=0.05, monotonicityDirection="increasing",
                                   hybrid_flag="FLCI", hybrid_kappa=None, returnLength=False,
                                   postPeriodMomentsOnly=True,
                                   gridPoints=1000, grid_ub=None, grid_lb=None, seed=0):
    """Compute confidence set for Delta^SDM(M) = SD + monotonicity."""
    if l_vec is None:
        l_vec = np.zeros(numPostPeriods); l_vec[0] = 1.0
    if hybrid_kappa is None:
        hybrid_kappa = alpha / 10
    A = _create_A_SDM(numPrePeriods, numPostPeriods, monotonicityDirection, postPeriodMomentsOnly=False)
    d = _create_d_SDM(numPrePeriods, numPostPeriods, M, postPeriodMomentsOnly=False)
    IDset_fn = lambda **kw: _generic_IDset(
        lambda **_: _create_A_SDM(numPrePeriods, numPostPeriods, monotonicityDirection),
        lambda **_: _create_d_SDM(numPrePeriods, numPostPeriods, M),
        np.zeros(numPrePeriods + numPostPeriods), l_vec, numPrePeriods, numPostPeriods)
    return _generic_cs_sd_type(betahat, sigma, numPrePeriods, numPostPeriods, l_vec,
                                A, d, alpha, hybrid_flag, hybrid_kappa, returnLength,
                                postPeriodMomentsOnly, gridPoints, grid_ub, grid_lb, seed,
                                IDset_fn, {'M': M})


# --- DeltaRMB: RM + Bias Direction ---

def _create_A_RMB(numPrePeriods, numPostPeriods, Mbar=1, s=0, max_positive=True, dropZero=True, biasDirection="positive"):
    A_RM = _create_A_RM(numPrePeriods, numPostPeriods, Mbar, s, max_positive, dropZero)
    A_B = _create_A_B(numPrePeriods, numPostPeriods, biasDirection)
    return np.vstack([A_RM, A_B])

def _create_d_RMB(numPrePeriods, numPostPeriods, Mbar=0, s=0, max_positive=True, dropZero=True):
    d_RM = _create_d_RM(numPrePeriods, numPostPeriods, Mbar=Mbar, s=s, max_positive=max_positive, dropZero=dropZero)
    d_B = np.zeros(numPostPeriods)
    return np.concatenate([d_RM, d_B])

def computeConditionalCS_DeltaRMB(betahat, sigma, numPrePeriods, numPostPeriods,
                                   l_vec=None, Mbar=0, alpha=0.05, hybrid_flag="LF",
                                   hybrid_kappa=None, returnLength=False,
                                   biasDirection="positive", postPeriodMomentsOnly=True,
                                   gridPoints=1000, grid_ub=None, grid_lb=None, seed=0):
    """Compute confidence set for Delta^RMB(Mbar) = RM + bias direction."""
    if l_vec is None:
        l_vec = np.zeros(numPostPeriods); l_vec[0] = 1.0
    if hybrid_kappa is None:
        hybrid_kappa = alpha / 10
    min_s = -(numPrePeriods - 1)
    return _generic_cs_rm_type(
        betahat, sigma, numPrePeriods, numPostPeriods, l_vec,
        create_A_fn=_create_A_RMB,
        create_d_fn=_create_d_RMB,
        A_kwargs_fn=lambda s, mp: dict(numPrePeriods=numPrePeriods, numPostPeriods=numPostPeriods,
                                        Mbar=Mbar, s=s, max_positive=mp, biasDirection=biasDirection),
        d_kwargs_fn=lambda s, mp: dict(numPrePeriods=numPrePeriods, numPostPeriods=numPostPeriods,
                                        Mbar=Mbar, s=s, max_positive=mp),
        min_s=min_s, alpha=alpha, hybrid_flag=hybrid_flag, hybrid_kappa=hybrid_kappa,
        returnLength=returnLength, postPeriodMomentsOnly=postPeriodMomentsOnly,
        gridPoints=gridPoints, grid_ub=grid_ub, grid_lb=grid_lb, seed=seed)


# --- DeltaRMM: RM + Monotonicity ---

def _create_A_RMM(numPrePeriods, numPostPeriods, Mbar=1, s=0, max_positive=True, dropZero=True, monotonicityDirection="increasing"):
    A_RM = _create_A_RM(numPrePeriods, numPostPeriods, Mbar, s, max_positive, dropZero)
    A_M = _create_A_M(numPrePeriods, numPostPeriods, monotonicityDirection)
    return np.vstack([A_RM, A_M])

def _create_d_RMM(numPrePeriods, numPostPeriods, Mbar=0, s=0, max_positive=True, dropZero=True):
    d_RM = _create_d_RM(numPrePeriods, numPostPeriods, Mbar=Mbar, s=s, max_positive=max_positive, dropZero=dropZero)
    d_M = np.zeros(numPrePeriods + numPostPeriods)
    return np.concatenate([d_RM, d_M])

def computeConditionalCS_DeltaRMM(betahat, sigma, numPrePeriods, numPostPeriods,
                                   l_vec=None, Mbar=0, alpha=0.05,
                                   monotonicityDirection="increasing",
                                   hybrid_flag="LF", hybrid_kappa=None, returnLength=False,
                                   postPeriodMomentsOnly=True,
                                   gridPoints=1000, grid_ub=None, grid_lb=None, seed=0):
    """Compute confidence set for Delta^RMM(Mbar) = RM + monotonicity."""
    if l_vec is None:
        l_vec = np.zeros(numPostPeriods); l_vec[0] = 1.0
    if hybrid_kappa is None:
        hybrid_kappa = alpha / 10
    min_s = -(numPrePeriods - 1)
    return _generic_cs_rm_type(
        betahat, sigma, numPrePeriods, numPostPeriods, l_vec,
        create_A_fn=_create_A_RMM,
        create_d_fn=_create_d_RMM,
        A_kwargs_fn=lambda s, mp: dict(numPrePeriods=numPrePeriods, numPostPeriods=numPostPeriods,
                                        Mbar=Mbar, s=s, max_positive=mp, monotonicityDirection=monotonicityDirection),
        d_kwargs_fn=lambda s, mp: dict(numPrePeriods=numPrePeriods, numPostPeriods=numPostPeriods,
                                        Mbar=Mbar, s=s, max_positive=mp),
        min_s=min_s, alpha=alpha, hybrid_flag=hybrid_flag, hybrid_kappa=hybrid_kappa,
        returnLength=returnLength, postPeriodMomentsOnly=postPeriodMomentsOnly,
        gridPoints=gridPoints, grid_ub=grid_ub, grid_lb=grid_lb, seed=seed)


# --- DeltaSDRM: SD + RM hybrid (second-differences bounded by max pre-period second-difference) ---

def _create_A_SDRM(numPrePeriods, numPostPeriods, Mbar=1, s=0, max_positive=True, dropZero=True):
    """Create constraint matrix for Delta^SDRM (second-diff bounded by max pre-period second-diff)."""
    totalPeriods = numPrePeriods + numPostPeriods
    # Second-difference matrix including t=0
    Atilde = np.zeros((totalPeriods - 1, totalPeriods + 1))
    for r in range(totalPeriods - 1):
        Atilde[r, r] = 1.0
        Atilde[r, r + 1] = -2.0
        Atilde[r, r + 2] = 1.0

    # v_max_dif for second-difference at reference period s
    # R uses 1-indexed: v_max_dif[(numPrePeriods+1+s-2):(numPrePeriods+1+s)]
    # In 0-indexed: start at (numPrePeriods+s-2)
    v_max_dif = np.zeros((1, totalPeriods + 1))
    idx = numPrePeriods + s - 2  # convert from R's 1-indexing
    v_max_dif[0, idx] = 1.0
    v_max_dif[0, idx + 1] = -2.0
    v_max_dif[0, idx + 2] = 1.0

    if not max_positive:
        v_max_dif = -v_max_dif

    A_UB = np.vstack([
        np.tile(v_max_dif, (numPrePeriods - 1, 1)),  # -1 because second-diff has one fewer row for pre
        np.tile(Mbar * v_max_dif, (numPostPeriods, 1))
    ])
    # Pad A_UB if needed to match Atilde rows
    if A_UB.shape[0] < Atilde.shape[0]:
        # The last pre-period second-diff row
        A_UB = np.vstack([A_UB, np.tile(Mbar * v_max_dif, (Atilde.shape[0] - A_UB.shape[0], 1))])

    A = np.vstack([Atilde - A_UB, -Atilde - A_UB])
    row_norms = np.sum(A ** 2, axis=1)
    A = A[row_norms > 1e-10, :]

    if dropZero:
        A = np.delete(A, numPrePeriods, axis=1)
    return A

def _create_d_SDRM(numPrePeriods, numPostPeriods, Mbar=0, s=0, max_positive=True, dropZero=True):
    A_SDRM = _create_A_SDRM(numPrePeriods, numPostPeriods, Mbar=Mbar, s=s, max_positive=max_positive, dropZero=dropZero)
    return np.zeros(A_SDRM.shape[0])

def computeConditionalCS_DeltaSDRM(betahat, sigma, numPrePeriods, numPostPeriods,
                                    l_vec=None, Mbar=0, alpha=0.05,
                                    hybrid_flag="LF", hybrid_kappa=None, returnLength=False,
                                    postPeriodMomentsOnly=True,
                                    gridPoints=1000, grid_ub=None, grid_lb=None, seed=0):
    """Compute confidence set for Delta^SDRM(Mbar) = SD + RM."""
    if l_vec is None:
        l_vec = np.zeros(numPostPeriods); l_vec[0] = 1.0
    if hybrid_kappa is None:
        hybrid_kappa = alpha / 10
    min_s = -(numPrePeriods - 2)
    return _generic_cs_rm_type(
        betahat, sigma, numPrePeriods, numPostPeriods, l_vec,
        create_A_fn=_create_A_SDRM,
        create_d_fn=_create_d_SDRM,
        A_kwargs_fn=lambda s, mp: dict(numPrePeriods=numPrePeriods, numPostPeriods=numPostPeriods,
                                        Mbar=Mbar, s=s, max_positive=mp),
        d_kwargs_fn=lambda s, mp: dict(numPrePeriods=numPrePeriods, numPostPeriods=numPostPeriods,
                                        Mbar=Mbar, s=s, max_positive=mp),
        min_s=min_s, alpha=alpha, hybrid_flag=hybrid_flag, hybrid_kappa=hybrid_kappa,
        returnLength=returnLength, postPeriodMomentsOnly=postPeriodMomentsOnly,
        gridPoints=gridPoints, grid_ub=grid_ub, grid_lb=grid_lb, seed=seed)


# --- DeltaSDRMB: SDRM + Bias Direction ---

def _create_A_SDRMB(numPrePeriods, numPostPeriods, Mbar=1, s=0, max_positive=True, dropZero=True, biasDirection="positive"):
    A_SDRM = _create_A_SDRM(numPrePeriods, numPostPeriods, Mbar, s, max_positive, dropZero)
    A_B = _create_A_B(numPrePeriods, numPostPeriods, biasDirection)
    return np.vstack([A_SDRM, A_B])

def _create_d_SDRMB(numPrePeriods, numPostPeriods, Mbar=0, s=0, max_positive=True, dropZero=True):
    d_SDRM = _create_d_SDRM(numPrePeriods, numPostPeriods, Mbar=Mbar, s=s, max_positive=max_positive, dropZero=dropZero)
    d_B = np.zeros(numPostPeriods)
    return np.concatenate([d_SDRM, d_B])

def computeConditionalCS_DeltaSDRMB(betahat, sigma, numPrePeriods, numPostPeriods,
                                     l_vec=None, Mbar=0, alpha=0.05,
                                     hybrid_flag="LF", hybrid_kappa=None, returnLength=False,
                                     biasDirection="positive", postPeriodMomentsOnly=True,
                                     gridPoints=1000, grid_ub=None, grid_lb=None, seed=0):
    """Compute confidence set for Delta^SDRMB(Mbar) = SDRM + bias direction."""
    if l_vec is None:
        l_vec = np.zeros(numPostPeriods); l_vec[0] = 1.0
    if hybrid_kappa is None:
        hybrid_kappa = alpha / 10
    min_s = -(numPrePeriods - 2)
    return _generic_cs_rm_type(
        betahat, sigma, numPrePeriods, numPostPeriods, l_vec,
        create_A_fn=_create_A_SDRMB,
        create_d_fn=_create_d_SDRMB,
        A_kwargs_fn=lambda s, mp: dict(numPrePeriods=numPrePeriods, numPostPeriods=numPostPeriods,
                                        Mbar=Mbar, s=s, max_positive=mp, biasDirection=biasDirection),
        d_kwargs_fn=lambda s, mp: dict(numPrePeriods=numPrePeriods, numPostPeriods=numPostPeriods,
                                        Mbar=Mbar, s=s, max_positive=mp),
        min_s=min_s, alpha=alpha, hybrid_flag=hybrid_flag, hybrid_kappa=hybrid_kappa,
        returnLength=returnLength, postPeriodMomentsOnly=postPeriodMomentsOnly,
        gridPoints=gridPoints, grid_ub=grid_ub, grid_lb=grid_lb, seed=seed)


# --- DeltaSDRMM: SDRM + Monotonicity ---

def _create_A_SDRMM(numPrePeriods, numPostPeriods, Mbar=1, s=0, max_positive=True, dropZero=True, monotonicityDirection="increasing"):
    A_SDRM = _create_A_SDRM(numPrePeriods, numPostPeriods, Mbar, s, max_positive, dropZero)
    A_M = _create_A_M(numPrePeriods, numPostPeriods, monotonicityDirection)
    return np.vstack([A_SDRM, A_M])

def _create_d_SDRMM(numPrePeriods, numPostPeriods, Mbar=0, s=0, max_positive=True, dropZero=True):
    d_SDRM = _create_d_SDRM(numPrePeriods, numPostPeriods, Mbar=Mbar, s=s, max_positive=max_positive, dropZero=dropZero)
    d_M = np.zeros(numPrePeriods + numPostPeriods)
    return np.concatenate([d_SDRM, d_M])

def computeConditionalCS_DeltaSDRMM(betahat, sigma, numPrePeriods, numPostPeriods,
                                     l_vec=None, Mbar=0, alpha=0.05,
                                     monotonicityDirection="increasing",
                                     hybrid_flag="LF", hybrid_kappa=None, returnLength=False,
                                     postPeriodMomentsOnly=True,
                                     gridPoints=1000, grid_ub=None, grid_lb=None, seed=0):
    """Compute confidence set for Delta^SDRMM(Mbar) = SDRM + monotonicity."""
    if l_vec is None:
        l_vec = np.zeros(numPostPeriods); l_vec[0] = 1.0
    if hybrid_kappa is None:
        hybrid_kappa = alpha / 10
    min_s = -(numPrePeriods - 2)
    return _generic_cs_rm_type(
        betahat, sigma, numPrePeriods, numPostPeriods, l_vec,
        create_A_fn=_create_A_SDRMM,
        create_d_fn=_create_d_SDRMM,
        A_kwargs_fn=lambda s, mp: dict(numPrePeriods=numPrePeriods, numPostPeriods=numPostPeriods,
                                        Mbar=Mbar, s=s, max_positive=mp, monotonicityDirection=monotonicityDirection),
        d_kwargs_fn=lambda s, mp: dict(numPrePeriods=numPrePeriods, numPostPeriods=numPostPeriods,
                                        Mbar=Mbar, s=s, max_positive=mp),
        min_s=min_s, alpha=alpha, hybrid_flag=hybrid_flag, hybrid_kappa=hybrid_kappa,
        returnLength=returnLength, postPeriodMomentsOnly=postPeriodMomentsOnly,
        gridPoints=gridPoints, grid_ub=grid_ub, grid_lb=grid_lb, seed=seed)


# =========================================================================
# PHASE 5: SENSITIVITY RESULTS AND UBLB
# =========================================================================

def _extract_lb_ub(ci_result):
    """Extract lb, ub from a CI grid result (DataFrame)."""
    if isinstance(ci_result, pd.DataFrame):
        accepted = ci_result[ci_result['accept'] == 1]['grid']
        if len(accepted) == 0:
            return float('nan'), float('nan')
        return float(accepted.min()), float(accepted.max())
    return float('nan'), float('nan')


def constructOriginalCS(betahat, sigma, numPrePeriods, numPostPeriods,
                        l_vec=None, alpha=0.05):
    """Construct original confidence set under parallel trends (normal CI)."""
    betahat = np.asarray(betahat, dtype=float).ravel()
    sigma_np = np.asarray(sigma, dtype=float)
    if l_vec is None:
        l_vec = np.zeros(numPostPeriods)
        l_vec[0] = 1.0
    else:
        l_vec = np.asarray(l_vec, dtype=float).ravel()

    post_sigma = sigma_np[numPrePeriods:numPrePeriods + numPostPeriods,
                          numPrePeriods:numPrePeriods + numPostPeriods]
    stdError = float(np.sqrt(l_vec @ post_sigma @ l_vec))
    pe = float(l_vec @ betahat[numPrePeriods:numPrePeriods + numPostPeriods])
    z = scistats.norm.ppf(1 - alpha / 2)
    return pd.DataFrame({
        'lb': [pe - z * stdError],
        'ub': [pe + z * stdError],
        'method': ['Original'],
        'Delta': [None]
    })


def createSensitivityResults(betahat, sigma, numPrePeriods, numPostPeriods,
                              method=None, Mvec=None, l_vec=None,
                              monotonicityDirection=None, biasDirection=None,
                              alpha=0.05, seed=0):
    """
    Sensitivity analysis for smoothness restrictions (Delta^SD and variants).

    Args:
        method: 'FLCI', 'Conditional', 'C-F', or 'C-LF'
        Mvec: vector of M values (default: auto)
        monotonicityDirection: 'increasing'/'decreasing' (for SDM)
        biasDirection: 'positive'/'negative' (for SDB)

    Returns:
        DataFrame with columns: lb, ub, method, Delta, M
    """
    betahat = np.asarray(betahat, dtype=float).ravel()
    sigma_np = np.asarray(sigma, dtype=float)
    if l_vec is None:
        l_vec = np.zeros(numPostPeriods)
        l_vec[0] = 1.0
    else:
        l_vec = np.asarray(l_vec, dtype=float).ravel()

    if Mvec is None:
        if numPrePeriods == 1:
            Mvec = np.linspace(0, float(np.sqrt(sigma_np[0, 0])), 10)
        else:
            Mub = DeltaSD_upperBound_Mpre(betahat, sigma_np, numPrePeriods, alpha=0.05)
            Mvec = np.linspace(0, Mub, 10)

    if monotonicityDirection is None and biasDirection is None:
        delta_type = "DeltaSD"
        if method is None:
            method = "FLCI"
    elif biasDirection is not None:
        delta_type = "DeltaSDPB" if biasDirection == "positive" else "DeltaSDNB"
        if method is None:
            method = "C-F"
    else:
        delta_type = "DeltaSDI" if monotonicityDirection == "increasing" else "DeltaSDD"
        if method is None:
            method = "C-F"

    method_to_flag = {"FLCI": None, "Conditional": "ARP", "C-F": "FLCI", "C-LF": "LF"}
    if method not in method_to_flag:
        raise ValueError("Method must be: FLCI, Conditional, C-F, or C-LF")

    results = []
    for M_val in Mvec:
        if method == "FLCI":
            temp = find_optimal_flci(betahat, sigma_np, M=M_val, numPrePeriods=numPrePeriods,
                                     numPostPeriods=numPostPeriods, l_vec=l_vec, alpha=alpha, seed=seed)
            results.append({'lb': temp['FLCI'][0], 'ub': temp['FLCI'][1],
                           'method': 'FLCI', 'Delta': delta_type, 'M': M_val})
        else:
            hf = method_to_flag[method]
            if monotonicityDirection is not None:
                temp = computeConditionalCS_DeltaSDM(
                    betahat, sigma_np, numPrePeriods, numPostPeriods,
                    l_vec=l_vec, M=M_val, alpha=alpha,
                    monotonicityDirection=monotonicityDirection,
                    hybrid_flag=hf, seed=seed)
            elif biasDirection is not None:
                temp = computeConditionalCS_DeltaSDB(
                    betahat, sigma_np, numPrePeriods, numPostPeriods,
                    l_vec=l_vec, M=M_val, alpha=alpha,
                    biasDirection=biasDirection,
                    hybrid_flag=hf, seed=seed)
            else:
                temp = computeConditionalCS_DeltaSD(
                    betahat, sigma_np, numPrePeriods, numPostPeriods,
                    l_vec=l_vec, M=M_val, alpha=alpha,
                    hybrid_flag=hf, seed=seed)
            lb, ub = _extract_lb_ub(temp)
            results.append({'lb': lb, 'ub': ub,
                           'method': method, 'Delta': delta_type, 'M': M_val})
    return pd.DataFrame(results)


def createSensitivityResults_relativeMagnitudes(betahat, sigma, numPrePeriods, numPostPeriods,
                                                 bound="deviation from parallel trends",
                                                 method="C-LF", Mbarvec=None, l_vec=None,
                                                 monotonicityDirection=None, biasDirection=None,
                                                 alpha=0.05, gridPoints=1000,
                                                 grid_ub=None, grid_lb=None, seed=0):
    """
    Sensitivity analysis for relative magnitude restrictions.

    Args:
        bound: 'deviation from parallel trends' (RM) or 'deviation from linear trend' (SDRM)
        method: 'C-LF' or 'Conditional'
        Mbarvec: vector of Mbar values (default: 0 to 2)

    Returns:
        DataFrame with columns: lb, ub, method, Delta, Mbar
    """
    betahat = np.asarray(betahat, dtype=float).ravel()
    sigma_np = np.asarray(sigma, dtype=float)
    if l_vec is None:
        l_vec = np.zeros(numPostPeriods)
        l_vec[0] = 1.0
    else:
        l_vec = np.asarray(l_vec, dtype=float).ravel()

    if Mbarvec is None:
        Mbarvec = np.linspace(0, 2, 10)

    if monotonicityDirection is not None and biasDirection is not None:
        raise ValueError("Select either shape or sign restriction, not both.")

    if method == "C-LF":
        hybrid_flag = "LF"
    elif method == "Conditional":
        hybrid_flag = "ARP"
    else:
        raise ValueError("method must be 'Conditional' or 'C-LF'")

    results = []
    for Mbar_val in Mbarvec:
        kw = dict(betahat=betahat, sigma=sigma_np, numPrePeriods=numPrePeriods,
                  numPostPeriods=numPostPeriods, l_vec=l_vec, Mbar=Mbar_val,
                  alpha=alpha, hybrid_flag=hybrid_flag,
                  gridPoints=gridPoints, grid_ub=grid_ub, grid_lb=grid_lb, seed=seed)

        if bound == "deviation from parallel trends":
            if monotonicityDirection is None and biasDirection is None:
                dt = "DeltaRM"
                temp = computeConditionalCS_DeltaRM(**kw)
            elif monotonicityDirection is not None:
                dt = "DeltaRMI" if monotonicityDirection == "increasing" else "DeltaRMD"
                kw['monotonicityDirection'] = monotonicityDirection
                temp = computeConditionalCS_DeltaRMM(**kw)
            else:
                dt = "DeltaRMPB" if biasDirection == "positive" else "DeltaRMNB"
                kw['biasDirection'] = biasDirection
                temp = computeConditionalCS_DeltaRMB(**kw)
        elif bound == "deviation from linear trend":
            if numPrePeriods == 1:
                raise ValueError("Not enough pre-periods for Delta^SDRM.")
            if monotonicityDirection is None and biasDirection is None:
                dt = "DeltaSDRM"
                temp = computeConditionalCS_DeltaSDRM(**kw)
            elif monotonicityDirection is not None:
                dt = "DeltaSDRMI" if monotonicityDirection == "increasing" else "DeltaSDRMD"
                kw['monotonicityDirection'] = monotonicityDirection
                temp = computeConditionalCS_DeltaSDRMM(**kw)
            else:
                dt = "DeltaSDRMPB" if biasDirection == "positive" else "DeltaSDRMNB"
                kw['biasDirection'] = biasDirection
                temp = computeConditionalCS_DeltaSDRMB(**kw)
        else:
            raise ValueError("bound must be 'deviation from parallel trends' or 'deviation from linear trend'")

        lb, ub = _extract_lb_ub(temp)
        results.append({'lb': lb, 'ub': ub, 'method': method, 'Delta': dt, 'Mbar': Mbar_val})

    return pd.DataFrame(results)


# --- Upper and Lower Bound M Functions (from ublbM_functions.R) ---

def _create_A_and_D_SD_prePeriods(numPrePeriods):
    """Create A and d for SD pre-period constraints (for bounding M)."""
    if numPrePeriods < 2:
        raise ValueError("Can't estimate M with < 2 pre-period coefficients")
    Atilde = np.zeros((numPrePeriods - 1, numPrePeriods))
    Atilde[numPrePeriods - 2, (numPrePeriods - 2):numPrePeriods] = [1.0, -2.0]
    for r in range(numPrePeriods - 2):
        Atilde[r, r:(r + 3)] = [1.0, -2.0, 1.0]
    A_pre = np.vstack([Atilde, -Atilde])
    d = np.ones(A_pre.shape[0])
    return {'A': A_pre, 'd': d}


def _test_in_identified_set_Max(M, y, sigma, A, alpha, d):
    """APR test for bounding M."""
    y = np.asarray(y, dtype=float).ravel()
    sigma_np = np.asarray(sigma, dtype=float)
    A_np = np.asarray(A, dtype=float)
    d_np = np.asarray(d, dtype=float).ravel()

    d_mod = d_np * M
    sigmaTilde = np.sqrt(np.diag(A_np @ sigma_np @ A_np.T))
    Atilde = np.diag(1.0 / sigmaTilde) @ A_np
    dtilde = (1.0 / sigmaTilde) * d_mod

    normalizedMoments = Atilde @ y - dtilde
    maxLocation = int(np.argmax(normalizedMoments))
    maxMoment = normalizedMoments[maxLocation]

    nrows = Atilde.shape[0]
    T_B = np.zeros((1, nrows))
    T_B[0, maxLocation] = 1.0
    iota = np.ones((nrows, 1))

    gamma = (T_B @ Atilde).T
    Abar = Atilde - iota @ (T_B @ Atilde)
    dbar = (np.eye(nrows) - iota @ T_B) @ dtilde

    sigmabar = float(np.sqrt(gamma.T @ sigma_np @ gamma))
    c_vec = (sigma_np @ gamma) / float(gamma.T @ sigma_np @ gamma)
    z = (np.eye(len(y)) - c_vec @ gamma.T) @ y
    VLoVUpVec = _vlo_vup_fn(
        torch.tensor(gamma), torch.tensor(sigma_np),
        torch.tensor(Abar), torch.tensor(dbar), torch.tensor(z))
    VLo, VUp = VLoVUpVec

    mu_val = float(T_B @ dtilde)
    criticalVal = _norminvp_generalized(1 - alpha, float(VLo), float(VUp),
                                         mu=mu_val, sd=sigmabar)
    reject = (maxMoment + mu_val) > criticalVal
    return bool(reject)


def DeltaSD_upperBound_Mpre(betahat, sigma, numPrePeriods, alpha=0.05):
    """Construct upper bound for M at 1-alpha level from pre-period coefficients."""
    betahat = np.asarray(betahat, dtype=float).ravel()
    sigma_np = np.asarray(sigma, dtype=float)
    assert numPrePeriods > 1

    prePeriod_coef = betahat[:numPrePeriods]
    prePeriod_sigma = sigma_np[:numPrePeriods, :numPrePeriods]

    A_SD = _create_A_SD(numPrePeriods, numPostPeriods=0)
    prePeriodCoefDiffs = A_SD @ prePeriod_coef
    prePeriodSigmaDiffs = A_SD @ prePeriod_sigma @ A_SD.T
    seDiffs = np.sqrt(np.diag(prePeriodSigmaDiffs))
    upperBoundVec = prePeriodCoefDiffs + scistats.norm.ppf(1 - alpha) * seDiffs
    return float(np.max(upperBoundVec))


def DeltaSD_lowerBound_Mpre(betahat, sigma, numPrePeriods, alpha=0.05,
                             grid_ub=None, gridPoints=1000):
    """Construct lower bound for M using APR conditional test from pre-period coefficients."""
    betahat = np.asarray(betahat, dtype=float).ravel()
    sigma_np = np.asarray(sigma, dtype=float)
    assert numPrePeriods > 1

    prePeriod_coef = betahat[:numPrePeriods]
    prePeriod_sigma = sigma_np[:numPrePeriods, :numPrePeriods]

    if grid_ub is None:
        grid_ub = 3 * float(np.max(np.sqrt(np.diag(prePeriod_sigma))))

    Ad = _create_A_and_D_SD_prePeriods(numPrePeriods)
    mGrid = np.linspace(0, grid_ub, gridPoints)

    accept_results = []
    for m_val in mGrid:
        reject = _test_in_identified_set_Max(m_val, prePeriod_coef, prePeriod_sigma,
                                              Ad['A'], alpha, Ad['d'])
        accept_results.append(1 - int(reject))

    resultsGrid = np.column_stack([mGrid, accept_results])
    if np.sum(resultsGrid[:, 1]) == 0:
        warnings.warn("ARP conditional test rejects all values of M. Increase grid upper bound.")
        return float('inf')
    else:
        return float(np.min(resultsGrid[resultsGrid[:, 1] == 1, 0]))
