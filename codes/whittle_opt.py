# whittle_exp.py
import time
import numpy as np
import mpmath as mp
from scipy.optimize import minimize
from data_gener import binsized
from spectral import periodogram, exp_fd, powerlaw_fd, gaussian_fd, sinc
from joblib import Parallel, delayed
from collections import OrderedDict
try:
    import incomplete_gamma as _incomplete_gamma # type: ignore
    _HAS_CPP_GAMMA = True
except ImportError:
    _incomplete_gamma = None
    _HAS_CPP_GAMMA = False

# C++ backend note:
# - Module: `incomplete_gamma` (see `hawkes/incomplete_gamma.cpp`).
# - Algorithm: upper incomplete gamma Γ(s, z) using
#   (1) series for lower gamma when |z| < |s| + 1,
#   (2) Lentz continued fraction for upper gamma otherwise,
#   with upward shift to Re(s) > 0.5 and recurrence to shift back.
# - Tuning: eps=1e-12, max_iter=6000 (current C++ settings).
# - Accuracy: relative error is meaningful when |Γ(s, z)| is not tiny;
#   for very small magnitudes, absolute error is more indicative.

##########---------- Exponential Hawkes Whittle Estimation -------------#
#######################################################################

def whittle_neglog_exponential_fd(param, omega, I, Delta, K_alias):
    S = exp_fd(omega, param, Delta, K_alias)
    if np.any(~np.isfinite(S)) or np.any(S <= 0):
        return 1e15
    return float(np.sum(np.log(S) + I / S))

def whittle_hawkes_exponential(
    counts: np.ndarray,
    binsize: float,
    x0=None,
    K_alias: int = 3,
    maxiter: int = 100,
    ftol: float = 1e-5,
):
    counts = np.asarray(counts, dtype=float)
    omega, I = periodogram(counts)

    if x0 is None:
        mean_count = counts.mean()
        baseline0 = max(mean_count / binsize * 0.5, 0.1)
        mu0 = 0.5
        beta0 = 1.0
        x0 = np.array([baseline0, mu0, beta0], dtype=float)

    bounds = [(0.01, None), (0.01, 0.99), (0.01, None)]

    res = minimize(
        whittle_neglog_exponential_fd,
        x0=x0,
        args=(omega, I, binsize, K_alias),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter, "ftol": ftol, "disp": False},
    )
    return res.x, res

##########---------- Power-law Hawkes Whittle Estimation -------------#
#######################################################################

_A_CACHE = OrderedDict()

# History note:
# - Cache was added when A(xi) relied on slow mpmath; kept for optional reuse.
# - Gamma upper bounds used for early stability checks were removed to match
#   the paper's unbounded gamma, leaving only numerical overflow guards.

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def _logit(p):
    return np.log(p / (1.0 - p))

def _gamma_upper_complex(s, z):
    if _HAS_CPP_GAMMA:
        return _incomplete_gamma.compute_complex_incomplete_gamma( # type: ignore
            float(s.real), float(s.imag), float(z.real), float(z.imag)
        )
    return mp.gammainc(s, z, mp.inf)

def _gamma_upper_complex_vec(s, xi, a):
    if _HAS_CPP_GAMMA:
        return _incomplete_gamma.compute_complex_incomplete_gamma_xi( # type: ignore
            float(s.real), float(s.imag), xi, float(a)
        )
    return np.array([_gamma_upper_complex(s, 1j * x * a) for x in xi], dtype=complex)

def _compute_A_vectorized(xi, log_gamma, a):
    gamma = np.exp(log_gamma)
    xi = np.asarray(xi, dtype=float)
    xi_abs = np.abs(xi)
    A = np.empty_like(xi_abs, dtype=np.complex128)
    zero_mask = xi_abs == 0.0
    if np.any(zero_mask):
        A[zero_mask] = 1.0
    if np.any(~zero_mask):
        xi_nz = xi_abs[~zero_mask]
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                A_pos = (
                    gamma * (a ** gamma)
                    * np.exp(1j * xi_nz * a)
                    * (1j * xi_nz) ** gamma
                    * _gamma_upper_complex_vec(-gamma, xi_nz, a)
                )
        except (FloatingPointError, OverflowError, ZeroDivisionError):
            A_pos = np.full(xi_nz.shape, np.nan + 0j, dtype=np.complex128)
        neg_mask = xi[~zero_mask] < 0.0
        A_nz = A_pos.copy()
        if np.any(neg_mask):
            A_nz[neg_mask] = np.conj(A_pos[neg_mask])
        A[~zero_mask] = A_nz
    return A

def precompute_freq_stuff(counts, Delta, K_alias, k0=200, step_hi=5):
    omega, I = periodogram(counts)
    idx = np.argsort(omega)
    omega, I = omega[idx], I[idx]
    N = omega.size
    k0 = min(k0, N)
    omega = np.concatenate([omega[:k0], omega[k0::step_hi]])
    # print("omega =", omega, "N =", N)
    I = np.concatenate([I[:k0], I[k0::step_hi]])
    
    xi_list = []
    sinc2_list = []
    
    for k in range(-K_alias, K_alias + 1):
        omg = omega + 2 * np.pi * k
        xi_list.append(omg / Delta)
        sinc2_list.append(sinc(omg / 2.0) ** 2)
    
    return omega, I, xi_list, sinc2_list

def pick_powerlaw_init_by_grid(
    counts,
    Delta,
    K_alias,
    a=1.5,
    s=5.0,
    k0=200,
    step_hi=5,
    mu_grid=(0.25, 0.5, 0.75),
    gamma_grid=(0.5, 1.0, 1.5, 2.0, 2.5, 3.2, 4.0, 5.0, 7.5, 10.0),
    n_jobs=-1,
    backend="loky",
    parallel_min_size=5000,
    use_cache=True,
    cache_max=256,
    tol_log=1e-4,
):
    """
    Coarse grid search for (mu, gamma) to initialize Whittle optimization.
    Returns (mu0, gamma0, best_obj).
    """
    counts = np.asarray(counts, float)
    _, I, xi_list, sinc2_list = precompute_freq_stuff(
        counts, Delta, K_alias, k0=k0, step_hi=step_hi
    )

    best_mu = None
    best_gamma = None
    best_obj = np.inf

    for mu in mu_grid:
        for gamma in gamma_grid:
            if not (1e-6 < mu < 1 - 1e-6):
                continue
            if not (1e-3 < gamma):
                continue
            theta2 = np.array([_logit(mu), np.log(gamma) / s], float)
            obj = whittle_obj_mu_gamma_vec(
                theta2,
                I,
                xi_list,
                sinc2_list,
                Delta,
                a,
                s,
            )
            if (not np.isfinite(obj)) or (obj >= 1e14):
                continue
            if obj < best_obj:
                best_obj = obj
                best_mu = mu
                best_gamma = gamma

    return best_mu, best_gamma, best_obj

def compute_A_vector_serial(xi, log_gamma, a, alias_id, tol_log=1e-4, use_cache=True):
    """
    Calculate A(xi) serially for one alias block.
    """
    if use_cache:
        lg_key = round(float(log_gamma) / tol_log) * tol_log
    else:
        lg_key = log_gamma

    gamma = np.exp(lg_key)
    A = np.empty_like(xi, dtype=np.complex128)

    for i in range(xi.size):
        if xi[i] == 0.0:
            A[i] = 1.0
            continue
        xi_abs = abs(xi[i])
        z = 1j * xi_abs * a
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                A_pos = (
                    gamma * (a ** gamma)
                    * np.exp(1j * xi_abs * a)
                    * (1j * xi_abs) ** gamma
                    * _gamma_upper_complex(-gamma, z)
                )
            A[i] = np.conj(A_pos) if xi[i] < 0.0 else A_pos
        except (FloatingPointError, OverflowError, ZeroDivisionError):
            A[i] = np.nan

    return A

def whittle_obj_mu_gamma_vec(
    theta2,
    I,
    xi_list,
    sinc2_list,
    Delta,
    a,
    s,
):
    """
    Vectorized objective without cache.
    """
    logit_mu, u = theta2
    mu = _sigmoid(logit_mu)

    log_gamma = s * u
    log_max = np.log(np.finfo(float).max)
    if (not np.isfinite(log_gamma)) or (log_gamma > log_max - 5.0):
        return 1e15

    gamma = np.exp(log_gamma)
    if not (1e-6 < mu < 1 - 1e-6) or not (1e-3 < gamma):
        return 1e15

    C = np.zeros_like(I, float)
    for xi, sinc2_k in zip(xi_list, sinc2_list):
        A = _compute_A_vectorized(xi, log_gamma, a)
        if not np.all(np.isfinite(A)):
            return 1e15
        C += sinc2_k / (np.abs(1.0 - mu * A) ** 2)

    C *= Delta
    m_hat = float(np.mean(I / C))
    if not np.isfinite(m_hat) or m_hat <= 0:
        return 1e15

    return I.size * np.log(m_hat) + float(np.sum(np.log(C))) + I.size

def whittle_powerlaw(
    counts,
    Delta,
    K_alias: int = 2,
    a: float = 1.5,
    maxiter: int = 40,
    s: float = 5.0,
    k0: int = 200,
    step_hi: int = 5,
    ftol: float = 1e-5,
    eps: float = 1e-3,
    n_jobs: int = -1,
    backend: str = "loky",
    parallel_min_size: int = 5000,
    use_cache: bool = False,
    cache_max: int = 256,
    tol_log: float = 1e-4,
    use_grid_init: bool = True,
    mu_grid=(0.25, 0.5, 0.75),
    gamma_grid=(0.5, 1.0, 1.5, 2.0, 2.5, 3.2, 4.0, 5.0, 7.5, 10.0),
    theta0=None,
    init_jitter=None,
):
    """
    Vectorized Whittle estimation for power-law Hawkes.
    Returns: (eta, mu, gamma), scipy OptimizeResult
    """
    counts = np.asarray(counts, float)
    omega, I, xi_list, sinc2_list = precompute_freq_stuff(
        counts, Delta, K_alias, k0=k0, step_hi=step_hi
    )

    if theta0 is None and use_grid_init:
        mu0, gamma0, best_obj = pick_powerlaw_init_by_grid(
            counts,
            Delta,
            K_alias,
            a=a,
            s=s,
            k0=k0,
            step_hi=step_hi,
            mu_grid=mu_grid,
            gamma_grid=gamma_grid,
            n_jobs=1,
            backend="loky",
            parallel_min_size=0,
            use_cache=False,
            cache_max=0,
            tol_log=1e-6,
        )
        if (mu0 is None) or (gamma0 is None) or (not np.isfinite(best_obj)) or (best_obj >= 1e14):
            mu0, gamma0 = 0.5, 2.5
    elif theta0 is None:
        mu0, gamma0 = 0.5, 2.5

    if theta0 is None:
        theta0 = np.array([_logit(mu0), np.log(gamma0) / s], float)
    if init_jitter is not None:
        theta0 = np.asarray(theta0, float) + np.asarray(init_jitter, float)

    res = minimize(
        whittle_obj_mu_gamma_vec,
        x0=theta0,
        args=(I, xi_list, sinc2_list, Delta, a, s),
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": ftol, "disp": False, "eps": eps},
    )

    logit_mu, u = res.x
    mu = _sigmoid(logit_mu)
    gamma = np.exp(s * u)

    log_gamma = s * u
    log_max = np.log(np.finfo(float).max)
    if (not np.isfinite(log_gamma)) or (log_gamma > log_max - 5.0):
        return np.array([np.nan, np.nan, np.nan], float), res

    C = np.zeros_like(I, float)
    for xi, sinc2_k in zip(xi_list, sinc2_list):
        A = _compute_A_vectorized(xi, log_gamma, a)
        if not np.all(np.isfinite(A)):
            return np.array([np.nan, np.nan, np.nan], float), res
        C += sinc2_k / (np.abs(1.0 - mu * A) ** 2)

    C *= Delta
    m_hat = float(np.mean(I / C))
    eta_hat = m_hat * (1.0 - mu)

    return np.array([eta_hat, mu, gamma], float), res

##########---------- Guassian Hawkes Whittle Estimation -------------#
#######################################################################

def whittle_neglog_gaussian_fd(param, omega, I, Delta, K_alias):
    S = gaussian_fd(omega, param, Delta, K_alias)
    if np.any(~np.isfinite(S)) or np.any(S <= 0):
        return 1e15
    return float(np.sum(np.log(S) + I / S))

def whittle_hawkes_gaussian(
    counts: np.ndarray,
    binsize: float,
    x0=None,
    K_alias: int = 3,
    maxiter: int = 100,
    ftol: float = 1e-5,
):
    counts = np.asarray(counts, dtype=float)
    omega, I = periodogram(counts)

    if x0 is None:
        mean_count = counts.mean()
        baseline0 = max(mean_count / binsize * 0.5, 0.1)
        mu0 = 0.5
        sigma0 = 1.0
        nu0 = 4.0
        x0 = np.array([baseline0, mu0, sigma0, nu0], dtype=float)

    bounds = [(0.01, None), (0.01, 0.99), (0.01, None), (0.01, None)]

    res = minimize(
        whittle_neglog_gaussian_fd,
        x0=x0,
        args=(omega, I, binsize, K_alias),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter, "ftol": ftol, "disp": False},
    )
    return res.x, res

def whittle_gaussian_opt(
    counts,
    Delta,
    K_alias,
    x0=None,
    maxiter=100,
    ftol=1e-5,
):
    """
    Small optimizer for real (windowed) count data using the Gaussian Whittle estimator.
    counts: 1D array of binned counts (window data).
    """
    counts = np.asarray(counts, dtype=float)
    par_hat, res = whittle_hawkes_gaussian(
        counts,
        Delta,
        x0=x0,
        K_alias=K_alias,
        maxiter=maxiter,
        ftol=ftol,
    )
    return par_hat, res
