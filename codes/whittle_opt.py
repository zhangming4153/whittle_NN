# whittle_exp.py
import numpy as np
from scipy.optimize import minimize
from spectral import periodogram, sinc, exp_fd,powerlaw_fd
import mpmath as mp

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def _logit(p):
    return np.log(p / (1.0 - p))

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


# =====================
# Frequency precomputation
# =====================
def precompute_freq_stuff(counts, Delta, K_alias, k0=200, step_hi=5):
    omega, I = periodogram(counts)
    idx = np.argsort(omega)
    omega, I = omega[idx], I[idx]

    N = omega.size
    k0 = min(k0, N)

    omega = np.concatenate([omega[:k0], omega[k0::step_hi]])
    I     = np.concatenate([I[:k0],     I[k0::step_hi]])

    xi_list = []
    sinc2_list = []

    for k in range(-K_alias, K_alias + 1):
        omg = omega + 2 * np.pi * k
        xi_list.append(omg / Delta)
        sinc2_list.append(sinc(omg / 2.0) ** 2)

    return omega, I, xi_list, sinc2_list


# =====================
# Cached A(xi; gamma)
# =====================
_A_cache = {}

def compute_A_vector_cached(xi, log_gamma, a, alias_id, tol_log=1e-3):
    """
    Cache A(xi) = FT(h) evaluated at fixed log_gamma (binned)
    """
    lg_key = round(float(log_gamma) / tol_log) * tol_log
    key = (alias_id, lg_key, xi.size)

    if key in _A_cache:
        return _A_cache[key]

    gamma = np.exp(lg_key)
    A = np.empty_like(xi, dtype=np.complex128)

    for i in range(xi.size):
        z = 1j * xi[i] * a
        A[i] = (
            gamma * (a ** gamma)
            * np.exp(1j * xi[i] * a)
            * (1j * xi[i]) ** gamma
            * mp.gammainc(-gamma, z, mp.inf)
        )

    _A_cache[key] = A
    return A


# =====================
# Whittle objective (profiled m)
# =====================
def whittle_obj_mu_gamma(theta2, I, xi_list, sinc2_list, Delta, a, s):
    """
    theta2 = [logit_mu, u], gamma = exp(s * u)
    """
    logit_mu, u = theta2
    mu = _sigmoid(logit_mu)

    log_gamma = s * u
    if not np.isfinite(log_gamma):
        return 1e15

    gamma = np.exp(log_gamma)
    if not (1e-6 < mu < 1 - 1e-6) or not (1e-3 < gamma < 5.0):
        return 1e15

    C = np.zeros_like(I, float)
    for j, (xi, sinc2_k) in enumerate(zip(xi_list, sinc2_list)):
        A = compute_A_vector_cached(xi, log_gamma, a, alias_id=j)
        C += sinc2_k / (np.abs(1.0 - mu * A) ** 2)

    C *= Delta
    m_hat = float(np.mean(I / C))
    if not np.isfinite(m_hat) or m_hat <= 0:
        return 1e15

    return I.size * np.log(m_hat) + float(np.sum(np.log(C))) + I.size


# =====================
# Main fast estimator
# =====================
def whittle_powerlaw_fast(
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
):
    """
    Fast Whittle estimation for power-law Hawkes.
    Returns: (eta, mu, gamma), scipy OptimizeResult
    """
    _A_cache.clear()

    counts = np.asarray(counts, float)
    omega, I, xi_list, sinc2_list = precompute_freq_stuff(
        counts, Delta, K_alias, k0=k0, step_hi=step_hi
    )

    mu0 = 0.5
    gamma0 = 2.5
    theta0 = np.array([_logit(mu0), np.log(gamma0) / s], float)

    res = minimize(
        whittle_obj_mu_gamma,
        x0=theta0,
        args=(I, xi_list, sinc2_list, Delta, a, s),
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": ftol, "disp": False, "eps": eps},
    )

    logit_mu, u = res.x
    mu = _sigmoid(logit_mu)
    gamma = np.exp(s * u)

    log_gamma = s * u
    C = np.zeros_like(I, float)
    for j, (xi, sinc2_k) in enumerate(zip(xi_list, sinc2_list)):
        A = compute_A_vector_cached(xi, log_gamma, a, alias_id=j)
        C += sinc2_k / (np.abs(1.0 - mu * A) ** 2)

    C *= Delta
    m_hat = float(np.mean(I / C))
    eta_hat = m_hat * (1.0 - mu)

    # print("nfev:", res.nfev)
    # print("A cache size:", len(_A_cache))

    return np.array([eta_hat, mu, gamma], float), res
