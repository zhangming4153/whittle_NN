import numpy as np
import mpmath as mp
from scipy.optimize import minimize
from numba import njit
_NUMBA_AVAILABLE = True


def loglik_exp_hawkes(events, eta, mu, beta, T):
    """
    Exponential Hawkes with reproduction kernel h(t) = mu * beta * exp(-beta t), t>0
    events: may contain t < 0 (burn-in) and 0 <= t <= T (observed).
    Likelihood is for [0, T], but history includes t < 0.
    """
    events = np.sort(np.asarray(events, float))
    pre_events = events[events < 0]
    obs_events = events[(events >= 0) & (events <= T)]
    n = len(obs_events)
    
    # ---------- 1. Integral term ∫_0^T λ(t) dt ----------
    # baseline part
    integral_term = eta * T

    # 1.1 contribution from observed events 0 <= s <= T:
    # ∫_0^T mu*beta*exp(-beta(t - s)) 1_{t>s} dt = mu*(1 - exp(-beta*(T - s)))
    if n > 0:
        dt_obs = T - obs_events
        contrib_obs = np.sum(1.0 - np.exp(-beta * dt_obs))
    else:
        contrib_obs = 0.0

    # 1.2 contribution from history events s < 0:
    # For s < 0, kernel is active on the whole [0,T]:
    # ∫_0^T mu*beta*exp(-beta(t - s)) dt = mu * exp(beta*s) * (1 - exp(-beta*T))
    if pre_events.size > 0:
        contrib_pre = np.sum(np.exp(beta * pre_events)) * (1.0 - np.exp(-beta * T))
    else:
        contrib_pre = 0.0

    integral_term += mu * (contrib_obs + contrib_pre)
    
    # ---------- 2. Log-sum term Σ log λ(t_i), t_i ∈ [0,T] ----------
    if n == 0:
        # no observed events → purely integral
        return -integral_term

    log_intensity_sum = 0.0
    recursive_term = 0.0  # this will store sum_j exp(-beta (t_i - t_j))

    for i in range(n):
        ti = obs_events[i]
        if i == 0:
            # first observed event: history includes all pre_events (t < 0)
            if pre_events.size > 0:
                dt0 = ti - pre_events           # >0
                recursive_term = np.sum(np.exp(-beta * dt0))
            else:
                recursive_term = 0.0
        else:
            # recursive update for later events
            dt = obs_events[i] - obs_events[i-1]
            recursive_term = np.exp(-beta * dt) * (recursive_term + 1.0)

        lam = eta + (mu * beta) * recursive_term
        if lam <= 0:
            return -np.inf
        log_intensity_sum += np.log(lam)
    return log_intensity_sum - integral_term

def neg_loglik_theta(theta, events, T):
    eta, mu, beta = theta
    # 简单保护：参数越界就返回 +inf
    if eta <= 0 or mu <= 0 or mu >= 1 or beta <= 0:
        return np.inf
    return -loglik_exp_hawkes(events, eta, mu, beta, T)

def fit_hawkes_lbfgsb(events, T, theta0=None, maxiter=500, ftol=1e-5):
    events = np.asarray(events, float)
    # 初始值：可以用简单经验值
    if theta0 is None:
        # 粗糙一点：η ~ n/T * 0.5, μ ~ 0.5, β ~ 1
        n = len(events)
        theta0 = np.array([max(n / T * 0.5, 1e-3), 0.5, 1.0])
    
    # 边界：η>0, μ∈(0,1), β>0
    eps = 1e-6
    bounds = [
        
        (eps, None),       # eta > 0
        (eps, 1.0 - eps),  # 0 < mu < 1
        (eps, None)        # beta > 0
    ]
    
    res = minimize(
        neg_loglik_theta,
        theta0,
        args=(events, T),
        method='L-BFGS-B',
        bounds=bounds,
        options={'disp': False, 'maxiter': maxiter, 'ftol': ftol}
    )
    return res

def h_power(t, gamma, a=1.5):
    return gamma * (a ** gamma) * (a + t) ** (-1.0 - gamma)

if _NUMBA_AVAILABLE:
    @njit(cache=True)
    def _loglik_power_hawkes_full_numba(events, obs, pre, R, eta, mu, gamma, a, T):
        if obs.size == 0:
            return -(eta * T)

        log_sum = 0.0
        a_pow = a ** gamma
        for i in range(obs.size):
            ti = obs[i]
            Ri = R[i]
            hist_term = 0.0
            for j in range(Ri):
                dt = ti - events[j]
                hist_term += gamma * a_pow * (a + dt) ** (-1.0 - gamma)

            lam = eta + mu * hist_term
            if lam <= 0.0 or not np.isfinite(lam):
                return -np.inf
            log_sum += np.log(lam)

        integral = eta * T
        contrib_obs = 0.0
        for i in range(obs.size):
            dt_obs = T - obs[i]
            contrib_obs += 1.0 - (a / (a + dt_obs)) ** gamma

        contrib_pre = 0.0
        if pre.size:
            for i in range(pre.size):
                dt0 = -pre[i]
                dtT = T - pre[i]
                contrib_pre += (a / (a + dt0)) ** gamma - (a / (a + dtT)) ** gamma

        integral += mu * (contrib_obs + contrib_pre)
        return log_sum - integral

def precompute_full(events, T):
    events = np.sort(np.asarray(events, float))
    obs = events[(events >= 0.0) & (events <= T)]
    pre = events[events < 0.0]
    R = np.searchsorted(events, obs, side="left")
    return events, obs, pre, R

def loglik_power_hawkes_full(events, obs, pre, R, eta, mu, gamma, a, T):
    if _NUMBA_AVAILABLE:
        return _loglik_power_hawkes_full_numba(events, obs, pre, R, eta, mu, gamma, a, T)

    if obs.size == 0:
        return -(eta * T)

    log_sum = 0.0
    for i, ti in enumerate(obs):
        hist = events[:R[i]]  # ALL events before ti
        if hist.size:
            dt = ti - hist
            # dt is automatically > 0 because hist < ti
            hist_term = np.sum(h_power(dt, gamma=gamma, a=a))
        else:
            hist_term = 0.0

        lam = eta + mu * hist_term
        if lam <= 0 or not np.isfinite(lam):
            return -np.inf
        log_sum += np.log(lam)

    # exact integral (same as your current one)
    integral = eta * T
    dt_obs = T - obs
    contrib_obs = np.sum(1.0 - (a / (a + dt_obs)) ** gamma)

    if pre.size:
        dt0 = -pre
        dtT = T - pre
        contrib_pre = np.sum((a / (a + dt0)) ** gamma - (a / (a + dtT)) ** gamma)
    else:
        contrib_pre = 0.0

    integral += mu * (contrib_obs + contrib_pre)
    return log_sum - integral

def pick_gamma0_by_grid(
    events,
    T,
    a=1.5,
    eta0=None,
    gamma_grid=(1.0, 1.5, 2.0, 2.5, 3.2, 4.0, 5.0),
    mu_grid=(0.25, 0.5, 0.75),
):
    """
    Evaluate NLL at a small grid of gamma values with (eta, mu) fixed.
    Return (eta0, mu0_best, gamma0_best).
    """
    events = np.asarray(events, float)
    events_s, obs, pre, R = precompute_full(events, T)
    n = obs.size

    best_g = None
    best_mu = None
    best_nll = np.inf

    for mu in mu_grid:
        for g in gamma_grid:
            eta0 = max((1.0 - mu) * (n / T), 1e-3)
            ll = loglik_power_hawkes_full(events_s, obs, pre, R, eta0, mu, g, a=a, T=T)
            nll = 1e50 if not np.isfinite(ll) else -ll
            if nll < best_nll:
                best_nll = nll
                best_g = g
                best_mu = mu

    return eta0, best_mu, best_g, (events_s, obs, pre, R)

def fit_power_hawkes_lbfgsb(events, T, a=1.5, theta0=None, maxiter=60, ftol=1e-4, precomputed=None):
    events = np.asarray(events, float)
    if precomputed is None:
        events_s, obs, pre, R = precompute_full(events, T)
    else:
        events_s, obs, pre, R = precomputed
    n = obs.size

    if theta0 is None:
        mu0 = 0.5
        eta0 = max((1 - mu0) * (n / T), 1e-3)
        gamma0 = 2.5
        theta0 = np.array([eta0, mu0, gamma0], float)

    bounds = [
        (1e-6, None),       # eta > 0
        (1e-6, 1 - 1e-6),   # 0 < mu < 1
        (0.5, 10.0)         # gamma
    ]

    def neg_ll(theta):
        eta, mu, gamma = theta
        ll = loglik_power_hawkes_full(events_s, obs, pre, R, eta, mu, gamma, a=a, T=T)
        if not np.isfinite(ll):
            return 1e50
        return -ll

    res = minimize(
        neg_ll,
        theta0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"disp": False,
                 "ftol": ftol,
                "maxiter": maxiter}
    )
    return res

def fit_power_hawkes_A(events, T, a=1.5, maxiter=60, ftol=1e-4,
                             mu0=0.5,
                             gamma_grid=(1.0, 1.5, 2.0, 2.5, 3.2, 4.0, 5.0),
                             verbose=False):
    # Step 1: pick gamma0 cheaply
    eta0, mu0, gamma0, precomputed = pick_gamma0_by_grid(events, T, a=a,  gamma_grid=gamma_grid)

    if verbose:
        print(f"[grid pick] eta0={eta0:.4f}, mu0={mu0:.3f}, gamma0={gamma0:.3f}")

    # Step 2: run ONE optimizer from that theta0
    theta0 = np.array([eta0, mu0, gamma0], float)
    res = fit_power_hawkes_lbfgsb(
        events, T, a=a, theta0=theta0, maxiter=maxiter, ftol=ftol, precomputed=precomputed
    )
    return res

def estimate_power_hawkes(events, 
                          T, 
                          a=1.5, 
                          mu0=0.5,
                          gamma_grid=(0.6, 1.0, 1.7, 2.9, 5.0),
                          gamma_refine=(1.8, 2.2, 2.6, 3.0),
                          maxiter=80,
                          ftol=1e-5):
    
    eta0, mu0, gamma0, precomputed = pick_gamma0_by_grid(events, T, a=a, gamma_grid=gamma_grid)
    refine_grid = tuple(sorted(set(gamma0 * f for f in gamma_refine)))
    eta0, mu0, gamma0, _ = pick_gamma0_by_grid(
        events, T, a=a, eta0=eta0, gamma_grid=refine_grid
    )
    theta0 = np.array([eta0, mu0, gamma0], float)
    res = fit_power_hawkes_lbfgsb(
        events, T, a=a, theta0=theta0, maxiter=maxiter, ftol=ftol, precomputed=precomputed
    )
    return res.x, res.success
