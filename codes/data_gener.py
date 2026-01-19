import numpy as np
from numba import njit

def binsized(events, Delta, T):
    events = np.asarray(events, float)
    events = events[(events > 0) & (events <= T)]
    edges = np.arange(0.0, T + Delta, Delta)
    counts, _ = np.histogram(events, bins=edges)
    return counts

def Branching(eta=1.0, mu=0.5, beta=1.0, T=1000, burn_in=100.0, seed=2024):
    rng = np.random.default_rng(seed)
    t0 = -burn_in
    length = T + burn_in

    N0 = rng.poisson(eta * length)
    immigrants = rng.uniform(low=t0, high=T, size=N0)

    events = []
    stack = list(immigrants)

    while stack:
        ptime = stack.pop()
        events.append(ptime)

        kids = rng.poisson(mu)
        if kids == 0:
            continue

        inter = rng.exponential(1 / beta, size=kids)
        ctimes = ptime + inter
        ctimes = ctimes[ctimes <= T]
        stack.extend(ctimes.tolist())

    events = np.asarray(events, float)
    events = events[events > 0.0]
    events.sort()
    return events


@njit(cache=True)
def ogata_power_hawkes(
    T, eta, mu, gamma, a, burn_in, seed, max_events=200_000
):
    np.random.seed(seed)

    ev = np.empty(max_events, dtype=np.float64)
    m = 0

    t = -burn_in
    lambda_star = eta
    h0 = gamma / a

    while t < T and m < max_events:
        lambda_bar = lambda_star
        if lambda_bar <= 0:
            break

        t_cand = t + np.random.exponential(1.0 / lambda_bar)
        if t_cand > T:
            break

        lam_cand = eta
        for i in range(m):
            dt = t_cand - ev[i]
            if dt > 0.0:
                lam_cand += mu * gamma * (a ** gamma) * (a + dt) ** (-1.0 - gamma)

        if np.random.random() <= lam_cand / lambda_bar:
            ev[m] = t_cand
            m += 1
            lambda_star = lam_cand + mu * h0
        else:
            lambda_star = lam_cand

        t = t_cand

    return ev[:m].copy()
