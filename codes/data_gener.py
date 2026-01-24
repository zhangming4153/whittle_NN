import csv
import numpy as np
from numba import njit

def binsized(events, Delta, T, dynamic=False, dynamic_times=None):
    events = np.asarray(events, float)
    events = events[(events > 0) & (events <= T)]  # Filter events within the total time T
    
    if dynamic:
        if dynamic_times is None:
            raise ValueError("dynamic_times must be provided for dynamic binning.")
        
        # Ensure dynamic_times is sorted to define the bin edges correctly
        dynamic_times = np.sort(dynamic_times)
        
        # Add edges for the start and end
        edges = np.concatenate(([0.0], dynamic_times, [T]))
        
        # Efficient counting of events in each dynamic bin
        counts = np.zeros(len(edges) - 1, dtype=int)
        for i in range(len(edges) - 1):
            counts[i] = np.sum((events > edges[i]) & (events <= edges[i + 1]))
        return counts
    
    else:
        # Fixed binning with evenly spaced intervals
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

import matplotlib.pyplot as plt

def shifted_gaussian_kernel(t, tj, nu, sigma):
    """Noncausal shifted Gaussian kernel on the full real line (paper setting)."""
    return (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-((t - tj - nu) ** 2) / (2.0 * sigma ** 2))

def NonCausal_Gaussian_Hawkes(eta=1.0, mu=0.5, sigma=0.3, nu=0.3, T=1000, burn_in=100.0, seed=2024, max_events=100000):
    rng = np.random.default_rng(seed)
    events = []
    t = 0.0
    h_max = 1.0 / (sigma * np.sqrt(2.0 * np.pi))

    while t < T and len(events) < max_events:
        # Ogata thinning with a conservative bound; kernel uses the paper's noncausal form.
        lambda_star = eta + mu * len(events) * h_max
        if lambda_star <= 0:
            break

        t += rng.exponential(1.0 / lambda_star)
        if t > T:
            break

        intensity = eta
        for tj in events:
            intensity += mu * shifted_gaussian_kernel(t, tj, nu, sigma)

        if rng.random() <= intensity / lambda_star:
            events.append(t)

    return np.asarray(events, dtype=float)


# Define the number of days per year
# total_days = sum([365, 365, 365, 366, 365, 365, 365, 366, 365, 365, 365, 366, 365, 365, 365, 366])
# days_per_month = [
#     31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,  # 2005
#     31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,  # 2006
#     31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,  # 2007
#     31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,  # 2008 (Leap Year)
#     31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,  # 2009
#     31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,  # 2010
#     31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,  # 2011
#     31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,  # 2012 (Leap Year)
#     31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,  # 2013
#     31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,  # 2014
#     31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,  # 2015
#     31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,  # 2016 (Leap Year)
#     31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,  # 2017
#     31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,  # 2018
#     31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,  # 2019
#     31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,  # 2020 (Leap Year)
# ]
# # print(sum(days_per_month), total_days)  # Verify total days calculation

# dynamic_times = np.cumsum(days_per_month)
# Calculate total days


# Parameters for the simulation
# eta = 0.05  # base intensity
# mu = 0.7   # scaling factor for Gaussian kernel
# sigma = 0.3  # standard deviation of the Gaussian kernel
# nu = 0.2    # shift of the Gaussian kernel
# T = total_days    # total time
# burn_in = 0  # burn-in period
# seed = 2025
# counting_period = 30  # days

# # Simulate the events using non-causal Gaussian Hawkes process
# events = NonCausal_Gaussian_Hawkes(eta=eta, mu=mu, sigma=sigma, nu=nu, T=T, burn_in=burn_in, seed=seed)
# bindata = binsized(events, Delta= counting_period, T=T, dynamic=True, dynamic_times=dynamic_times)

# # print(len(bindata), len(dynamic_times))

# plt.plot(bindata, marker='o', linestyle='-', color='b', label='Event counts per month')
# plt.xlabel('Month')
# plt.ylabel('Event Occurrence')
# plt.title('Event Simulation based on Gaussian Kernel (2005-2020)')
# plt.legend()
# plt.show()

# import pandas as pd

# 使用pandas读取Excel（文件内容是xlsx格式）
# df = pd.read_excel("hawkes/measles_2005_2020_shandong.csv", engine="openpyxl")
# y = df.iloc[:, 1]

# plt.plot(y, marker="o", linestyle="-")
# plt.xlabel("Index")
# plt.ylabel(df.columns[1])
# plt.title(f"Line plot of {df.columns[1]}")
# plt.tight_layout()
# plt.show()

# df = pd.read_csv("hawkes/measles_2022_2025_us.csv", encoding ="utf-8")
# y = df.iloc[:, 2]

# plt.plot(y, marker="o", linestyle="-")
# plt.xlabel("Index")
# plt.ylabel(df.columns[1])
# plt.title(f"Line plot of {df.columns[1]}")
# plt.tight_layout()
# plt.show()
