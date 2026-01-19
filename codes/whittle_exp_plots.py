import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from data_gener import Branching, binsized
from whittle_opt import whittle_hawkes_exponential


def generate_events(true_param, T, seed):
    return Branching(
        eta=true_param[0],
        mu=true_param[1],
        beta=true_param[2],
        T=T,
        seed=seed
    )

# Function to perform binning (分箱处理)
def get_binned_data(events, Delta, T):
    return binsized(events, Delta, T)

# Function to estimate parameters using Whittle method
def estimate_with_whittle(bindata, Delta, true_param, K_alias, maxiter=100, ftol=1e-5):
    return whittle_hawkes_exponential(
        bindata,
        binsize=Delta,
        x0=true_param,
        K_alias=K_alias,
        maxiter=maxiter,
        ftol=ftol,
    )

# Function to estimate parameters using Maximum Likelihood Estimation (MLE)
def estimate_with_mle(events, T, true_param, K_alias, R):
    # Replace this with your MLE estimation process
    # Here we can either call a different function or implement MLE logic
    pass

# Function to simulate data, perform binning, and estimate parameters using Whittle method
def simulate_and_estimate(true_param, T_list, Delta, R, K_alias, maxiter=100, ftol=1e-5, time_every=0):
    mse_eta, mse_mu, mse_beta = [], [], []

    for T in T_list:
        estimates = []
        fail_T = 0

        block_elapsed = 0.0
        for r in range(R):
            seed = 2025 + r + 100_000 * T

            # --- EXP Hawkes DGP ---
            events = generate_events(true_param, T, seed)

            # bin counts
            bindata = get_binned_data(events, Delta, T)

            # --- Whittle estimate (EXP kernel) ---
            start_time = time.perf_counter()
            par_hat, res = estimate_with_whittle(
                bindata, Delta, true_param, K_alias, maxiter=maxiter, ftol=ftol
            )
            block_elapsed += time.perf_counter() - start_time

            if time_every and (r + 1) % time_every == 0:
                print(
                    f"T={T}, Delta={Delta}: "
                    f"Block({time_every}) Time = {block_elapsed:.4f}s"
                )
                block_elapsed = 0.0

            if (not res.success) or (not np.all(np.isfinite(par_hat))):
                fail_T += 1
                continue
        
            estimates.append(par_hat)
        
        estimates = np.asarray(estimates)

        if estimates.size == 0:
            mse_eta.append(np.nan)
            mse_mu.append(np.nan)
            mse_beta.append(np.nan)
        else:
            mse = np.mean((estimates - true_param) ** 2, axis=0)
            mse_eta.append(mse[0])
            mse_mu.append(mse[1])
            mse_beta.append(mse[2])

    return mse_eta, mse_mu, mse_beta

def simulate_and_collect_estimates(
    true_param,
    T,
    Delta,
    R,
    K_alias,
    maxiter=100,
    ftol=1e-5,
    time_every=0,
):
    estimates = []

    block_elapsed = 0.0
    for r in range(R):
        seed = 2025 + r + 100_000 * T

        # --- EXP Hawkes DGP ---
        events = generate_events(true_param, T, seed)

        # bin counts
        bindata = get_binned_data(events, Delta, T)

        # --- Whittle estimate (EXP kernel) ---
        start_time = time.perf_counter()
        par_hat, res = estimate_with_whittle(
            bindata, Delta, true_param, K_alias, maxiter=maxiter, ftol=ftol
        )
        block_elapsed += time.perf_counter() - start_time

        if time_every and (r + 1) % time_every == 0:
            print(
                f"T={T}, Delta={Delta}: "
                f"Block({time_every}) Time = {block_elapsed:.4f}s"
            )
            block_elapsed = 0.0

        if (not res.success) or (not np.all(np.isfinite(par_hat))):
            continue

        estimates.append(par_hat)

    estimates = np.asarray(estimates)

    if estimates.size == 0:
        empty = np.array([], dtype=float)
        return empty, empty, empty

    return estimates[:, 0], estimates[:, 1], estimates[:, 2]

# Function to plot boxplot results
def plot_boxplot_results(ax, estimates, labels, true_values):
    try:
        ax.boxplot(
            estimates,
            tick_labels=labels,
            flierprops=dict(
                marker="o",
                markerfacecolor="black",
                markeredgecolor="black",
                markersize=2
            )
        )
    except TypeError:
        ax.boxplot(
            estimates,
            labels=labels,
            flierprops=dict(
                marker="o",
                markerfacecolor="black",
                markeredgecolor="black",
                markersize=2
            )
        )
    positions = np.arange(1, len(labels) + 1)
    means = [np.nanmean(e) if np.size(e) else np.nan for e in estimates]
    ax.plot(positions, means, "D", color="tab:blue", markersize=6, label="Mean")
    ax.plot(positions, true_values, "o", color="red", markersize=6, label="True value")

    # Legend handled at the figure level to avoid per-subplot duplication.

# Function to plot MSE curves for each Delta and param (including beta)
def plot_mse_curves(ax, T_list, mse_eta, mse_mu, mse_beta, Delta, param):
    ax.loglog(T_list, mse_eta, "o-", label="η")
    ax.loglog(T_list, mse_mu, "^-", label="μ")
    ax.loglog(T_list, mse_beta, "s-", label="β")

    T_ref = np.array(T_list, dtype=float)
    ax.loglog(T_ref, 3.0 * T_ref**(-1), "--", color="gray", label="Slope = -1")

    ax.set_title(f"Δ={Delta}")
    ax.set_ylabel(f"MSE, β={param[2]}")  # Use the value of beta from param
    ax.set_xlabel("T")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

# MSE Simulation and plotting
def run_simulation_mse(
    true_param,
    T_list,
    Delta_list,
    R,
    K_alias,
    maxiter=100,
    ftol=1e-5,
    time_every=0,
    print_summary=True,
):
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(Delta_list),
        figsize=(15, 4),  # Adjust size as necessary
        sharey=True
    )
    axes = np.atleast_1d(axes).ravel()  # Normalize to 1D array of Axes

    mse_results = []
    summary_lines = []
    summary = {}

    for iD, Delta in enumerate(Delta_list):
        ax = axes[iD]  # Get the correct ax for each Delta

        # Simulate and estimate the parameters
        mse_eta, mse_mu, mse_beta = simulate_and_estimate(
            true_param,
            T_list,
            Delta,
            R,
            K_alias,
            maxiter=maxiter,
            ftol=ftol,
            time_every=time_every,
        )
        mse_results.append((mse_eta, mse_mu, mse_beta))
        summary[Delta] = {
            "mse_eta": np.asarray(mse_eta),
            "mse_mu": np.asarray(mse_mu),
            "mse_beta": np.asarray(mse_beta),
        }

        # Plot the MSE curves
        plot_mse_curves(ax, T_list, mse_eta, mse_mu, mse_beta, Delta, true_param)

        est_matrix = np.column_stack([mse_eta, mse_mu, mse_beta])
        mean_vals = np.nanmean(est_matrix, axis=0)
        std_vals = np.nanstd(est_matrix, axis=0)
        mean_str = np.array2string(mean_vals, precision=8, separator=" ")
        std_str = np.array2string(std_vals, precision=8, separator=" ")
        summary_lines.append(
            f"Delta={Delta}, beta={true_param[2]} | mean={mean_str} | std={std_str}"
        )

    # Add legends and finalize plot
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=4, frameon=False)

    # Figure title and layout
    fig.suptitle(
        "Whittle (Exp kernel) MSE vs T | cols = Δ",
        y=1.03,
        fontsize=12
    )

    # Align horizontal scale across subplots (log scale requires positive limits)
    all_vals = np.concatenate([np.ravel(v) for trio in mse_results for v in trio])
    all_vals = all_vals[np.isfinite(all_vals) & (all_vals > 0)]
    if all_vals.size:
        ymin = np.min(all_vals)
        ymax = np.percentile(all_vals, 90) * 1.2
        for ax in axes:
            ax.set_ylim(ymin, ymax)

    fig.tight_layout()
    plt.show()

    if print_summary:
        print("\n========== Whittle Exp | MSE vs T ==========")
        for line in summary_lines:
            print(line)

    return summary

# Boxplot Simulation and plotting
def run_simulation_boxplot(
    true_param,
    T_list,
    Delta_list,
    R,
    K_alias,
    labels=None,
    maxiter=100,
    ftol=1e-5,
    time_every=0,
    print_summary=True,
):
    if labels is None:
        labels = ["eta", "mu", "beta"]

    fig, axes = plt.subplots(
        nrows=len(T_list),
        ncols=len(Delta_list),
        figsize=(12, 8),  # Adjust size as necessary
        sharey=True
    )
    axes = np.array(axes, dtype=object)
    if axes.ndim == 0:
        axes = np.array([[axes.item()]])
    elif axes.ndim == 1:
        axes = axes.reshape(len(T_list), len(Delta_list))

    summary = {}
    summary_lines = []

    for iT, T in enumerate(T_list):
        for iD, Delta in enumerate(Delta_list):
            ax = axes[iT, iD]

            eta_est, mu_est, beta_est = simulate_and_collect_estimates(
                true_param,
                T,
                Delta,
                R,
                K_alias,
                maxiter=maxiter,
                ftol=ftol,
                time_every=time_every,
            )

            # Collect estimates for boxplot
            if eta_est.size == 0 or mu_est.size == 0 or beta_est.size == 0:
                estimates = [np.array([np.nan]), np.array([np.nan]), np.array([np.nan])]
            else:
                estimates = [eta_est, mu_est, beta_est]

            # Plot the boxplot
            plot_boxplot_results(ax, estimates, labels, true_param)
            ax.set_ylim(-0.25, 4.25)

            if iD == 0:
                ax.set_ylabel(f"T={T}")

            est_matrix = np.column_stack(estimates)
            mean_vals = np.nanmean(est_matrix, axis=0)
            std_vals = np.nanstd(est_matrix, axis=0)
            summary[(T, Delta)] = (mean_vals, std_vals)
            mean_str = np.array2string(mean_vals, precision=8, separator=" ")
            std_str = np.array2string(std_vals, precision=8, separator=" ")
            summary_lines.append(
                f"T={T}, Delta={Delta} | mean={mean_str} | std={std_str}"
            )

    # Add legends and finalize plot
    legend_elements = [
        Line2D([0], [0], color="orange", lw=2, label="Median (boxplot)"),
        Line2D([0], [0], marker="D", color="tab:blue", linestyle="None", markersize=6, label="Mean"),
        Line2D([0], [0], marker="o", color="red", linestyle="None", markersize=6, label="True value"),
    ]
    legend_labels = [str(h.get_label()) for h in legend_elements]
    fig.legend(legend_elements, legend_labels, loc="upper right", ncol=4, frameon=False, fontsize=9)

    # Figure title and layout
    fig.suptitle(
        "Whittle (EXP Kernel) Boxplot",
        y=1.03,
        fontsize=12
    )

    fig.tight_layout()
    plt.show()

    if print_summary:
        print("\n========== Whittle Exp | Boxplot Summary ==========")
        for line in summary_lines:
            print(line)

    return summary

# eta_true = 1.0
# mu_true = 0.5
# T_list = [100, 150, 230, 350, 550, 800]
# Delta_list = [2.0, 1.0, 0.5, 0.25]
# beta_list = [1.0, 0.5]   # rows
# K_alias = 3
# R = 10

# # Run the simulation
# true_param1= [eta_true, mu_true, beta_list[0]]
# true_param2 = [eta_true, mu_true, beta_list[1]]

# run_simulation_mse(true_param1, T_list, Delta_list, R, K_alias)
# run_simulation_mse(true_param2, T_list, Delta_list, R, K_alias)
