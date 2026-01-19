import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.lines import Line2D
from data_gener import ogata_power_hawkes
from mle_opt import fit_power_hawkes_A, estimate_power_hawkes
import time

def plot_estimates(
    T_list,
    gamma_list,
    eta_true,
    mu_true,
    a_true,
    R,

    burn_in=100,
    maxiter=80,
    ftol=1e-5,
    seed_base=2025,
    time_every=100,
    return_records=False,
):
    """
    This function generates boxplots for parameter estimates based on a unified estimator.
    """
    labels = ["eta", "mu", "gamma"]

    fig, axes = plt.subplots(
        nrows=len(T_list),
        ncols=len(gamma_list),
        figsize=(4.2 * len(gamma_list), 3.4 * len(T_list)),
        sharey=True
    )

    axes = np.atleast_2d(axes)
    axes = axes.reshape(len(T_list), len(gamma_list))

    summary_lines = []
    records = []

    for iT, T in enumerate(T_list):
        for iP, gamma_true in enumerate(gamma_list):
            ax = axes[iT, iP]
            true_param = np.array([eta_true, mu_true, gamma_true])

            estimates = []
            fail = 0

            print(f"\nRunning T={T}, gamma={gamma_true} ...")

            block_elapsed = 0.0
            for r in range(R):
                seed = seed_base + r + 10_000 * iT + 100 * iP

                # Generate events using ogata_power_hawkes
                events = ogata_power_hawkes(
                    T=T,
                    eta=eta_true,
                    mu=mu_true,
                    gamma=gamma_true,  # parameter that varies in each loop
                    a=a_true,
                    burn_in=burn_in,
                    seed=seed
                )

                start_time = time.time()
                estimates_r, success = estimate_power_hawkes(
                    events,
                    T=T,
                    a=a_true,
                    maxiter=maxiter,
                    ftol=ftol,
                )
                elapsed_time = time.time() - start_time
                block_elapsed += elapsed_time
                if time_every and (r + 1) % time_every == 0:
                    print(
                        f"Repetition {r + 1}: "
                        f"Block({time_every}) Time = {block_elapsed:.4f}s"
                    )
                    block_elapsed = 0.0

                if not success:
                    fail += 1
                    continue

                estimates.append(estimates_r)

                # Record the results along with the elapsed time
                if return_records:
                    records.append({
                        "T": T, "gamma": gamma_true, "rep": r,
                        "success": success, "time": elapsed_time,
                    })

            estimates = np.asarray(estimates)

            # --- Plot results for this (T, param) ---
            if estimates.size == 0:
                ax.text(0.5, 0.5, "all failed", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"T={T}, gamma={gamma_true}, Rep={R}", fontsize = 10)
                ax.set_xticks([1, 2, 3])
                ax.set_xticklabels(labels)
                summary_lines.append(f"T={T}, gamma={gamma_true} | Rep=0 | fail={fail}")
                continue

            try:
                ax.boxplot(
                    estimates,
                    tick_labels=labels,
                    showfliers=True,
                    flierprops=dict(
                        marker="o",
                        markerfacecolor="black",
                        markeredgecolor="black",
                        markersize=3
                    )
                )
            except TypeError:
                ax.boxplot(
                    estimates,
                    labels=labels,
                    showfliers=True,
                    flierprops=dict(
                        marker="o",
                        markerfacecolor="black",
                        markeredgecolor="black",
                        markersize=3
                    )
                )

            mean_est = estimates.mean(axis=0)
            std_est = estimates.std(axis=0, ddof=1) if estimates.shape[0] > 1 else np.full(3, np.nan)

            ax.scatter(np.arange(1, 4), mean_est, color="tab:blue", marker="D", s=18, zorder=4)

            # true values
            ax.scatter(np.arange(1, 4), true_param, color="red", s=22, zorder=5)

            ax.set_title(f"T={T}, gamma={gamma_true}, Rep={len(estimates)}", fontsize = 10)
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.set_ylim(0, 4)

            # Append time information to the summary
            summary_lines.append(
                f"T={T}, gamma={gamma_true} | mean={mean_est} | std={std_est} | Rep={len(estimates)} | fail={fail}"
            )

    fig.suptitle(f"Powerlaw MLE", fontsize=10, y=1.02)

    for ax in axes[-1, :]:
        ax.set_xlabel("Gamma")

    legend_elements = [
        Line2D([0], [0], color="orange", lw=2, label="Median"),
        Line2D([0], [0], marker="D", color="tab:blue", linestyle="None", markersize=6, label="Mean"),
        Line2D([0], [0], marker="o", color="red", linestyle="None", markersize=6, label="True value"),
    ]
    fig.legend(handles=legend_elements, loc="upper right", frameon=False, fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()

    if return_records:
        return summary_lines, records
    return summary_lines


def plot_mse_vs_T(
    T_list,
    gamma_list,
    eta_true,
    mu_true,
    a_true,
    burn_in,
    R,
    labels=None,
    maxiter=60,
    ftol=1e-4,
    time_every=100,
    gamma_grid=(0.6, 1.0, 1.7, 2.9, 5.0),
):
    """
    This function generates MSE vs T plots for different gamma values and displays them using boxplots.
    """
    if labels is None:
        labels = ["eta", "mu", "gamma"]

    # Create the subplots
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(gamma_list),
        figsize=(4.5 * len(gamma_list), 4),
        sharey=True
    )

    axes = np.atleast_2d(axes)

    # Loop through gamma values
    for iA, gamma_true in enumerate(gamma_list):
        true_param = np.array([eta_true, mu_true, gamma_true], float)
        ax = axes[0, iA]
        mse_eta, mse_mu, mse_gamma = [], [], []
        fail_counts = []

        for T in T_list:
            estimates = []
            fail = 0
            print(f"Starting T={T}, gamma={gamma_true}...")

            block_elapsed = 0.0
            for r in range(R):
                seed = 2025 + 10_000 * int(T) + 100 * int(10 * gamma_true) + r

                events = ogata_power_hawkes(
                    float(T), float(eta_true), float(mu_true), float(gamma_true),
                    float(a_true), float(burn_in), int(seed)
                )

                start_time = time.perf_counter()

                # Estimate parameters using MLE
                res = fit_power_hawkes_A(
                    events,
                    T=float(T),
                    a=a_true,
                    maxiter=maxiter,
                    ftol=ftol,
                    mu0=0.5,
                    gamma_grid=gamma_grid,
                    verbose=False
                )
                block_elapsed += time.perf_counter() - start_time

                if (res is None) or (not getattr(res, "success", False)) or (not np.all(np.isfinite(res.x))):
                    fail += 1
                    continue

                estimates.append(res.x)

                if time_every and (r + 1) % time_every == 0:
                    print(
                        f"T={T}, gamma={gamma_true}: "
                        f"Block({time_every}) Time = {block_elapsed:.4f}s"
                    )
                    block_elapsed = 0.0

            estimates = np.asarray(estimates, float)
            fail_counts.append(fail)

            if estimates.size == 0:
                mse_eta.append(np.nan)
                mse_mu.append(np.nan)
                mse_gamma.append(np.nan)
                print(f"[WARN] all fits failed at T={T}, gamma={gamma_true}")
                continue

            mse = np.mean((estimates - true_param) ** 2, axis=0)
            mse_eta.append(mse[0])
            mse_mu.append(mse[1])
            mse_gamma.append(mse[2])

            if fail > 0:
                print(f"T={T}, gamma={gamma_true}: kept {len(estimates)}/{R} (fail={fail})")

        # Plot for this gamma value
        ax.loglog(T_list, mse_eta, 'o-', label='η')
        ax.loglog(T_list, mse_mu, '^-', label='μ')
        ax.loglog(T_list, mse_gamma, 's-', label='γ')

        T_ref = np.array(T_list, dtype=float)
        ax.loglog(T_ref, T_ref ** (-1), '--', color='gray', label='Slope = -1')
        ax.set_title(f"γ={gamma_true}")
        ax.set_xticks([100, 300, 1000])
        ax.get_xaxis().set_major_formatter(ScalarFormatter())

    # Global title and adjustments
    fig.subplots_adjust(top=0.85)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle("Powerlaw MLE | MSE vs T", y=1.10, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()
