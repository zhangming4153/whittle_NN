import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.lines import Line2D
from data_gener import Branching, ogata_power_hawkes
from mle_opt import fit_hawkes_lbfgsb, fit_power_hawkes_A, estimate_power_hawkes

def generate_boxplot(
    T,
    beta_true,
    R,
    eta_true,
    mu_true,
    labels,
    ax,
    maxiter=500,
    ftol=1e-5,
    time_every=100,
):
    estimates = []
    block_elapsed = 0.0
    for r in range(R):
        events = Branching(
            eta=eta_true,
            mu=mu_true,
            beta=beta_true,
            T=T,
            burn_in=100,
            seed=2025 + r + 100_000 * T
        )
        start_time = time.perf_counter()
        res = fit_hawkes_lbfgsb(events, T=T, maxiter=maxiter, ftol=ftol)
        block_elapsed += time.perf_counter() - start_time
        estimates.append(res.x)

        if time_every and (r + 1) % time_every == 0:
            print(
                f"T={T}, beta={beta_true}: "
                f"Block({time_every}) Time = {block_elapsed:.4f}s"
            )
            block_elapsed = 0.0

    estimates = np.asarray(estimates)  # (R, 3)
    try:
        ax.boxplot(
            estimates,
            tick_labels=labels,
            flierprops=dict(
                marker='o',
                markerfacecolor='black',
                markeredgecolor='black',
                markersize=2
            )
        )
    except TypeError:
        ax.boxplot(
            estimates,
            labels=labels,
            flierprops=dict(
                marker='o',
                markerfacecolor='black',
                markeredgecolor='black',
                markersize=2
            )
        )

    positions = np.arange(1, estimates.shape[1] + 1)
    mean_est = estimates.mean(axis=0)
    ax.scatter(positions, mean_est, s=20, color='tab:blue', marker='D', zorder=5, label="Mean")

    # True value dots
    true_values = [eta_true, mu_true, beta_true]
    ax.scatter(positions, true_values, s=20, color='red', zorder=6, label="True value")

    ax.set_title(f"T={T}, beta={beta_true}, Rep={R}")
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_ylim(-0.25, 4.25)

    return estimates

def plot_mse(
        T_list, 
        mse_data, 
        axes):
    
    for iB, entry in enumerate(mse_data):
        ax = axes[0, iB]
        beta_true = entry["beta"]
        mse_eta = entry["mse_eta"]
        mse_mu = entry["mse_mu"]
        mse_beta = entry["mse_beta"]

        # ---- plot for this beta ----
        ax.loglog(T_list, mse_eta, '--', label='η')
        ax.loglog(T_list, mse_mu, '--', label='μ')
        ax.loglog(T_list, mse_beta, '--', label='β')

        T_ref = np.array(T_list, dtype=float)
        ax.loglog(T_ref, T_ref**(-1), '--', color='gray', label='Slope = -1')

        ax.set_title(f"β={beta_true}")
        ax.set_xlabel("T")
        ax.set_xticks([100, 300, 1000])
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
        if iB == 0:
            ax.set_ylabel("MSE")
    return axes

def mle_exp_mse(
    T_list,
    beta_list,
    eta_true,
    mu_true,
    R,
    maxiter=500,
    ftol=1e-5,
    time_every=100,
    save_path=None,
    print_summary=True,
):
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(beta_list),
        figsize=(4.5 * len(beta_list), 4 * 1),
        sharey=True
    )

    axes = np.atleast_2d(axes)

    mse_data = []

    for beta_true in beta_list:
        true_param = np.array([eta_true, mu_true, beta_true])
        mse_eta = []
        mse_mu = []
        mse_beta = []

        for T in T_list:
            estimates = []
            block_elapsed = 0.0

            for r in range(R):
                events = Branching(
                    eta=eta_true,
                    mu=mu_true,
                    beta=beta_true,
                    T=T,
                    burn_in=100,
                    seed=2025 + r + 100_000 * T
                )
                start_time = time.perf_counter()
                res = fit_hawkes_lbfgsb(events, T=T, maxiter=maxiter, ftol=ftol)
                block_elapsed += time.perf_counter() - start_time
                estimates.append(res.x)

                if time_every and (r + 1) % time_every == 0:
                    print(
                        f"T={T}, beta={beta_true}: "
                        f"Block({time_every}) Time = {block_elapsed:.4f}s"
                    )
                    block_elapsed = 0.0

            estimates = np.asarray(estimates)  # (R, 3)
            mse = np.mean((estimates - true_param) ** 2, axis=0)

            mse_eta.append(mse[0])
            mse_mu.append(mse[1])
            mse_beta.append(mse[2])

        mse_data.append({
            "beta": beta_true,
            "mse_eta": np.asarray(mse_eta),
            "mse_mu": np.asarray(mse_mu),
            "mse_beta": np.asarray(mse_beta),
        })
    axes = plot_mse(T_list, mse_data, axes)

    fig.subplots_adjust(top=0.85)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle("Exponential MLE | MSE vs T", y=1.10, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    # Show the plot
    plt.show()

    if print_summary:
        print("\n========== Exp MLE | MSE vs T ==========")
        for entry in mse_data:
            beta_true = entry["beta"]
            for T, mse_eta, mse_mu, mse_beta in zip(T_list, entry["mse_eta"], entry["mse_mu"], entry["mse_beta"]):
                print(
                    f"beta={beta_true}, T={T} | "
                    f"mse_eta={mse_eta:.6g} | mse_mu={mse_mu:.6g} | mse_beta={mse_beta:.6g}"
                )

    if save_path:
        np.savez(
            save_path,
            T_list=np.asarray(T_list),
            beta_list=np.asarray(beta_list),
            mse_eta=np.stack([entry["mse_eta"] for entry in mse_data]),
            mse_mu=np.stack([entry["mse_mu"] for entry in mse_data]),
            mse_beta=np.stack([entry["mse_beta"] for entry in mse_data]),
        )

    return mse_data

def mle_exp_boxplot(
    T_list,
    beta_list,
    eta_true,
    mu_true,
    R,
    labels=None,
    maxiter=500,
    ftol=1e-5,
    time_every=100,
    print_summary=True,
):
    if labels is None:
        labels = ["eta", "mu", "beta"]

    fig, axes = plt.subplots(nrows=len(T_list), ncols=len(beta_list),
                             figsize=(5*len(beta_list), 8), sharey=True)
    axes = np.array(axes, dtype=object)
    if axes.ndim == 0:
        axes = np.array([[axes.item()]])
    elif axes.ndim == 1:
        axes = axes.reshape(len(T_list), len(beta_list))

    summary = {}
    for i, T in enumerate(T_list):
        for j, beta_true in enumerate(beta_list):
            ax = axes[i, j]
            estimates = generate_boxplot(
                T=T,
                beta_true=beta_true,
                R=R,
                eta_true=eta_true,
                mu_true=mu_true,
                labels=labels,
                ax=ax,
                maxiter=maxiter,
                ftol=ftol,
                time_every=time_every,
            )
            summary[(T, beta_true)] = (estimates.mean(axis=0), estimates.std(axis=0))
    
    fig.suptitle("Exponential MLE", y=1.02, fontsize=14)

    # One legend for the whole figure
    legend_elements = [
        Line2D([0], [0], color='orange', lw=2, label='Median'),
        Line2D([0], [0], marker='D', color='tab:blue', linestyle='None', markersize=6, label='Mean'),
        Line2D([0], [0], marker='o', color='red', linestyle='None', markersize=6, label='True value'),
    ]
    legend_labels = [str(h.get_label()) for h in legend_elements]
    fig.legend(legend_elements, legend_labels, loc="upper right", frameon=False, fontsize=9)

    fig.tight_layout()
    plt.show()

    if print_summary:
        print("\n========== BOXPLOT SUMMARY ==========")
        for (T, beta_true), (m, s) in summary.items():
            print(f"T={T}, beta={beta_true} | mean={m} | std={s}")

    return summary

def mle_power_boxplot(
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
    return_records=True,
    verbose=True,
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
    records = [] if return_records else None

    for iT, T in enumerate(T_list):
        for iP, gamma_true in enumerate(gamma_list):
            ax = axes[iT, iP]
            true_param = np.array([eta_true, mu_true, gamma_true])

            estimates = []
            fail = 0

            if verbose:
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
                if verbose and time_every and (r + 1) % time_every == 0:
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
            median_est = np.median(estimates, axis=0)
            mse_est = np.mean((estimates - true_param) ** 2, axis=0)
            std_est = estimates.std(axis=0, ddof=1) if estimates.shape[0] > 1 else np.full(3, np.nan)

            ax.scatter(np.arange(1, 4), mean_est, color="tab:blue", marker="D", s=18, zorder=4)

            # true values
            ax.scatter(np.arange(1, 4), true_param, color="red", s=22, zorder=5)

            ax.set_title(f"T={T}, gamma={gamma_true}, Rep={len(estimates)}", fontsize = 10)
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.set_ylim(-0.25, 4.25)

            # Append time information to the summary
            mean_str = np.array2string(mean_est, precision=8, separator=" ")
            median_str = np.array2string(median_est, precision=8, separator=" ")
            mse_str = np.array2string(mse_est, precision=8, separator=" ")
            std_str = np.array2string(std_est, precision=8, separator=" ")
            summary_lines.append(
                f"T={T}, gamma={gamma_true} | mean={mean_str} | median={median_str} | "
                f"mse={mse_str} | std={std_str} | Rep={len(estimates)} | fail={fail}"
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

def mle_power_mse(
    T_list,
    gamma_list,
    eta_true,
    mu_true,
    a_true,
    R,
    burn_in=100,
    labels=None,
    maxiter=60,
    ftol=1e-4,
    time_every=100,
    gamma_grid=(0.6, 1.0, 1.7, 2.9, 5.0),
    return_records=True,
    verbose=True,
    print_summary=True,
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

    summary_lines = []
    records = []

    # Loop through gamma values
    for iA, gamma_true in enumerate(gamma_list):
        true_param = np.array([eta_true, mu_true, gamma_true], float)
        ax = axes[0, iA]
        mse_eta, mse_mu, mse_gamma = [], [], []

        for T in T_list:
            estimates = []
            fail = 0
            if verbose:
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
                    gamma_grid=gamma_grid,
                    verbose=False
                )
                elapsed_time = time.perf_counter() - start_time
                block_elapsed += elapsed_time

                success = (res is not None) and getattr(res, "success", False) and np.all(np.isfinite(res.x))
                if return_records:
                    records.append({
                        "T": T, "gamma": gamma_true, "rep": r,
                        "success": success, "time": elapsed_time,
                    })

                if not success:
                    fail += 1
                    continue

                estimates.append(res.x)

                if verbose and time_every and (r + 1) % time_every == 0:
                    print(
                        f"T={T}, gamma={gamma_true}: "
                        f"Block({time_every}) Time = {block_elapsed:.4f}s"
                    )
                    block_elapsed = 0.0

            estimates = np.asarray(estimates, float)

            if estimates.size == 0:
                mse_eta.append(np.nan)
                mse_mu.append(np.nan)
                mse_gamma.append(np.nan)
                if verbose:
                    print(f"[WARN] all fits failed at T={T}, gamma={gamma_true}")
                summary_lines.append(
                    f"T={T}, gamma={gamma_true} | mse=[nan nan nan] | Rep=0 | fail={fail}"
                )
                continue

            mse = np.mean((estimates - true_param) ** 2, axis=0)
            mse_eta.append(mse[0])
            mse_mu.append(mse[1])
            mse_gamma.append(mse[2])

            if verbose and fail > 0:
                print(f"T={T}, gamma={gamma_true}: kept {len(estimates)}/{R} (fail={fail})")
            summary_lines.append(
                f"T={T}, gamma={gamma_true} | "
                f"mse_eta={mse[0]:.6g} | mse_mu={mse[1]:.6g} | mse_gamma={mse[2]:.6g}"
            )

        # Plot for this gamma value
        ax.loglog(T_list, mse_eta, '--', label='η')
        ax.loglog(T_list, mse_mu, '--', label='μ')
        ax.loglog(T_list, mse_gamma, '--', label='γ')

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

    if print_summary:
        print("\n========== Powerlaw MLE | MSE vs T ==========")
        for line in summary_lines:
            print(line)

    if return_records:
        return summary_lines, records
    return summary_lines
