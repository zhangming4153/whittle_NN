import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.lines import Line2D
from data_gener import Branching
from mle_opt import fit_hawkes_lbfgsb

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

def create_boxplots(
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
                             figsize=(14, 8), sharey=True)
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

def compute_mse_data(
    T_list,
    beta_list,
    eta_true,
    mu_true,
    R,
    maxiter=500,
    ftol=1e-5,
    time_every=100,
):
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

    return mse_data


def plot_mse(T_list, mse_data, axes):
    for iB, entry in enumerate(mse_data):
        ax = axes[0, iB]
        beta_true = entry["beta"]
        mse_eta = entry["mse_eta"]
        mse_mu = entry["mse_mu"]
        mse_beta = entry["mse_beta"]

        # ---- plot for this beta ----
        ax.loglog(T_list, mse_eta, 'o-', label='η')
        ax.loglog(T_list, mse_mu, '^-', label='μ')
        ax.loglog(T_list, mse_beta, 's-', label='β')

        T_ref = np.array(T_list, dtype=float)
        ax.loglog(T_ref, T_ref**(-1), '--', color='gray', label='Slope = -1')

        ax.set_title(f"β={beta_true}")
        ax.set_xlabel("T")
        ax.set_xticks([100, 300, 1000])
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
        if iB == 0:
            ax.set_ylabel("MSE")
    return axes

def create_mse_plot(
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

    mse_data = compute_mse_data(
        T_list,
        beta_list,
        eta_true,
        mu_true,
        R,
        maxiter=maxiter,
        ftol=ftol,
        time_every=time_every,
    )
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
            print(
                f"beta={entry['beta']} | "
                f"mse_eta={entry['mse_eta']} | "
                f"mse_mu={entry['mse_mu']} | "
                f"mse_beta={entry['mse_beta']}"
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
