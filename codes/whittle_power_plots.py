import numpy as np
import matplotlib.pyplot as plt
import time
from types import SimpleNamespace
from matplotlib.lines import Line2D
from data_gener import ogata_power_hawkes, binsized
from whittle_opt import whittle_powerlaw_fast


def simulate_and_estimate_powerlaw(
    T_list,
    Delta,
    true_param,
    R,
    K_alias,
    a_true,
    maxiter,
    s=5.0,
    k0=200,
    step_hi=5,
    ftol=1e-5,
    eps=1e-3,
    retry_on_fail=True,
    retry_maxiter=80,
    retry_ftol=1e-6,
    time_every=0,
):
    mse_eta, mse_mu, mse_gamma = [], [], []

    for T in T_list:
        estimates = []

        block_elapsed = 0.0
        for r in range(R):
            seed = 2025 + r + 100_000 * int(Delta * 1000) + 1_000_000 * T

            events = ogata_power_hawkes(
                T=T,
                eta=true_param[0],
                mu=true_param[1],
                gamma=true_param[2],
                a=a_true,
                burn_in=100.0,
                seed=seed,
            )

            bindata = binsized(events, Delta, T)

            start_time = time.perf_counter()
            try:
                par_hat, res = whittle_powerlaw_fast(
                    bindata,
                    Delta,
                    K_alias=K_alias,
                    a=a_true,
                    maxiter=maxiter,
                    s=s,
                    k0=k0,
                    step_hi=step_hi,
                    ftol=ftol,
                    eps=eps,
                )
            except ValueError:
                par_hat = np.array([np.nan, np.nan, np.nan], dtype=float)
                res = SimpleNamespace(success=False)
            block_elapsed += time.perf_counter() - start_time

            if retry_on_fail and ((not res.success) or (not np.all(np.isfinite(par_hat)))):
                try:
                    par_hat, res = whittle_powerlaw_fast(
                        bindata,
                        Delta,
                        K_alias=K_alias,
                        a=a_true,
                        maxiter=retry_maxiter,
                        s=s,
                        k0=k0,
                        step_hi=step_hi,
                        ftol=retry_ftol,
                        eps=eps,
                    )
                except ValueError:
                    par_hat = np.array([np.nan, np.nan, np.nan], dtype=float)
                    res = SimpleNamespace(success=False)
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
            mse_eta.append(np.nan)
            mse_mu.append(np.nan)
            mse_gamma.append(np.nan)
        else:
            mse = np.mean((estimates - true_param) ** 2, axis=0)
            mse_eta.append(mse[0])
            mse_mu.append(mse[1])
            mse_gamma.append(mse[2])

    return mse_eta, mse_mu, mse_gamma


def plot_mse_curves(ax, T_list, mse_eta, mse_mu, mse_gamma, Delta, gamma_true, show_xlabel):
    ax.loglog(T_list, mse_eta, "o-", label="η")
    ax.loglog(T_list, mse_mu, "^-", label="μ")
    ax.loglog(T_list, mse_gamma, "s-", label="γ")

    T_ref = np.array(T_list, dtype=float)
    ax.loglog(T_ref, T_ref ** (-1), "--", color="gray", label="Slope = -1")

    ax.set_title(f"Δ={Delta}\n")
    ax.set_ylabel(f"MSE, γ={gamma_true}")
    if show_xlabel:
        ax.set_xlabel("T")

    ax.grid(True, which="both", linestyle="--", alpha=0.5)


def run_powerlaw_mse_grid(
    T_list,
    Delta_list,
    gamma_true_list,
    R,
    K_alias,
    eta_true,
    mu_true,
    a_true,
    maxiter,
    s=5.0,
    k0=200,
    step_hi=5,
    ftol=1e-5,
    eps=1e-3,
    retry_on_fail=True,
    retry_maxiter=80,
    retry_ftol=1e-6,
    time_every=0,
    print_summary=True,
):
    fig, axes = plt.subplots(
        nrows=len(gamma_true_list),
        ncols=len(Delta_list),
        figsize=(4.5 * len(Delta_list), 4 * len(gamma_true_list)),
        sharey=True,
    )
    axes = np.atleast_2d(axes)
    axes = axes.reshape(len(gamma_true_list), len(Delta_list))

    summary = {}
    summary_lines = []

    for iG, gamma_true in enumerate(gamma_true_list):
        true_param = np.array([eta_true, mu_true, gamma_true], dtype=float)

        for iD, Delta in enumerate(Delta_list):
            ax = axes[iG, iD]
            mse_eta, mse_mu, mse_gamma = simulate_and_estimate_powerlaw(
                T_list,
                Delta,
                true_param,
                R,
                K_alias,
                a_true,
                maxiter,
                s=s,
                k0=k0,
                step_hi=step_hi,
                ftol=ftol,
                eps=eps,
                retry_on_fail=retry_on_fail,
                retry_maxiter=retry_maxiter,
                retry_ftol=retry_ftol,
                time_every=time_every,
            )
            summary[(gamma_true, Delta)] = {
                "mse_eta": np.asarray(mse_eta),
                "mse_mu": np.asarray(mse_mu),
                "mse_gamma": np.asarray(mse_gamma),
            }
            mean_vals = np.nanmean(
                np.column_stack([mse_eta, mse_mu, mse_gamma]),
                axis=0,
            )
            std_vals = np.nanstd(
                np.column_stack([mse_eta, mse_mu, mse_gamma]),
                axis=0,
            )
            mean_str = np.array2string(mean_vals, precision=8, separator=" ")
            std_str = np.array2string(std_vals, precision=8, separator=" ")
            summary_lines.append(
                f"Gamma={gamma_true}, Delta={Delta} | mean={mean_str} | std={std_str}"
            )
            plot_mse_curves(
                ax,
                T_list,
                mse_eta,
                mse_mu,
                mse_gamma,
                Delta,
                gamma_true,
                show_xlabel=(iG == len(gamma_true_list) - 1),
            )

    fig.subplots_adjust(top=0.85)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=4, frameon=False)

    fig.suptitle("Whittle (Power-law) MSE vs T | rows=γ, cols=Δ", y=1.03, fontsize=12)
    fig.tight_layout()
    plt.show()

    if print_summary:
        print("\n========== Whittle Power | MSE vs T ==========")
        for line in summary_lines:
            print(line)

    return summary



def _plot_boxplot(ax, estimates, labels):
    try:
        ax.boxplot(
            estimates,
            tick_labels=labels,
            showfliers=True,
            flierprops=dict(
                marker="o",
                markerfacecolor="black",
                markeredgecolor="black",
                markersize=3,
            ),
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
                markersize=3,
            ),
        )


def run_powerlaw_boxplot_grid(
    T_list,
    Delta_list,
    true_param,
    R,
    K_alias,
    a_true,
    maxiter,
    k0=200,
    step_hi=5,
    ftol=1e-5,
    eps=1e-3,
    retry_on_fail=True,
    retry_maxiter=80,
    retry_ftol=1e-6,
    time_every=0,
    print_summary=True,
    return_records=False,
):
    labels = ["eta", "mu", "gamma"]

    fig, axes = plt.subplots(
        nrows=len(T_list),
        ncols=len(Delta_list),
        figsize=(4.2 * len(Delta_list), 3.4 * len(T_list)),
        sharey=True,
    )
    axes = np.atleast_2d(axes)

    summary_lines = []
    records = [] if return_records else None

    for iT, T in enumerate(T_list):
        for iD, Delta in enumerate(Delta_list):
            ax = axes[iT, iD]
            estimates = []
            fail = 0

            block_elapsed = 0.0
            for r in range(R):
                seed = 2025 + r + 10_000 * iT + 100 * iD

                events = ogata_power_hawkes(
                    T=T,
                    eta=true_param[0],
                    mu=true_param[1],
                    gamma=true_param[2],
                    a=a_true,
                    burn_in=100.0,
                    seed=seed,
                )

                bindata = binsized(events, Delta, T)

              
               
                start_time = time.perf_counter()
                try:
                    par_hat, res = whittle_powerlaw_fast(
                        bindata,
                        Delta,
                        K_alias=K_alias,
                        a=a_true,
                        maxiter=maxiter,
                        k0=k0,
                        step_hi=step_hi,
                        ftol=ftol,
                        eps=eps,
                    )
                except ValueError:
                    par_hat = np.array([np.nan, np.nan, np.nan], dtype=float)
                    res = SimpleNamespace(success=False)
                block_elapsed += time.perf_counter() - start_time

                if retry_on_fail and ((not res.success) or (not np.all(np.isfinite(par_hat)))):
                    try:
                        par_hat, res = whittle_powerlaw_fast(
                            bindata,
                            Delta,
                            K_alias=K_alias,
                            a=a_true,
                            maxiter=retry_maxiter,
                            k0=k0,
                            step_hi=step_hi,
                            ftol=retry_ftol,
                            eps=eps,
                        )
                    except ValueError:
                        par_hat = np.array([np.nan, np.nan, np.nan], dtype=float)
                        res = SimpleNamespace(success=False)
                
                if time_every and (r + 1) % time_every == 0:
                    print(
                        f"T={T}, Delta={Delta}: "
                        f"Block({time_every}) Time = {block_elapsed:.4f}s"
                    )
                    block_elapsed = 0.0

                ok = (res.success and np.all(np.isfinite(par_hat)))
                if not ok:
                    fail += 1
                    if return_records:
                        records.append(
                            {
                                "T": T,
                                "Delta": Delta,
                                "rep": r,
                                "success": False,
                                "nfev": getattr(res, "nfev", None),
                                "nit": getattr(res, "nit", None),
                                "fun": getattr(res, "fun", None),
                            }
                        )
                    continue

                estimates.append(par_hat)
                if return_records:
                    records.append(
                        {
                            "T": T,
                            "Delta": Delta,
                            "rep": r,
                            "eta_hat": par_hat[0],
                            "mu_hat": par_hat[1],
                            "gamma_hat": par_hat[2],
                            "success": True,
                            "nfev": res.nfev,
                            "nit": res.nit,
                            "fun": res.fun,
                        }
                    )

            estimates = np.asarray(estimates)

            if estimates.size == 0:
                ax.text(
                    0.5,
                    0.5,
                    "all failed",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(f"T={T}, Δ={Delta}\nRep=0, fail={fail}")
                ax.set_xticks([1, 2, 3])
                ax.set_xticklabels(labels)
                summary_lines.append(f"T={T}, Δ={Delta} | Rep=0 | fail={fail}")
                continue

            _plot_boxplot(ax, estimates, labels)

            mean_est = estimates.mean(axis=0)
            std_est = (
                estimates.std(axis=0, ddof=1)
                if estimates.shape[0] > 1
                else np.full(3, np.nan)
            )

            ax.scatter(
                np.arange(1, 4),
                mean_est,
                color="tab:blue",
                marker="D",
                s=18,
                zorder=4,
            )
            ax.scatter(np.arange(1, 4), true_param, color="red", s=22, zorder=5)

            ax.set_title(f"T={T}, Δ={Delta}\nRep={len(estimates)}, fail={fail}")
            ax.grid(axis="y", linestyle="--", alpha=0.4)

            summary_lines.append(
                f"T={T}, Δ={Delta} | mean={mean_est} | std={std_est} | Rep={len(estimates)} | fail={fail}"
            )

    fig.suptitle(
        "Whittle Estimator (Power-law Hawkes)\nDistribution across (T, Δ)",
        fontsize=13,
        y=1.02,
    )

    for ax in axes[-1, :]:
        ax.set_xlabel("Parameter")

    legend_elements = [
        Line2D([0], [0], color="orange", lw=2, label="Median (boxplot)"),
        Line2D([0], [0], marker="D", color="tab:blue", linestyle="None", markersize=6, label="Mean"),
        Line2D([0], [0], marker="o", color="red", linestyle="None", markersize=6, label="True value"),
    ]
    fig.legend(handles=legend_elements, loc="upper right", frameon=False)

    fig.tight_layout()
    plt.show()

    if print_summary:
        print("\n========== SUMMARY ==========")
        for line in summary_lines:
            print(line)

    if return_records:
        return summary_lines, records
    return summary_lines
