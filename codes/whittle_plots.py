import numpy as np
import matplotlib.pyplot as plt
import time
from types import SimpleNamespace
from matplotlib.lines import Line2D
from data_gener import ogata_power_hawkes, binsized, Branching
from whittle_opt import whittle_powerlaw, whittle_hawkes_exponential

def _logit(p):
    return np.log(p / (1.0 - p))

def simulate_and_estimate_exp(true_param, T_list, Delta, R, K_alias, maxiter=100, ftol=1e-5, time_every=0):
    # Instead of collecting MSE, we now collect estimates for each parameter (eta, mu, beta)
    eta_estimates, mu_estimates, beta_estimates = [], [], []

    for T in T_list:
        estimates = []
        fail_T = 0

        block_elapsed = 0.0
        for r in range(R):
            seed = 2025 + r + 100_000 * T

            # --- EXP Hawkes DGP ---
            events = Branching(
                eta=true_param[0],
                mu=true_param[1],
                beta=true_param[2],
                T=T,
                seed=seed
            )

            # bin counts
            bindata = binsized(events, Delta, T)

            # --- Whittle estimate (EXP kernel) ---
            start_time = time.perf_counter()
            par_hat, res = whittle_hawkes_exponential(
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

        # Store estimates for each parameter (eta, mu, beta) per T
        eta_estimates.append([e[0] for e in estimates])
        mu_estimates.append([e[1] for e in estimates])
        beta_estimates.append([e[2] for e in estimates])

    return eta_estimates, mu_estimates, beta_estimates

def plot_boxplot_results(ax, estimates, labels, true_values):
    ax.boxplot(
        estimates,
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
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)

def plot_exp_mse_curves(ax, T_list, eta_estimates, mu_estimates, beta_estimates, Delta, true_param):
    mse_eta, mse_mu, mse_beta = [], [], []
    
    # Calculate MSE for each parameter across all T values
    for iT, T in enumerate(T_list):
        # Compute MSE for eta, mu, beta by calculating squared error
        mse_eta.append(np.mean((np.array(eta_estimates[iT]) - true_param[0]) ** 2))
        mse_mu.append(np.mean((np.array(mu_estimates[iT]) - true_param[1]) ** 2))
        mse_beta.append(np.mean((np.array(beta_estimates[iT]) - true_param[2]) ** 2))

    # Plot the MSE values
    ax.loglog(T_list, mse_eta, "o-", label="η")
    ax.loglog(T_list, mse_mu, "^-", label="μ")
    ax.loglog(T_list, mse_beta, "s-", label="β")

    T_ref = np.array(T_list, dtype=float)
    ax.loglog(T_ref, 3.0 * T_ref**(-1), "--", color="gray", label="Slope = -1")

    ax.set_title(f"Δ={Delta}")
    ax.set_ylabel(f"MSE, β={true_param[2]}")  # Use the value of beta from param
    ax.set_xlabel("T")
    ax.set_ylim(1e-3, 1e1)
    ax.set_xticks([100, 300, 1000])
    ax.set_xticklabels(["100", "300", "1000"])
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    return np.asarray([mse_eta, mse_mu, mse_beta], dtype=float)

def whittle_exp_boxplot(
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

            # Now we get estimates directly (not MSE values)
            eta_est, mu_est, beta_est = simulate_and_estimate_exp(
                true_param,
                [T],  # Ensure T is passed as list for compatibility
                Delta,
                R,
                K_alias,
                maxiter=maxiter,
                ftol=ftol,
                time_every=time_every,
            )

            # Collect estimates for boxplot
            estimates = [eta_est[0], mu_est[0], beta_est[0]]  # List of all estimates

            # Plot the boxplot
            plot_boxplot_results(ax, estimates, labels, true_param)
            ax.set_ylim(-0.25, 4.25)
            ax.set_title(f"T={T}, Δ={Delta}, Rep={R}", fontsize=9)

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

def whittle_exp_mse(
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
        eta_estimates, mu_estimates, beta_estimates = simulate_and_estimate_exp(
            true_param,
            T_list,
            Delta,
            R,
            K_alias,
            maxiter=maxiter,
            ftol=ftol,
            time_every=time_every,
        )
        summary[Delta] = {
            "eta_estimates": np.asarray(eta_estimates),
            "mu_estimates": np.asarray(mu_estimates),
            "beta_estimates": np.asarray(beta_estimates),
        }

        # Plot the MSE curves using the updated estimates
        mse_vals = plot_exp_mse_curves(
            ax, T_list, eta_estimates, mu_estimates, beta_estimates, Delta, true_param
        )
        mse_results.append(mse_vals)

        mean_vals = np.nanmean(mse_vals, axis=1)
        std_vals = np.nanstd(mse_vals, axis=1)
        mean_str = np.array2string(mean_vals, precision=8, separator=" ")
        std_str = np.array2string(std_vals, precision=8, separator=" ")
        summary_lines.append(
            f"Delta={Delta}, beta={true_param[2]} | mse_mean={mean_str} | mse_std={std_str}"
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

def simulate_and_estimate_powerlaw(
    T_list,
    Delta,
    true_param,
    R,
    K_alias,
    a_true,
    maxiter=100,
    s=5.0,
    k0=200,
    step_hi=1,
    ftol=1e-6,
    eps=1e-4,
    retry_on_fail=True,
    retry_ftol=1e-8,
    retry_eps=1e-4,
    retry_attempts=5,
    time_every=100,
):
    eta_estimates, mu_estimates, gamma_estimates = [], [], []

    for T in T_list:
        estimates = []

        block_elapsed = 0.0
        for r in range(R):
            seed = 2025 + r + 100_000 * int(Delta * 1000) + 1_000_000 * T
            rng = np.random.default_rng(seed + 12345)

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
                par_hat, res = whittle_powerlaw(
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

            if retry_on_fail and ((not res.success) or (not np.all(np.isfinite(par_hat)))):
                best = None
                base_theta = res.x if hasattr(res, "x") and np.all(np.isfinite(res.x)) else None
                for _ in range(retry_attempts):
                    if base_theta is None:
                        mu0 = rng.uniform(0.02, 0.98)
                        gamma0 = np.exp(rng.uniform(np.log(0.3), np.log(20.0)))
                        theta0 = np.array([_logit(mu0), np.log(gamma0) / s], float)
                    else:
                        theta0 = np.asarray(base_theta, float) + rng.normal(scale=0.2, size=2)
                    try:
                        cand_hat, cand_res = whittle_powerlaw(
                            bindata,
                            Delta,
                            K_alias=K_alias,
                            a=a_true,
                            maxiter=maxiter,
                            s=s,
                            k0=k0,
                            step_hi=step_hi,
                            ftol=retry_ftol,
                            eps=retry_eps,
                            use_grid_init=False,
                            theta0=theta0,
                        )
                    except ValueError:
                        continue
                    if not np.all(np.isfinite(cand_hat)) or not getattr(cand_res, "success", False):
                        continue
                    if (best is None) or (cand_res.fun < best[1].fun):
                        best = (cand_hat, cand_res)
                if best is not None:
                    par_hat, res = best

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

        # Return individual parameter estimates for boxplot plotting
        eta_estimates.append([e[0] for e in estimates])  # eta estimates for each T
        mu_estimates.append([e[1] for e in estimates])   # mu estimates for each T
        gamma_estimates.append([e[2] for e in estimates])  # gamma estimates for each T

    return eta_estimates, mu_estimates, gamma_estimates

def plot_power_mse_curves(ax, T_list, eta_estimates, mu_estimates, gamma_estimates, Delta, gamma_true, true_param):
    mse_eta, mse_mu, mse_gamma = [], [], []
    
    # Calculate MSE for each parameter across all T values
    for iT, T in enumerate(T_list):
        # Compute MSE for eta, mu, gamma by calculating squared error
        mse_eta.append(np.mean((np.array(eta_estimates[iT]) - true_param[0]) ** 2))
        mse_mu.append(np.mean((np.array(mu_estimates[iT]) - true_param[1]) ** 2))
        mse_gamma.append(np.mean((np.array(gamma_estimates[iT]) - true_param[2]) ** 2))

    T_ref = np.array(T_list, dtype=float)
    # Small horizontal jitter to reduce marker overlap.
    T_eta = T_ref * 0.95
    T_mu = T_ref * 1.00
    T_gamma = T_ref * 1.05

    ax.loglog(T_eta, mse_eta, "o-", label="η")
    ax.loglog(T_mu, mse_mu, "^-", label="μ")
    ax.loglog(T_gamma, mse_gamma, "s-", label="γ")

    ax.loglog(T_ref, T_ref ** (-1), "--", color="gray", label="Slope = -1")

    ax.set_title(f"Δ={Delta}\n")
    ax.set_ylabel(f"MSE, γ={gamma_true}")
    ax.set_xlabel("T")
    ax.set_xticks([100, 300, 1000])
    ax.set_xticklabels(["100", "300", "1000"])

    ax.grid(True, which="both", linestyle="--", alpha=0.5)

def plot_boxplot(ax, estimates, labels):
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

def whittle_powerlaw_mse(
    T_list,
    Delta_list,
    true_param,
    R,
    K_alias,
    a_true,
    maxiter=100,
    s=1,
    k0=200,
    step_hi=1,
    ftol=1e-6,
    eps=1e-6,
    retry_on_fail=True,
    retry_ftol=1e-8,
    retry_eps=1e-4,
    retry_attempts=5,
    time_every=0,
    print_summary=True,
    return_records=True,
):
    true_param = np.asarray(true_param, float)
    if true_param.shape != (3,):
        raise ValueError("true_param must be a length-3 array: (eta, mu, gamma)")

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(Delta_list),
        figsize=(4.5 * len(Delta_list), 4),
        sharey=True,
    )
    axes = np.atleast_2d(axes)
    axes = axes.reshape(1, len(Delta_list))

    summary = {}
    summary_lines = []

    for iD, Delta in enumerate(Delta_list):
        ax = axes[0, iD]
        eta_estimates, mu_estimates, gamma_estimates = simulate_and_estimate_powerlaw(
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
            retry_ftol=retry_ftol,
            retry_eps=retry_eps,
            retry_attempts=retry_attempts,
            time_every=time_every,
            )
        summary[Delta] = {
            "eta_estimates": np.asarray(eta_estimates, dtype=object),
            "mu_estimates": np.asarray(mu_estimates, dtype=object),
            "gamma_estimates": np.asarray(gamma_estimates, dtype=object),
        }

        def _mse_per_t(estimates_list, true_value):
            out = []
            for vals in estimates_list:
                if len(vals) == 0:
                    out.append(np.nan)
                else:
                    arr = np.asarray(vals, float)
                    out.append(np.mean((arr - true_value) ** 2))
            return np.asarray(out, float)

        mse_eta = _mse_per_t(eta_estimates, true_param[0])
        mse_mu = _mse_per_t(mu_estimates, true_param[1])
        mse_gamma = _mse_per_t(gamma_estimates, true_param[2])

        log_mse_eta = np.log(mse_eta)
        log_mse_mu = np.log(mse_mu)
        log_mse_gamma = np.log(mse_gamma)

        for t_val, eta_val, mu_val, gamma_val in zip(T_list, log_mse_eta, log_mse_mu, log_mse_gamma):
            summary_lines.append(
                f"Delta={Delta}, T={t_val} | log_mse_eta={eta_val:.6g} | "
                f"log_mse_mu={mu_val:.6g} | log_mse_gamma={gamma_val:.6g}"
            )
        plot_power_mse_curves(
            ax,
            T_list,
            eta_estimates,
            mu_estimates,
            gamma_estimates,
            Delta,
            true_param[2],
            true_param,
        )

    fig.subplots_adjust(top=0.85)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle("Whittle (Power-law) MSE vs T across Δ", y=1.10, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()

    if print_summary:
        print("\n========== Whittle Power | MSE vs T ==========")
        for line in summary_lines:
            print(line)

    if return_records:
        return summary_lines, summary
    return summary

def whittle_powerlaw_boxplot(
    T_list,
    Delta_list,
    true_param,
    R,
    K_alias,
    a_true,
    maxiter=100,
    s=5.0,
    k0=200,
    step_hi=1,
    ftol=1e-6,
    eps=1e-4,
    retry_on_fail=True,
    retry_ftol=1e-8,
    retry_eps=1e-4,
    retry_attempts=5,
    time_every=0,
    print_summary=True,
    return_records=True,
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
            gamma_vals = []
            fun_vals = []
            fail = 0
            block_elapsed = 0.0

            for r in range(R):
                seed = 2025 + r + 10_000 * iT + 100 * iD
                rng = np.random.default_rng(seed + 12345)

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
                    par_hat, res = whittle_powerlaw(
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
                    best = None
                    base_theta = res.x if hasattr(res, "x") and np.all(np.isfinite(res.x)) else None
                    for _ in range(retry_attempts):
                        if base_theta is None:
                            mu0 = rng.uniform(0.02, 0.98)
                            gamma0 = np.exp(rng.uniform(np.log(0.3), np.log(20.0)))
                            theta0 = np.array([_logit(mu0), np.log(gamma0) / s], float)
                        else:
                            theta0 = np.asarray(base_theta, float) + rng.normal(scale=0.2, size=2)
                        try:
                            cand_hat, cand_res = whittle_powerlaw(
                                bindata,
                                Delta,
                                K_alias=K_alias,
                                a=a_true,
                                maxiter=maxiter,
                                s=s,
                                k0=k0,
                                step_hi=step_hi,
                                ftol=retry_ftol,
                                eps=retry_eps,
                                use_grid_init=False,
                                theta0=theta0,
                            )
                        except ValueError:
                            continue
                        if not np.all(np.isfinite(cand_hat)) or not getattr(cand_res, "success", False):
                            continue
                        if (best is None) or (cand_res.fun < best[1].fun):
                            best = (cand_hat, cand_res)
                    if best is not None:
                        par_hat, res = best

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
                        records.append( # type: ignore
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

                # Append the estimates of eta, mu, and gamma to their respective lists
                estimates.append(par_hat)
                gamma_vals.append(par_hat[2])
                fun_vals.append(res.fun)

                if return_records:
                    records.append( # type: ignore
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

            # Convert estimates into a NumPy array for easier manipulation
            estimates = np.asarray(estimates)

            # Plot boxplot with estimates
            plot_boxplot(ax, estimates, labels)

            # Calculate summary statistics for each parameter
            mean_est = estimates.mean(axis=0)
            median_est = np.median(estimates, axis=0)
            mse_est = np.mean((estimates - true_param) ** 2, axis=0)
            std_est = (
                estimates.std(axis=0, ddof=1)
                if estimates.shape[0] > 1
                else np.full(3, np.nan)
            )

            # Plot the mean and true values
            ax.scatter(
                np.arange(1, 4),
                mean_est,
                color="tab:blue",
                marker="D",
                s=18,
                zorder=4,
            )
            ax.scatter(np.arange(1, 4), true_param, color="red", s=22, zorder=5)

            ax.set_title(f"T={T}, Δ={Delta}")
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.set_ylim(0.0, 4.0)

            mean_str = np.array2string(mean_est, precision=8, separator=" ")
            median_str = np.array2string(median_est, precision=8, separator=" ")
            mse_str = np.array2string(mse_est, precision=8, separator=" ")
            std_str = np.array2string(std_est, precision=8, separator=" ")
            gamma_vals = np.asarray(gamma_vals, float)
            fun_vals = np.asarray(fun_vals, float)
            outlier_mask = gamma_vals > 10.0
            outlier_ratio = float(np.mean(outlier_mask)) if gamma_vals.size else 0.0
            outlier_fun = float(np.nanmean(fun_vals[outlier_mask])) if np.any(outlier_mask) else np.nan
            summary_lines.append(
                f"T={T}, Δ={Delta} | mean={mean_str} | median={median_str} | mse={mse_str} "
                f"| std={std_str} | gamma>10={outlier_ratio:.1%} | fun@outliers={outlier_fun:.6g} "
                f"| Rep={len(estimates)} | fail={fail}"
            )

    fig.suptitle(
        "Whittle Estimator (Power-law Hawkes) across (T, Δ)",
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
