# spectral.py
import numpy as np
import matplotlib.pyplot as plt
from data_gener import binsized
import mpmath as mp
try:
    import incomplete_gamma as _incomplete_gamma
    _HAS_CPP_GAMMA = True
except ImportError:
    _incomplete_gamma = None
    _HAS_CPP_GAMMA = False

def sinc(x: np.ndarray) -> np.ndarray:
    out = np.ones_like(x, dtype=float)
    mask = (x != 0)
    out[mask] = np.sin(x[mask]) / x[mask]
    return out

def _Gamma_upper_complex(s, z):
    """Upper incomplete gamma Γ(s, z) with complex z"""
    return mp.gammainc(s, z, mp.inf)

def _Gamma_upper_complex_vec(s, xi, a):
    if _HAS_CPP_GAMMA:
        return _incomplete_gamma.compute_complex_incomplete_gamma_xi(float(s.real), float(s.imag), xi, a)
    return np.array([_Gamma_upper_complex(s, 1j * x * a) for x in xi], dtype=complex)

########################--periodogram calculation--#########################
#########################################################################

def periodogram(counts: np.ndarray):
    x = np.asarray(counts, dtype=float)
    n = x.size
    x_centered = x - x.mean()  # 去均值
    dft = np.fft.fft(x_centered)  # 计算DFT
    I = (np.abs(dft) ** 2) / n  # 计算功率谱（幅度平方）
    
    # 生成频率值，确保包含负频率部分
    omega = 2 * np.pi * np.fft.fftfreq(n)
    mask = omega >= 0

    return omega[mask], I[mask]

def exp_periodogram_plot(events, delta=0.25, T=1000, para=(1.0, 0.5, 1.0), alias=3):
    bindata = binsized(events, delta, T)
    omega, I = periodogram(bindata)

    fc_vals = exp_fc(omega, para, delta)
    fd_vals = exp_fd(omega, para, delta, alias)

    plt.figure(figsize=(8, 4))
    plt.plot(omega, I, alpha=0.3, label="Periodogram I(w)")
    plt.plot(omega, fc_vals, lw=2, label="Theoretical f_c(w)")
    plt.plot(omega, fd_vals, lw=2, label="Theoretical f_d(w)")
    plt.yscale("log")
    plt.xlabel("w")
    plt.ylabel("Exponential spectral power")
    plt.ylim(1e-2, 1e1)
    plt.legend()
    plt.tight_layout()
    plt.show()

def powerlaw_periodogram_plot(events, delta=0.25, T=1000, para=(1.0, 0.5, 2.5), a=1.5, alias=3):
    bindata = binsized(events, delta, T)
    omega, I = periodogram(bindata)

    fd_vals = powerlaw_fd(omega, para, delta, alias, a=a)
    fc_vals = powerlaw_fc(omega, para, delta, a, sinc(omega / 2.0) ** 2)

    plt.figure(figsize=(8, 4))
    plt.plot(omega, I, alpha=0.3, label="Periodogram I(w)")
    plt.plot(omega, fc_vals, lw=2, label="Theoretical f_c(w)")
    plt.plot(omega, fd_vals, lw=2, label="Theoretical f_d(w)")
    plt.yscale("log")
    plt.xlabel("w")
    plt.ylabel("Powerlaw spectral power")
    plt.ylim(1e-2, 1e1)
    plt.legend()
    plt.tight_layout()
    plt.show()

####################---------- Exponential Hawkes spectral density -------------#
################################################################################

def exp_fc(omega: np.ndarray, param, Delta: float) -> np.ndarray:
    baseline, mu, beta = param
    m = baseline / (1.0 - mu)
    xi = omega / Delta
    h_tilde = mu * beta / (beta + 1j * xi)
    sinc2 = sinc(omega / 2.0) ** 2
    denom = np.abs(1.0 - h_tilde) ** 2
    return m * Delta * sinc2 / denom

def exp_fd(omega: np.ndarray, param, Delta: float, K_alias: int = 3) -> np.ndarray:
    fd = np.zeros_like(omega, dtype=float)
    for k in range(-K_alias, K_alias + 1):
        fd += exp_fc(omega + 2 * np.pi * k, param, Delta)
    return fd


####################---------- Power-law Hawkes spectral density -------------#
################################################################################

def powerlaw_htilde(xi: float | np.ndarray, mu: float, gamma: float, a: float) -> complex | np.ndarray:
    """
    Fourier transform of power-law kernel:
        h(t) = mu * gamma * a^gamma * (a + t)^(-1 - gamma)
    """
    xi_arr = np.asarray(xi, dtype=float)
    s = -gamma
    pow_term = (1j * xi_arr) ** gamma
    Gup = _Gamma_upper_complex_vec(s, xi_arr, a)
    h = (
        mu
        * gamma
        * (a ** gamma)
        * np.exp(1j * xi_arr * a)
        * pow_term
        * Gup
    )
    if np.isscalar(xi):
        return h.item()
    return h


def powerlaw_fc(
    omega_shifted: np.ndarray,
    param: tuple[float, float, float],  # (eta, mu, gamma)
    Delta: float,
    a: float,
    sinc2_shifted: np.ndarray,
) -> np.ndarray:
    eta, mu, gamma = param
    m = eta / (1.0 - mu)

    xi = omega_shifted / Delta
    h_tilde = powerlaw_htilde(xi, mu, gamma, a)
    denom = np.abs(1.0 - h_tilde) ** 2
    return (m * Delta * sinc2_shifted) / denom


def powerlaw_fd(
    omega: np.ndarray,
    param: tuple[float, float, float],
    Delta: float,
    K_alias: int = 2,
    a: float = 1.5,
) -> np.ndarray:
    k = np.arange(-K_alias, K_alias + 1)
    omg_shifted = omega[:, None] + 2 * np.pi * k[None, :]
    sinc2_k = sinc(omg_shifted / 2.0) ** 2
    fc_vals = powerlaw_fc(omg_shifted, param, Delta, a, sinc2_k)
    return np.sum(fc_vals, axis=1)

############################-- Guassian Hawkes spectral density -------------#
###########################################################################

def gaussian_htilde(xi: float, nu: float, mu: float, sigma: float) -> complex:
    """
    Fourier transform of Gaussian kernel:
        h(t) = mu * g(t; nu, sigma) on the full real line (paper setting).
    """
    return mu * np.exp(-0.5 * (xi ** 2) * (sigma ** 2)) * np.exp(1j * xi * nu)

def gaussian_fc(
    omega_shifted: np.ndarray,
    param: tuple[float, float, float, float],  # (eta, mu, sigma, nu)
    Delta: float,
    sinc2_shifted: np.ndarray,
) -> np.ndarray:
    eta, mu, sigma, nu = param  
    m = eta / (1.0 - mu)
    xi = omega_shifted / Delta
    h_tilde = gaussian_htilde(xi, nu, mu, sigma)
    denom = np.abs(1.0 - h_tilde) ** 2
    out = (m * Delta * sinc2_shifted) / denom

    return out

def gaussian_fd(
    omega: np.ndarray,
    param: tuple[float, float, float, float],
    Delta: float,
    K_alias: int = 2,
) -> np.ndarray:
    k = np.arange(-K_alias, K_alias + 1)
    omg_shifted = omega[:, None] + 2 * np.pi * k[None, :]
    sinc2_k = sinc(omg_shifted / 2.0) ** 2
    fc_values = gaussian_fc(omg_shifted, param, Delta, sinc2_k)
    return np.sum(fc_values, axis=1)
