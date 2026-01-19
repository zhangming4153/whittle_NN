# spectral.py
import numpy as np
import matplotlib.pyplot as plt
from data_gener import binsized
import mpmath as mp


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


# print(periodogram(np.array([1,2,3,4])))

def sinc(x: np.ndarray) -> np.ndarray:
    out = np.ones_like(x, dtype=float)
    mask = (x != 0)
    out[mask] = np.sin(x[mask]) / x[mask]
    return out

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

def _Gamma_upper_complex(s, z):
    """Upper incomplete gamma Γ(s, z) with complex z"""
    return mp.gammainc(s, z, mp.inf)

# =====================
# FT of power-law kernel
# =====================
def powerlaw_htilde(xi: float, mu: float, gamma: float, a: float) -> complex:
    """
    Fourier transform of power-law kernel:
        h(t) = mu * gamma * a^gamma * (a + t)^(-1 - gamma)
    """
    z = 1j * xi * a
    s = -gamma
    pow_term = (1j * xi) ** gamma
    Gup = complex(_Gamma_upper_complex(s, z))
    return (
        mu
        * gamma
        * (a ** gamma)
        * np.exp(1j * xi * a)
        * pow_term
        * Gup
    )


# =====================
# Hawkes spectral density (core)
# =====================
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
    out = np.empty_like(omega_shifted, dtype=float)

    for i in range(omega_shifted.size):
        h_tilde = powerlaw_htilde(xi[i], mu, gamma, a)
        denom = abs(1.0 - h_tilde) ** 2
        out[i] = (m * Delta * sinc2_shifted[i]) / denom

    return out


def powerlaw_fd(
    omega: np.ndarray,
    param: tuple[float, float, float],
    Delta: float,
    K_alias: int = 2,
    a: float = 1.5,
) -> np.ndarray:
    fd = np.zeros_like(omega, dtype=float)

    for k in range(-K_alias, K_alias + 1):
        omg = omega + 2 * np.pi * k
        sinc2_k = sinc(omg / 2.0) ** 2
        fd += powerlaw_fc(omg, param, Delta, a, sinc2_k)

    return fd
