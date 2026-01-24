import time
import numpy as np
import mpmath as mp

try:
    import incomplete_gamma as _incomplete_gamma
except ImportError as exc:
    raise SystemExit("incomplete_gamma module not found; build with setup.py first") from exc


def cpp_gamma_upper(a: complex, z: complex) -> complex:
    return _incomplete_gamma.compute_complex_incomplete_gamma(
        float(a.real), float(a.imag), float(z.real), float(z.imag)
    )


def sample_check(
    n_samples: int = 200,
    gamma_max: float = 10.0,
    xi_max: float = 50.0,
    a_param: float = 1.5,
    seed: int = 1234,
    rel_target: float = 1e-6,
):
    rng = np.random.default_rng(seed)
    rel_errors = []
    worst = (0.0, None, None, None, None)
    within = 0

    for _ in range(n_samples):
        gamma = rng.uniform(1e-3, gamma_max)
        xi = rng.uniform(-xi_max, xi_max)
        z = 1j * xi * a_param
        s = -gamma
        cpp = cpp_gamma_upper(s, z)
        mpv = mp.gammainc(s, z, mp.inf)
        denom = max(1e-12, abs(mpv))
        rel = abs(cpp - mpv) / denom
        rel_errors.append(rel)
        if rel <= rel_target:
            within += 1
        if rel > worst[0]:
            worst = (rel, gamma, xi, cpp, mpv)

    rel_errors = np.array(rel_errors, dtype=float)
    ratio = within / n_samples if n_samples else 0.0
    print(f"samples: {n_samples}")
    print(f"median rel err: {np.median(rel_errors):.3e}")
    print(f"p95 rel err: {np.percentile(rel_errors, 95):.3e}")
    print(f"max rel err: {rel_errors.max():.3e}")
    print(f"rel <= {rel_target:.1e}: {ratio:.2%}")
    print("worst case:")
    print(f"  gamma={worst[1]:.6g} xi={worst[2]:.6g}")
    print(f"  cpp={worst[3]}")
    print(f"  mp ={worst[4]}")


def speed_check(n_samples: int = 2000, a_param: float = 1.5):
    rng = np.random.default_rng(123)
    gamma = rng.uniform(1e-3, 10.0, size=n_samples)
    xi = rng.uniform(-50.0, 50.0, size=n_samples)
    z = 1j * xi * a_param
    s = -gamma

    time0 = time.time()
    for i in range(n_samples):
        _ = cpp_gamma_upper(s[i], z[i])
    time1 = time.time()
    print(f"cpp total: {time1 - time0:.4f}s")

    time0 = time.time()
    for i in range(n_samples):
        _ = mp.gammainc(s[i], z[i], mp.inf)
    time1 = time.time()
    print(f"mp total:  {time1 - time0:.4f}s")


if __name__ == "__main__":
    sample_check()
    speed_check()
