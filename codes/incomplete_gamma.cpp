#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <mpfr.h>
#include <cmath>
#include <complex>
#include <limits>

namespace py = pybind11;

namespace {
std::complex<double> complex_gamma(std::complex<double> z) {
    static const double pi = std::acos(-1.0);
    static const double g = 7.0;
    static const double coeffs[] = {
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    };

    if (z.real() < 0.5) {
        return pi / (std::sin(pi * z) * complex_gamma(std::complex<double>(1.0, 0.0) - z));
    }

    z -= std::complex<double>(1.0, 0.0);
    std::complex<double> x(coeffs[0], 0.0);
    for (size_t i = 1; i < sizeof(coeffs) / sizeof(coeffs[0]); ++i) {
        x += coeffs[i] / (z + std::complex<double>(static_cast<double>(i), 0.0));
    }

    std::complex<double> t = z + std::complex<double>(g + 0.5, 0.0);
    return std::sqrt(2.0 * pi) * std::pow(t, z + std::complex<double>(0.5, 0.0)) * std::exp(-t) * x;
}

std::complex<double> upper_gamma_contfrac(std::complex<double> a, std::complex<double> z) {
    const int max_iter = 6000;
    const double eps = 1e-12;
    const double tiny = 1e-30;

    std::complex<double> b = z + std::complex<double>(1.0, 0.0) - a;
    std::complex<double> c = std::complex<double>(1.0 / tiny, 0.0);
    std::complex<double> d = std::complex<double>(1.0, 0.0) / b;
    std::complex<double> h = d;

    for (int i = 1; i <= max_iter; ++i) {
        std::complex<double> an = -static_cast<double>(i) * (static_cast<double>(i) - a);
        b += std::complex<double>(2.0, 0.0);
        d = an * d + b;
        if (std::abs(d) < tiny) {
            d = std::complex<double>(tiny, 0.0);
        }
        c = b + an / c;
        if (std::abs(c) < tiny) {
            c = std::complex<double>(tiny, 0.0);
        }
        d = std::complex<double>(1.0, 0.0) / d;
        std::complex<double> del = d * c;
        h *= del;
        if (std::abs(del - std::complex<double>(1.0, 0.0)) < eps) {
            break;
        }
    }

    return h * std::exp(-z + a * std::log(z));
}

std::complex<double> lower_gamma_series(std::complex<double> a, std::complex<double> z) {
    const int max_iter = 6000;
    const double eps = 1e-12;

    std::complex<double> ap = a;
    std::complex<double> sum = std::complex<double>(1.0, 0.0) / a;
    std::complex<double> del = sum;

    for (int n = 1; n <= max_iter; ++n) {
        ap += std::complex<double>(1.0, 0.0);
        del *= z / ap;
        sum += del;
        if (std::abs(del) < eps * std::abs(sum)) {
            break;
        }
    }

    return sum * std::exp(-z + a * std::log(z));
}

std::complex<double> complex_incomplete_gamma(std::complex<double> a, std::complex<double> z) {
    const double z_zero = 1e-12;

    if (std::abs(z) < z_zero) {
        return complex_gamma(a);
    }

    std::complex<double> a_shifted = a;
    int n_shift = 0;
    if (a_shifted.real() <= 0.5) {
        n_shift = static_cast<int>(std::ceil(0.5 - a_shifted.real()));
        a_shifted += std::complex<double>(static_cast<double>(n_shift), 0.0);
    }

    std::complex<double> upper;
    if (std::abs(z) < std::abs(a_shifted) + 1.0) {
        std::complex<double> lower = lower_gamma_series(a_shifted, z);
        upper = complex_gamma(a_shifted) - lower;
    } else {
        upper = upper_gamma_contfrac(a_shifted, z);
    }

    for (int k = 0; k < n_shift; ++k) {
        std::complex<double> a_curr = a_shifted - std::complex<double>(static_cast<double>(k + 1), 0.0);
        std::complex<double> term = std::pow(z, a_curr) * std::exp(-z);
        upper = (upper - term) / a_curr;
    }

    if (!std::isfinite(upper.real()) || !std::isfinite(upper.imag())) {
        return std::complex<double>(std::numeric_limits<double>::quiet_NaN(), 0.0);
    }

    return upper;
}

}  // namespace

std::complex<double> compute_complex_incomplete_gamma(double real_a, double imag_a, double real_x, double imag_x) {
    std::complex<double> a(real_a, imag_a);  // Shape parameter as a complex number
    std::complex<double> x(real_x, imag_x);  // Upper limit as a complex number
    
    // Call the complex incomplete gamma function
    std::complex<double> result = complex_incomplete_gamma(a, x);
    
    return result;
}

py::array_t<std::complex<double>> compute_complex_incomplete_gamma_xi(
    double real_a,
    double imag_a,
    py::array_t<double, py::array::c_style | py::array::forcecast> xi,
    double a_scale
) {
    const auto buf = xi.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("xi must be a 1D array");
    }
    const auto n = static_cast<size_t>(buf.size);
    auto out = py::array_t<std::complex<double>>(buf.size);
    auto out_buf = out.request();

    const double* xi_ptr = static_cast<double*>(buf.ptr);
    auto* out_ptr = static_cast<std::complex<double>*>(out_buf.ptr);

    const std::complex<double> a(real_a, imag_a);
    for (size_t i = 0; i < n; ++i) {
        const std::complex<double> z(0.0, xi_ptr[i] * a_scale);
        out_ptr[i] = complex_incomplete_gamma(a, z);
    }
    return out;
}

PYBIND11_MODULE(incomplete_gamma, m) {
    m.def("compute_complex_incomplete_gamma", &compute_complex_incomplete_gamma, "Compute Complex Incomplete Gamma Function");
    m.def(
        "compute_complex_incomplete_gamma_xi",
        &compute_complex_incomplete_gamma_xi,
        "Compute Complex Incomplete Gamma Function for z = i * xi * a_scale"
    );
}
