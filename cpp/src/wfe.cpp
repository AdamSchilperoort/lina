#include "lina/wfe.h"

#include "lina/props.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <random>
#include <stdexcept>

#ifdef LINA_USE_FFTW
#include <fftw3.h>
#endif

namespace lina::wfe {

namespace {
constexpr double kPi = 3.141592653589793238462643383279502884;
}

// ---------------------------------------------------------------------------
// Zernike index conversions
// ---------------------------------------------------------------------------

// Replicates lina.wfe.noll_index_to_mn exactly, including the floor and
// integer-modulo conventions used in the original.
std::pair<int, int> noll_index_to_mn(int j) {
    // n = floor( (sqrt(8*(j-1)+1) - 1) / 2 )
    const double s = std::sqrt(8.0 * static_cast<double>(j - 1) + 1.0);
    const int n = static_cast<int>(std::floor((s - 1.0) / 2.0));

    // m = (-1)^j * ( n%2 + 2*floor( (j - n*(n+1)/2 - 1 + (n+1)%2 )/2 ) )
    // The original mixes integer and float arithmetic via numpy.floor;
    // emulate that here with explicit doubles to keep the same rounding.
    const int n_mod_2 = n % 2;
    const int n_plus_1_mod_2 = (n + 1) % 2;
    const double q = (static_cast<double>(j) - static_cast<double>(n) * (n + 1) / 2.0
                      - 1.0 + static_cast<double>(n_plus_1_mod_2)) / 2.0;
    const int sign = ((j % 2) == 0) ? 1 : -1;
    const int m = sign * (n_mod_2 + 2 * static_cast<int>(std::floor(q)));
    return {m, n};
}

int mn_to_noll_index(int m, int n) {
    // Branch matches lina.wfe.mn_to_noll_index exactly.
    int abs_m = std::abs(m);
    double j = 0.0;
    const int mod4 = ((n % 4) + 4) % 4;
    if (mod4 <= 1) {
        if (m > 0)  j = n * (n + 1) / 2.0 + abs_m + 0;
        else        j = n * (n + 1) / 2.0 + abs_m + 1;  // m <= 0
    } else {
        // mod4 in {2, 3}
        if (m < 0)  j = n * (n + 1) / 2.0 + abs_m + 0;
        else        j = n * (n + 1) / 2.0 + abs_m + 1;  // m >= 0
    }
    return static_cast<int>(j);
}

std::pair<int, int> fringe_index_to_mn(int j) {
    // g = ceil(sqrt(j) - 1)
    const double g = std::ceil(std::sqrt(static_cast<double>(j)) - 1.0);
    // n = g - 1 + ceil( (j - g^2) / 2 )
    const double n_d = g - 1.0 + std::ceil((static_cast<double>(j) - g * g) / 2.0);
    // m = (-1)^( ((j - g^2) % 2) + 1 ) * (2*g - n)
    const int mod = (((static_cast<int>(j - g * g) % 2) + 2) % 2);
    const int sign = ((mod + 1) % 2 == 0) ? 1 : -1;
    const double m_d = sign * (2.0 * g - n_d);
    return {static_cast<int>(m_d), static_cast<int>(n_d)};
}

int mn_to_fringe_index(int m, int n) {
    const int abs_m = std::abs(m);
    double j;
    if (m >= 0) {
        // j = (1 + (n + |m|)/2)^2 - 2|m| + 0
        const double base = 1.0 + (static_cast<double>(n + abs_m)) / 2.0;
        j = base * base - 2.0 * abs_m + 0.0;
    } else {
        const double base = 1.0 + (static_cast<double>(n + abs_m)) / 2.0;
        j = base * base - 2.0 * abs_m + 1.0;
    }
    return static_cast<int>(j);
}

// ---------------------------------------------------------------------------
// Frequency-grid construction
// ---------------------------------------------------------------------------

FreqGrid generate_freqs(double delt, double tmax) {
    const double fmax = 1.0 / delt / 2.0;
    // The Python code overwrites delf twice; the second assignment wins:
    //     delf = 1/(tmax + delt)
    //     delf = 1/(tmax)
    // We honour the final value (1/tmax).
    const double delf = 1.0 / tmax;
    const int Nf = static_cast<int>(std::round(fmax / delf)) + 1;

    FreqGrid out;
    out.freqs.resize(Nf);
    out.delf = delf;
    for (int i = 0; i < Nf; ++i) {
        // np.linspace(0, fmax, Nf) => freqs[i] = i * (fmax/(Nf-1)).
        out.freqs[i] = (Nf > 1)
            ? static_cast<double>(i) * fmax / static_cast<double>(Nf - 1)
            : 0.0;
    }

    const int Nt = 2 * (Nf - 1);
    out.times.resize(Nt);
    for (int i = 0; i < Nt; ++i) {
        // np.linspace(0, (Nt-1)*delt, Nt) => times[i] = i * delt.
        out.times[i] = static_cast<double>(i) * delt;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Knee PSD
// ---------------------------------------------------------------------------

std::vector<double> roll_psd(const std::vector<double>& freqs,
                             double beta,
                             double f_roll,
                             double alpha,
                             bool normalized) {
    std::vector<double> psd(freqs.size());
    const double beta_sq = beta * beta;
    const double norm = normalized ? (alpha - 1.0) / f_roll : 1.0;
    for (std::size_t i = 0; i < freqs.size(); ++i) {
        const double denom = std::pow(1.0 + freqs[i] / f_roll, alpha);
        psd[i] = beta_sq / denom * norm;
    }
    return psd;
}

// ---------------------------------------------------------------------------
// Time-series synthesis
// ---------------------------------------------------------------------------

TimeSeries generate_time_series(const std::vector<double>& psd,
                                const std::vector<double>& freqs,
                                double rms_target,
                                std::uint64_t seed) {
    if (psd.size() != freqs.size()) {
        throw std::invalid_argument(
            "generate_time_series: psd and freqs must have the same length");
    }
    if (psd.size() < 2) {
        throw std::invalid_argument(
            "generate_time_series: need at least 2 PSD samples");
    }

    const double fmax = freqs.back();
    const double delf = freqs[1] - freqs[0];
    const std::size_t Nf = psd.size();
    const std::size_t Nt = 2 * (Nf - 1);
    const double del_time = 1.0 / (2.0 * fmax);

    // 1) Build a one-sided power spectrum, doubling DC and Nyquist so
    // that the symmetric two-sided spectrum has the correct integral.
    std::vector<double> P_one_sided = psd;
    P_one_sided[0] *= 2.0;
    P_one_sided[Nf - 1] *= 2.0;

    // 2) Mirror into a two-sided spectrum of length Nt with conjugate
    // symmetry. P_two_sided[0..Nf-1] = P_one_sided; the second half is
    // P_one_sided[Nf-2 .. 1] reversed.
    std::vector<std::complex<double>> P_two_sided(Nt, {0.0, 0.0});
    for (std::size_t i = 0; i < Nf; ++i) {
        P_two_sided[i] = {P_one_sided[i], 0.0};
    }
    for (std::size_t i = Nf, k = Nf - 2; i < Nt; ++i, --k) {
        P_two_sided[i] = {P_one_sided[k], 0.0};
    }

    // 3) Amplitude = sqrt(power) * Nt * sqrt(delf/2).
    const double scale = static_cast<double>(Nt) * std::sqrt(delf / 2.0);
    std::vector<std::complex<double>> A(Nt);
    for (std::size_t i = 0; i < Nt; ++i) {
        A[i] = std::sqrt(P_two_sided[i]) * scale;
    }

    // 4) Random phases for bins 1..Nt/2; the negative-frequency bins
    // get the conjugate so the time series is real. We use a stable
    // C++ Mersenne Twister so the seed -> output mapping is portable
    // (numpy's PRNG bit-stream isn't reproducible from C++).
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uni(0.0, 2.0 * kPi);
    const std::size_t half = Nt / 2;
    std::vector<double> phases(half);
    for (std::size_t i = 0; i < half; ++i) phases[i] = uni(rng);

    const std::complex<double> j(0.0, 1.0);
    // amplitude_spectrum[1 .. half] *= exp( 2j * phases[0..half-1] )
    for (std::size_t i = 1; i <= half; ++i) {
        A[i] *= std::exp(2.0 * j * phases[i - 1]);
    }
    // amplitude_spectrum[half .. Nt-1] *= exp(-2j * phases[half-1 .. 0])
    // i.e. the upper half gets the conjugate phases in reverse order.
    for (std::size_t i = half; i < Nt; ++i) {
        const std::size_t k = (Nt - 1) - i;  // 0 when i == Nt-1, half-1 when i == half
        A[i] *= std::exp(-2.0 * j * phases[k]);
    }

    // 5) Inverse FFT (1D). We don't need the full lina::ifft (which is
    // 2D + fftshift) -- just a straight 1D inverse DFT with the numpy
    // convention y[k] = (1/N) * sum_n A[n] * exp(2*pi*i*k*n/N). For
    // typical Nt (10**4 .. 10**7) we use a Cooley-Tukey routine via
    // FFTW if it's available, otherwise an O(N log N) iterative FFT.
    std::vector<double> ts(Nt);
    {
        // Build an in-place complex buffer; FFTW backward = sum without
        // 1/N normalization, so divide by Nt afterwards.
        std::vector<std::complex<double>> buf = A;
#ifdef LINA_USE_FFTW
        fftw_plan plan = fftw_plan_dft_1d(
            static_cast<int>(Nt),
            reinterpret_cast<fftw_complex*>(buf.data()),
            reinterpret_cast<fftw_complex*>(buf.data()),
            FFTW_BACKWARD, FFTW_ESTIMATE);
        fftw_execute(plan);
        fftw_destroy_plan(plan);
#else
        // Naive O(N^2) DFT fallback. Only ever taken in builds where
        // LINA_USE_FFTW is off (which is unsupported elsewhere too).
        std::vector<std::complex<double>> tmp(Nt);
        for (std::size_t k = 0; k < Nt; ++k) {
            std::complex<double> s(0.0, 0.0);
            for (std::size_t n = 0; n < Nt; ++n) {
                const double phase = 2.0 * kPi * static_cast<double>(k * n) / static_cast<double>(Nt);
                s += buf[n] * std::complex<double>(std::cos(phase), std::sin(phase));
            }
            tmp[k] = s;
        }
        buf = tmp;
#endif
        for (std::size_t i = 0; i < Nt; ++i) {
            ts[i] = buf[i].real() / static_cast<double>(Nt);
        }
    }

    // 6) Optional rms rescale.
    if (rms_target > 0.0) {
        double sumsq = 0.0;
        for (double v : ts) sumsq += v * v;
        const double rms = std::sqrt(sumsq / static_cast<double>(Nt));
        if (rms > 0.0) {
            const double factor = rms_target / rms;
            for (double& v : ts) v *= factor;
        }
    }

    // 7) Build matching time vector.
    std::vector<double> times(Nt);
    for (std::size_t i = 0; i < Nt; ++i) {
        times[i] = static_cast<double>(i) * del_time;
    }

    return {std::move(ts), std::move(times)};
}

// ---------------------------------------------------------------------------
// Cumulative PSD
// ---------------------------------------------------------------------------

namespace {
// Composite Simpson's rule for an arbitrary x grid, matching the
// behaviour of scipy.integrate.simpson(y, x=x) on uniform input from
// generate_freqs(). For an even number of segments we apply Simpson
// pairwise; for an odd number we use a trapezoidal step on the last
// segment. Inputs of length < 2 return 0.
double simpson(const std::vector<double>& y, const std::vector<double>& x) {
    const std::size_t n = y.size();
    if (n < 2) return 0.0;
    if (n == 2) return 0.5 * (y[0] + y[1]) * (x[1] - x[0]);

    double total = 0.0;
    std::size_t i = 0;
    for (; i + 2 < n; i += 2) {
        // For uniform spacing, h0 == h1 and Simpson is (h/3)*(y0 + 4y1 + y2).
        // For arbitrary spacing the same form (h0+h1)/6 * (y0 + 4 y1 + y2)
        // is a 1st-order-accurate approximation; scipy.integrate.simpson
        // uses a slightly more general weighting for non-uniform x. Since
        // we always feed uniformly spaced freqs from generate_freqs, the
        // two are identical to floating-point precision here.
        const double h0 = x[i + 1] - x[i];
        const double h1 = x[i + 2] - x[i + 1];
        total += (h0 + h1) / 6.0 * (y[i] + 4.0 * y[i + 1] + y[i + 2]);
    }
    // Odd trailing segment -> trapezoid.
    if (i + 1 < n) {
        total += 0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i]);
    }
    return total;
}
} // namespace

std::pair<std::vector<double>, std::vector<double>>
compute_cumulative_psd(const std::vector<double>& freqs,
                       const std::vector<double>& psd) {
    if (freqs.size() != psd.size()) {
        throw std::invalid_argument(
            "compute_cumulative_psd: freqs and psd must have the same length");
    }
    if (freqs.size() < 2) {
        return {{}, {}};
    }

    // The Python reference iterates i in [1, N) and integrates over
    // `psd[0:i]` -- a slice of length `i`, NOT `i+1`. So:
    //   i=1 -> simpson([psd[0]],          [freqs[0]])         = 0   (single-sample)
    //   i=2 -> simpson([psd[0], psd[1]],  [freqs[0], freqs[1]]) = trapezoid
    //   i=3 -> simpson([psd[0..2]],       [freqs[0..2]])      = Simpson's rule
    // Mirror that exactly to keep cum_psd[0] == 0 and cum_psd[k] aligned with freqs[k].
    const std::size_t N = freqs.size();
    std::vector<double> cum(N - 1);
    std::vector<double> tail(N - 1);
    std::vector<double> y_seg, x_seg;
    y_seg.reserve(N); x_seg.reserve(N);
    for (std::size_t i = 1; i < N; ++i) {
        y_seg.assign(psd.begin(), psd.begin() + i);
        x_seg.assign(freqs.begin(), freqs.begin() + i);
        const double integral = simpson(y_seg, x_seg);
        cum[i - 1] = std::sqrt(std::max(integral, 0.0));
        tail[i - 1] = freqs[i];
    }
    return {std::move(cum), std::move(tail)};
}

} // namespace lina::wfe
