#pragma once

// Wavefront error / temporal-PSD synthesis primitives.
//
// This is the math-only subset of lina/wfe.py: Zernike-index
// conversions and PSD / time-series synthesis. The poppy-dependent
// generate_opd / generate_wfe stay in Python (poppy is itself a Python
// optical-modeling library; porting it is out of scope).

#include "lina/array.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace lina::wfe {

// ---------------------------------------------------------------------------
// Zernike index <-> (m, n) conversions. Pure integer math; the C++ output
// matches lina.wfe.{noll_index_to_mn, mn_to_noll_index, fringe_index_to_mn,
// mn_to_fringe_index} bit-exactly for all valid inputs.
// ---------------------------------------------------------------------------
std::pair<int, int> noll_index_to_mn(int j);
int mn_to_noll_index(int m, int n);

std::pair<int, int> fringe_index_to_mn(int j);
int mn_to_fringe_index(int m, int n);

// ---------------------------------------------------------------------------
// Frequency / time arrays for one-sided PSDs. Mirrors the *second*
// definition of generate_freqs() in wfe.py (the first is shadowed).
//
// Inputs:
//   delt -- temporal sample spacing [s]
//   tmax -- total record duration  [s]
// Outputs:
//   freqs -- one-sided frequency vector [0, fmax], length Nf = round(fmax/delf)+1
//   delf  -- frequency sample spacing
//   times -- 2*(Nf-1) uniform time samples
// ---------------------------------------------------------------------------
struct FreqGrid {
    std::vector<double> freqs;
    double delf;
    std::vector<double> times;
};
FreqGrid generate_freqs(double delt, double tmax);

// One-sided knee PSD. If `normalized` is true, the trailing
// (alpha-1)/f_roll factor makes the integral consistent with rms ~ beta.
//   psd = beta^2 / (1 + f/f_roll)^alpha   * (alpha-1)/f_roll  (when normalized)
//   psd = beta^2 / (1 + f/f_roll)^alpha                       (otherwise)
std::vector<double> roll_psd(const std::vector<double>& freqs,
                             double beta,
                             double f_roll,
                             double alpha,
                             bool normalized = true);

// Generate a real-valued time series whose one-sided PSD matches `psd`.
// Implements the same procedure as lina.wfe.generate_time_series:
//   1. Build a two-sided amplitude spectrum from the one-sided psd.
//   2. Apply random uniform phases [0, 2pi) (with conjugate symmetry).
//   3. Inverse-FFT.
//   4. Optionally rescale to the requested rms.
// `seed` is used for the per-bin random phases (numpy's PRNG isn't bit
// reproducible from C++, so we use a stable C++ Mersenne Twister with a
// documented PRNG sequence -- see the test for the seed -> output
// reference values).
struct TimeSeries {
    std::vector<double> values;
    std::vector<double> times;
};
TimeSeries generate_time_series(const std::vector<double>& psd,
                                const std::vector<double>& freqs,
                                double rms_target = 0.0,  // 0 means "do not rescale"
                                std::uint64_t seed = 123);

// Cumulative-PSD curve: sqrt(integral_0^f psd(f') df') for each f
// in `freqs`, evaluated by Simpson's rule like
// scipy.integrate.simpson. Returns (cumulative, freqs[1:]).
std::pair<std::vector<double>, std::vector<double>>
compute_cumulative_psd(const std::vector<double>& freqs,
                       const std::vector<double>& psd);

} // namespace lina::wfe
