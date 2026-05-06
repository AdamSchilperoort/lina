"""lina_cpp.wfe - C++-backed mirror of lina.wfe.

Math primitives (Zernike index conversions, generate_freqs, roll_psd,
generate_time_series, compute_cumulative_psd) delegate to the compiled
extension. The poppy-dependent generate_opd / generate_wfe wrappers are
re-exported from lina because porting poppy itself is out of scope.

Plotting helpers (plot_psd, plot_time_series, ...) re-export from lina
since they're matplotlib calls with no math benefit from C++.

Note on RNG reproducibility
---------------------------
The C++ generate_time_series uses a stable C++ Mersenne Twister so the
seed -> time-series mapping is portable across machines. The Python
lina.wfe.generate_time_series uses numpy's PRNG and produces a
*different* time series for the same `seed`. The PSD shape is identical
between the two backends (statistics match); only the per-realisation
sample sequence differs. If you need bit-exact agreement, pass the same
random phases to both.
"""

from __future__ import annotations

import numpy as np

from . import _core
from .math_module import ensure_np_array


# ---------------------------------------------------------------------------
# Zernike index conversions
# ---------------------------------------------------------------------------

def noll_index_to_mn(j):
    """Noll index j -> (m, n)."""
    return _core.noll_index_to_mn(int(j))


def mn_to_noll_index(m, n):
    """(m, n) -> Noll index j."""
    return _core.mn_to_noll_index(int(m), int(n))


def fringe_index_to_mn(j):
    """Fringe index j -> (m, n)."""
    return _core.fringe_index_to_mn(int(j))


def mn_to_fringe_index(m, n):
    """(m, n) -> Fringe index j."""
    return _core.mn_to_fringe_index(int(m), int(n))


# ---------------------------------------------------------------------------
# Frequency-grid + PSD synthesis
# ---------------------------------------------------------------------------

def generate_freqs(delt=0.1e-3, tmax=10.0, verbose=False):
    """Returns (freqs, delf, times). Matches lina.wfe.generate_freqs."""
    freqs, delf, times = _core.wfe_generate_freqs(float(delt), float(tmax))
    if verbose:
        print(
            f"Generated frequency vector with sampling of {delf:.2e}Hz and "
            f"maximum frequency of {freqs.max():.2e}Hz."
        )
    return freqs, delf, times


def roll_psd(freqs, beta, f_roll, alpha, normalized=True, verbose=True):
    """Knee PSD on a frequency grid. Matches lina.wfe.roll_psd."""
    freqs = ensure_np_array(freqs).astype(np.float64, copy=False)
    psd = _core.wfe_roll_psd(freqs, float(beta), float(f_roll), float(alpha), bool(normalized))
    if verbose:
        try:
            from scipy.integrate import simpson
            psd_rms = float(np.sqrt(simpson(psd, x=freqs)))
            print(f"\tRMS of generated knee PSD: {psd_rms:.3e}")
        except Exception:
            pass
    return psd


def generate_time_series(psd, freqs, rms=None, seed=123, return_np=False, verbose=False):
    """Synthesize a real-valued time series from a one-sided PSD.

    The C++ implementation uses a stable Mersenne Twister so the
    seed -> output mapping is reproducible across machines (numpy's
    PRNG bit-stream isn't guaranteed to be portable). PSD statistics
    match the Python version to floating-point precision.
    """
    psd = ensure_np_array(psd).astype(np.float64, copy=False)
    freqs = ensure_np_array(freqs).astype(np.float64, copy=False)
    rms_target = 0.0 if rms is None else float(rms)
    values, times = _core.wfe_generate_time_series(
        psd, freqs, rms_target=rms_target, seed=int(seed)
    )
    if verbose:
        ts_rms = float(np.sqrt(np.mean(np.square(values))))
        print(f"\tRMS of generated time series: {ts_rms:.3e} RMS")
    return values, times


def compute_cumulative_psd(freqs, psd):
    """Returns (cumulative_psd_sqrt, freqs[1:]). Matches lina.wfe."""
    freqs = ensure_np_array(freqs).astype(np.float64, copy=False)
    psd = ensure_np_array(psd).astype(np.float64, copy=False)
    return _core.wfe_compute_cumulative_psd(freqs, psd)


# ---------------------------------------------------------------------------
# Re-exports from lina.wfe -- poppy-dependent functions and plotting.
# ---------------------------------------------------------------------------

from lina.wfe import (  # noqa: E402, F401
    generate_opd,
    generate_wfe,
    plot_psd,
    plot_psd_and_time_series,
    plot_time_series,
)
