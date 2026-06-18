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
import astropy.units as u
import poppy

from . import _core
from .math_module import ensure_np_array, xp


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
# Poppy-dependent functions.
#
# We intentionally implement these wrappers locally (instead of re-exporting
# lina.wfe.generate_wfe/opd) so we can normalize mixed numpy/cupy outputs from
# poppy and keep least-squares robust across CUDA runtime combinations.
# ---------------------------------------------------------------------------


def _to_xp_array(arr):
    return xp.asarray(ensure_np_array(arr))


def _lstsq_cpp(modes, data):
    modes_np = ensure_np_array(modes).astype(np.float64, copy=False)
    data_np = ensure_np_array(data).astype(np.float64, copy=False)
    return ensure_np_array(_core.lstsq(modes_np, data_np))


def generate_opd(
    npix=1000,
    oversample=1,
    wavelength=500 * u.nm,
    slope=2.5,
    seed=1234,
    rms=10,
    remove_modes=3,
):
    diam = 10 * u.mm
    wf = poppy.FresnelWavefront(
        beam_radius=diam / 2,
        npix=npix,
        oversample=oversample,
        wavelength=wavelength,
    )
    wfe_opd = poppy.StatisticalPSDWFE(
        index=slope,
        wfe=rms,
        radius=diam / 2,
        seed=int(seed),
    ).get_opd(wf)
    circ = poppy.CircularAperture(radius=diam / 2).get_transmission(wf)

    wfe_opd_np = ensure_np_array(wfe_opd)
    circ_np = ensure_np_array(circ)
    bmask = circ_np > 0

    if remove_modes > 0:
        Zs = ensure_np_array(
            poppy.zernike.arbitrary_basis(circ_np, nterms=remove_modes, outside=0)
        )
        Zc_opd = _lstsq_cpp(Zs, wfe_opd_np)
        for i in range(remove_modes):
            wfe_opd_np -= Zc_opd[i] * Zs[i]

    wfe_rms = np.sqrt(np.mean(np.square(wfe_opd_np[bmask])))
    wfe_opd_np *= rms.to_value(u.m) / wfe_rms
    return _to_xp_array(wfe_opd_np * circ_np)


def generate_wfe(
    npix=1000,
    oversample=1,
    wavelength=500 * u.nm,
    opd_index=2.5,
    amp_index=2.5,
    opd_seed=1234,
    amp_seed=12345,
    opd_rms=10 * u.nm,
    amp_rms=0.05,
    remove_opd_modes=3,
    remove_amp_modes=3,
):
    diam = 10 * u.mm
    wf = poppy.FresnelWavefront(
        beam_radius=diam / 2,
        npix=npix,
        oversample=oversample,
        wavelength=wavelength,
    )
    wfe_amp = poppy.StatisticalPSDWFE(
        index=amp_index,
        wfe=amp_rms * u.nm,
        radius=diam / 2,
        seed=int(amp_seed),
    ).get_opd(wf)
    wfe_opd = poppy.StatisticalPSDWFE(
        index=opd_index,
        wfe=opd_rms,
        radius=diam / 2,
        seed=int(opd_seed),
    ).get_opd(wf)
    circ = poppy.CircularAperture(radius=diam / 2).get_transmission(wf)

    wfe_amp_np = ensure_np_array(wfe_amp)
    wfe_opd_np = ensure_np_array(wfe_opd)
    circ_np = ensure_np_array(circ)
    bmask = circ_np > 0

    if remove_amp_modes > 0:
        Zs = ensure_np_array(
            poppy.zernike.arbitrary_basis(circ_np, nterms=remove_amp_modes, outside=0)
        )
        Zc_amp = _lstsq_cpp(Zs, wfe_amp_np)
        for i in range(remove_amp_modes):
            wfe_amp_np -= Zc_amp[i] * Zs[i]
    wfe_amp_np = wfe_amp_np * 1e9 + 1

    if remove_opd_modes > 0:
        Zs = ensure_np_array(
            poppy.zernike.arbitrary_basis(circ_np, nterms=remove_opd_modes, outside=0)
        )
        Zc_opd = _lstsq_cpp(Zs, wfe_opd_np)
        for i in range(remove_opd_modes):
            wfe_opd_np -= Zc_opd[i] * Zs[i]
    wfe_rms = np.sqrt(np.mean(np.square(wfe_opd_np[bmask])))
    wfe_opd_np *= opd_rms.to_value(u.m) / wfe_rms

    return _to_xp_array(wfe_amp_np * circ_np), _to_xp_array(wfe_opd_np * circ_np)


# Plotting + index helpers still come directly from lina.wfe.
import lina_cpp._pyref.wfe as _lina_wfe  # noqa: E402


def create_zernike_modes(pupil_mask, nmodes=15, remove_modes=0, return_np=False):
    if remove_modes > 0:
        nmodes += remove_modes
    zernikes = poppy.zernike.arbitrary_basis(
        ensure_np_array(pupil_mask),
        nterms=nmodes,
        outside=0,
    )[remove_modes:]
    zernikes = _to_xp_array(zernikes)
    if return_np:
        return ensure_np_array(zernikes)
    return zernikes


plot_psd = _lina_wfe.plot_psd
plot_psd_and_time_series = _lina_wfe.plot_psd_and_time_series
plot_time_series = _lina_wfe.plot_time_series
