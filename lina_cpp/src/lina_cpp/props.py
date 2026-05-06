"""lina_cpp.props - C++-backed mirror of lina.props.

All five propagation primitives are native:
* fft / ifft (centered FFT with fftshift / ifftshift wrapping)
* ang_spec  (angular spectrum propagation)
* mft_forward / mft_reverse (matrix Fourier transform)
* make_vortex_phase_mask
* get_fresnel_TF

The Python signatures match lina.props exactly so user code that does
``from lina.props import fft, mft_forward`` keeps working when swapped
for ``from lina_cpp.props import fft, mft_forward``.
"""

from __future__ import annotations

import numpy as np

from . import _core
from .math_module import ensure_np_array

# ---------------------------------------------------------------------------
# FFT primitives
# ---------------------------------------------------------------------------

def fft(arr):
    """Forward 2D FFT with the lina convention (ifftshift -> fft -> fftshift)."""
    arr = ensure_np_array(arr)
    if not np.iscomplexobj(arr):
        arr = arr.astype(np.complex128, copy=False)
    else:
        arr = arr.astype(np.complex128, copy=False)
    return _core.fft_cpu(arr)


def ifft(arr):
    """Inverse 2D FFT with the lina convention (ifftshift -> ifft -> fftshift)."""
    arr = ensure_np_array(arr)
    if not np.iscomplexobj(arr):
        arr = arr.astype(np.complex128, copy=False)
    else:
        arr = arr.astype(np.complex128, copy=False)
    return _core.ifft_cpu(arr)


# Aliases for backward-compat with code that already calls _cpu suffix.
fft_cpu = _core.fft_cpu
ifft_cpu = _core.ifft_cpu


# ---------------------------------------------------------------------------
# Wave propagation
# ---------------------------------------------------------------------------

def ang_spec(wavefront, wavelength, distance, pixelscale):
    """Angular-spectrum propagation by `distance` (meters)."""
    arr = np.ascontiguousarray(ensure_np_array(wavefront), dtype=np.complex128)
    return _core.ang_spec(arr, float(wavelength), float(distance), float(pixelscale))


def get_fresnel_TF(dz, N, wavelength, fnum):
    """Fresnel defocus transfer function. Matches lina.props.get_fresnel_TF."""
    return _core.get_fresnel_TF(float(dz), int(N), float(wavelength), float(fnum))


# ---------------------------------------------------------------------------
# Vortex phase mask
# ---------------------------------------------------------------------------

def make_vortex_phase_mask(npix, charge=6, grid="odd", **kwargs):
    """Vortex phase mask of charge `charge` over an `npix`-pixel grid."""
    # The Python lina.props.make_vortex_phase_mask takes additional
    # optional keyword args (singularity, focal_length, pupil_diameter).
    # Our C++ version implements the most common case (charge + grid).
    # Fall back to lina if the caller passed additional kwargs we
    # don't yet handle.
    if kwargs:
        from lina.props import make_vortex_phase_mask as _impl
        return _impl(npix, charge=charge, grid=grid, **kwargs)
    return _core.make_vortex_phase_mask(int(npix), int(charge), str(grid))


# ---------------------------------------------------------------------------
# Matrix Fourier transform
# ---------------------------------------------------------------------------

def mft_forward(
    wavefront,
    npix,
    npsf,
    psf_pixelscale_lamD,
    convention="-",
    pp_centering="odd",
    fp_centering="odd",
):
    """Pupil -> focal-plane matrix Fourier transform."""
    arr = np.ascontiguousarray(ensure_np_array(wavefront), dtype=np.complex128)
    return _core.mft_forward(
        arr, int(npix), int(npsf), float(psf_pixelscale_lamD),
        str(convention), str(pp_centering), str(fp_centering),
    )


def mft_reverse(
    fpwf,
    psf_pixelscale_lamD,
    npix,
    N,
    convention="+",
    pp_centering="odd",
    fp_centering="odd",
):
    """Focal-plane -> pupil matrix Fourier transform."""
    arr = np.ascontiguousarray(ensure_np_array(fpwf), dtype=np.complex128)
    return _core.mft_reverse(
        arr, float(psf_pixelscale_lamD), int(npix), int(N),
        str(convention), str(pp_centering), str(fp_centering),
    )


# ---------------------------------------------------------------------------
# Things lina.props provides that aren't in C++ yet -- delegate to lina.
# ---------------------------------------------------------------------------

from lina.props import (  # noqa: E402, F401
    get_scaled_coords,
    make_mft_forward_matrices,
    make_mft_reverse_matrices,
)
