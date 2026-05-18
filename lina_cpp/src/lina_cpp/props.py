"""lina_cpp.props - C++-backed mirror of lina.props.

All five propagation primitives are native, with both CPU and GPU
backends in the compiled extension:

* fft / ifft                 (centered FFT with fftshift/ifftshift)
* ang_spec                   (angular spectrum propagation)
* mft_forward / mft_reverse  (matrix Fourier transform)
* make_vortex_phase_mask     (LPM)
* get_fresnel_TF             (Fresnel defocus transfer function)

The Python signatures match lina.props exactly, with an additional
optional ``device='cpu'|'gpu'`` keyword on each function. When
``device`` is ``None`` (the default), the module-level default device
set by :func:`lina_cpp.set_device` is used. When ``device='gpu'`` but
the extension was built without CUDA, the call falls back to CPU with
a one-shot RuntimeWarning.

Examples
--------

    from lina_cpp.props import fft, mft_forward
    import lina_cpp

    lina_cpp.set_device("gpu")       # global default
    ft = fft(arr)                    # runs on GPU
    psf = mft_forward(arr, npix=512, npsf=128, psf_pixelscale_lamD=0.2)

    ft_cpu = fft(arr, device="cpu")  # override for one call
"""

from __future__ import annotations

import numpy as np

from . import _core
from ._dispatch import resolve_device
from .math_module import ensure_np_array


def _prep_complex(arr):
    """Coerce input to contiguous complex128 numpy. Matches what the
    pybind bindings expect (the C++ side casts to std::complex<double>).
    """
    arr = ensure_np_array(arr)
    return np.ascontiguousarray(arr, dtype=np.complex128)


# ---------------------------------------------------------------------------
# FFT primitives
# ---------------------------------------------------------------------------

def fft(arr, device=None):
    """Forward 2D FFT with the lina convention (ifftshift -> fft -> fftshift).

    Parameters
    ----------
    arr : array-like, shape (N, N)
        Complex input (real arrays are promoted to complex).
    device : {'cpu', 'gpu', None}, optional
        Backend to use. ``None`` uses the module-level default.
    """
    impl = resolve_device("fft", device)
    arr = _prep_complex(arr)
    return _core.fft_gpu(arr) if impl == "gpu" else _core.fft_cpu(arr)


def ifft(arr, device=None):
    """Inverse 2D FFT with the lina convention (ifftshift -> ifft -> fftshift).

    See :func:`fft` for arguments.
    """
    impl = resolve_device("ifft", device)
    arr = _prep_complex(arr)
    return _core.ifft_gpu(arr) if impl == "gpu" else _core.ifft_cpu(arr)


# Direct aliases for callers that already know exactly which backend
# they want (used by parity tests and microbenchmarks).
fft_cpu = _core.fft_cpu
ifft_cpu = _core.ifft_cpu
if hasattr(_core, "fft_gpu"):
    fft_gpu = _core.fft_gpu
    ifft_gpu = _core.ifft_gpu


# ---------------------------------------------------------------------------
# Wave propagation
# ---------------------------------------------------------------------------

def ang_spec(wavefront, wavelength, distance, pixelscale, device=None):
    """Angular-spectrum propagation by ``distance`` (meters)."""
    impl = resolve_device("ang_spec", device)
    arr = _prep_complex(wavefront)
    if impl == "gpu":
        return _core.ang_spec_gpu(arr, float(wavelength),
                                  float(distance), float(pixelscale))
    return _core.ang_spec(arr, float(wavelength),
                          float(distance), float(pixelscale))


def get_fresnel_TF(dz, N, wavelength, fnum, device=None):
    """Fresnel defocus transfer function. Matches lina.props.get_fresnel_TF."""
    impl = resolve_device("get_fresnel_TF", device)
    if impl == "gpu":
        return _core.get_fresnel_TF_gpu(float(dz), int(N),
                                        float(wavelength), float(fnum))
    return _core.get_fresnel_TF(float(dz), int(N),
                                float(wavelength), float(fnum))


# ---------------------------------------------------------------------------
# Vortex phase mask
# ---------------------------------------------------------------------------

def make_vortex_phase_mask(npix, charge=6, grid="odd", device=None, **kwargs):
    """Vortex phase mask of charge ``charge`` over an ``npix``-pixel grid.

    Falls back to the pure-Python ``lina.props.make_vortex_phase_mask``
    when the caller passes the optional ``singularity`` /
    ``focal_length`` / ``pupil_diameter`` parameters that the C++ side
    does not implement yet.
    """
    if kwargs:
        from lina.props import make_vortex_phase_mask as _impl
        return _impl(npix, charge=charge, grid=grid, **kwargs)
    impl = resolve_device("make_vortex_phase_mask", device)
    if impl == "gpu":
        return _core.make_vortex_phase_mask_gpu(int(npix), int(charge), str(grid))
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
    device=None,
):
    """Pupil -> focal-plane matrix Fourier transform.

    GPU path uses cuBLAS zgemm; CPU path uses an O(N^3) triple loop
    (which is still adequate at typical npix=N <= 2048).
    """
    impl = resolve_device("mft_forward", device)
    arr = _prep_complex(wavefront)
    if impl == "gpu":
        return _core.mft_forward_gpu(
            arr, int(npix), int(npsf), float(psf_pixelscale_lamD),
            str(convention), str(pp_centering), str(fp_centering),
        )
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
    device=None,
):
    """Focal-plane -> pupil matrix Fourier transform.

    GPU path uses cuBLAS zgemm; CPU path uses an O(N^3) triple loop.
    """
    impl = resolve_device("mft_reverse", device)
    arr = _prep_complex(fpwf)
    if impl == "gpu":
        return _core.mft_reverse_gpu(
            arr, float(psf_pixelscale_lamD), int(npix), int(N),
            str(convention), str(pp_centering), str(fp_centering),
        )
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
