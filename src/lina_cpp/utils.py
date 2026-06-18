"""lina_cpp.utils - C++-backed mirror of lina.utils.

Math hot paths (mean, rms, pad_or_crop, lstsq, tikhonov_inverse,
beta_reg, create_annular_mask, create_annular_focal_plane_mask,
make_grid) and FITS I/O delegate to the compiled extension. Plotting,
filesystem helpers, and Zernike basis construction are re-exported
from lina.utils because they have no native C++ implementation yet
(or never need one).
"""

from __future__ import annotations

import numpy as np

from . import _core
from .math_module import ensure_np_array, xp

# ---------------------------------------------------------------------------
# Math hot paths backed by C++.
# ---------------------------------------------------------------------------

def mean(array, mask=None):
    """Mean of `array`, optionally restricted to entries where mask is true."""
    array = ensure_np_array(array).astype(np.float64, copy=False)
    if mask is None:
        return _core.mean(array)
    mask_bool = ensure_np_array(mask).astype(bool, copy=False)
    try:
        return _core.mean_masked(
            np.ravel(array),
            np.ravel(mask_bool).astype(np.uint8, copy=False),
        )
    except Exception:
        # Keep parity with lina.utils.mean(arr[mask]) for odd mask shapes.
        return float(np.mean(array[mask_bool]))


def rms(array, mask=None):
    """RMS of `array`, optionally restricted to entries where mask is true."""
    array = ensure_np_array(array).astype(np.float64, copy=False)
    if mask is None:
        return _core.rms(array)
    # The C++ extension only exposes unmasked rms today; defer to the
    # numpy implementation for the masked path. Numerically identical
    # to lina.utils.rms(arr, mask=...).
    from lina_cpp._pyref.utils import rms as _impl
    return _impl(array, mask=mask)


def make_grid(npix, pixelscale=1, half_shift=False):
    """2D coordinate grid (x, y), centred on the array."""
    # Python lina.utils.make_grid returns a 3D array stacking [y, x] (or
    # equivalent). Match that shape exactly.
    centering = "even" if half_shift else "odd"
    cy, cx = _core.make_grid(int(npix), float(pixelscale), centering)
    cy = xp.asarray(cy)
    cx = xp.asarray(cx)
    return xp.stack([cy, cx], axis=0) if cy.ndim == 2 else (cy, cx)


def pad_or_crop(arr_in, npix):
    """Pad or center-crop a 2D array to `npix` x `npix`."""
    arr_in = ensure_np_array(arr_in)
    out = _core.pad_or_crop(arr_in.astype(np.float64, copy=False), int(npix))
    return xp.asarray(out)


def lstsq(modes, data):
    """Least-squares projection: solve modes @ coeffs = data."""
    out = _core.lstsq(
        ensure_np_array(modes).astype(np.float64, copy=False),
        ensure_np_array(data).astype(np.float64, copy=False),
    )
    return xp.asarray(out)


def tikhonov_inverse(A, rcond=1e-15, return_all=False, return_np=False):
    """SVD-based pseudo-inverse with Tikhonov regularisation."""
    A = ensure_np_array(A).astype(np.float64, copy=False)
    out = _core.tikhonov_inverse(A, float(rcond), bool(return_all))
    if return_np:
        return out
    if return_all:
        return tuple(xp.asarray(x) if hasattr(x, "shape") else x for x in out)
    return xp.asarray(out)


def beta_reg(S, beta=-1, return_np=False):
    """Beta-regularised control-matrix inverse of the Jacobian ``S``.

    Native C++/CUDA: on the GPU backend this runs as a cuBLAS GEMM + cuSOLVER
    Cholesky solve entirely on the device (no cupy, no CPU thrash); on the CPU
    backend it uses the C++ (OpenBLAS/LAPACKE) path. Returns ``control`` shaped
    (nmodes, nprobes*nmask).
    """
    from ._dispatch import get_device, gpu_available
    S_np = ensure_np_array(S).astype(np.float64, copy=False)
    if get_device() == "gpu" and gpu_available() and hasattr(_core, "beta_reg_gpu"):
        out = _core.beta_reg_gpu(S_np, float(beta))
    else:
        out = _core.beta_reg(S_np, float(beta))
    if return_np:
        return out
    return xp.asarray(out)


def create_annular_mask(
    N,
    pixelscale,
    irad,
    orad,
    edge=None,
    x_shift=0,
    y_shift=0,
    return_np=False,
    centering="odd",
    rotation=0,
):
    """Annular pupil/Lyot mask. Matches lina.utils.create_annular_mask signature.

    Note: the C++ binding does not yet expose `centering` or `return_np`;
    they are accepted for API compatibility but the C++ path always uses
    'odd' centering and returns a numpy array. If you need 'even' centering
    today, fall back to lina.utils.create_annular_mask.
    """
    if centering != "odd":
        from lina_cpp._pyref.utils import create_annular_mask as _impl
        return _impl(
            N, pixelscale, irad, orad,
            edge=edge, x_shift=x_shift, y_shift=y_shift,
            return_np=return_np, centering=centering, rotation=rotation,
        )
    out = _core.create_annular_mask(
        int(N), float(pixelscale), float(irad), float(orad),
        edge=(None if edge is None else float(edge)),
        x_shift=float(x_shift), y_shift=float(y_shift),
        rotation=float(rotation),
    )
    out = out.astype(bool, copy=False)
    if return_np:
        return out
    return xp.asarray(out)


def create_annular_focal_plane_mask(
    npsf,
    psf_pixelscale,
    irad,
    orad,
    edge=None,
    centering="odd",
    rotation=0,
    x_shift=0,
    y_shift=0,
    return_np=False,
):
    """Focal-plane annular mask in lambda/D units. Matches lina.utils.* signature."""
    out = _core.create_annular_focal_plane_mask(
        int(npsf), float(psf_pixelscale), float(irad), float(orad),
        edge=(None if edge is None else float(edge)),
        centering=str(centering),
        rotation=float(rotation),
        x_shift=float(x_shift), y_shift=float(y_shift),
    )
    out = out.astype(bool, copy=False)
    if return_np:
        return out
    return xp.asarray(out)


def create_circ_mask(h, w, center=None, radius=None):
    """Pure-Python circular mask. No C++ port; defer to lina."""
    from lina_cpp._pyref.utils import create_circ_mask as _impl
    return _impl(h, w, center=center, radius=radius)


# ---------------------------------------------------------------------------
# FITS I/O - native C++ via cfitsio.
# ---------------------------------------------------------------------------

def save_fits(fpath, data, header=None, ow=True, quiet=False):
    """Write `data` (2D ndarray) as the primary HDU of a FITS file.

    Drop-in replacement for ``lina.utils.save_fits``.
    """
    arr = ensure_np_array(data)
    # Stringify header values for the cfitsio C path.
    if header is not None:
        header = {str(k): str(v) for k, v in header.items()}
    # C++ FITS writer currently expects a 2D primary image. For vectors or
    # higher-D arrays, fall back to the Python implementation to preserve
    # lina API behavior used by benchmark scripts.
    if getattr(arr, "ndim", 0) != 2:
        from lina_cpp._pyref.utils import save_fits as _save_fits_py
        return _save_fits_py(fpath, arr, header=header, ow=ow, quiet=quiet)
    return _core.save_fits(str(fpath), arr, header=header, ow=bool(ow), quiet=bool(quiet))


def load_fits(fpath, header=False):
    """Load 2D primary HDU. If header=True, return (data, header_dict)."""
    return _core.load_fits(str(fpath), bool(header))


def fits_available():
    """True if the C++ extension was built with cfitsio support."""
    return _core.fits_available()


# ---------------------------------------------------------------------------
# Re-exports from lina.utils (plotting, fs helpers, Zernike basis,
# rotation/interpolation that depend on scipy.ndimage, radial-contrast
# reductions, etc.). These either have no math benefit from C++ or
# require porting that we deferred to a follow-up.
# ---------------------------------------------------------------------------

from lina_cpp._pyref.utils import (  # noqa: E402, F401
    beta_reg as _lina_beta_reg,
    create_zernike_modes,
    delete_files,
    get_fnames,
    get_radial_contrast,
    get_radial_dist,
    imshow,
    interp_arr,
    load_pickle,
    make_dir,
    move_files,
    plot_radial_contrast,
    rotate_arr,
    save_pickle,
    tt_as_to_rms,
    tt_rms_to_as,
)
