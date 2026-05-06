"""lina_cpp.coro_utils - hardware control re-exported from lina.coro_utils.

The 28 hardware control functions (move_psf, set_zwo_*, set_cam_*,
home_block, set_dm/add_dm/zero_dm, ...) all wrap the Python INDI client
(purepyindi/purepyindi2). A native C++ port using the MagAOX
``pcf::IndiClient`` from ``MagAOX-scoob/INDI/libcommon`` is planned as a
follow-up; until then we re-export the Python implementations so a
caller can do::

    import lina_cpp as lina  # the C++-backed package
    lina.coro_utils.move_psf(...)  # calls lina.coro_utils under the hood

The math hot paths (normalize_coro_im, compute_contrast) ARE native C++
in :mod:`lina_cpp._core` and are wrapped here to delegate to the
extension while keeping the Python signature.
"""

from __future__ import annotations

import numpy as np

from . import _core
from .math_module import ensure_np_array


# ---------------------------------------------------------------------------
# Math hot paths -- native C++.
# ---------------------------------------------------------------------------

def normalize_coro_im(raw_im, im_params, ref_params, dark_im=0.0, verbose=True):
    """Normalize a coronagraph image by exposure-time and reference flux."""
    raw_arr = ensure_np_array(raw_im).astype(np.float64, copy=False)
    if not np.isscalar(dark_im):
        dark_im = ensure_np_array(dark_im).astype(np.float64, copy=False)
    return _core.normalize_coro_im(raw_arr, im_params, ref_params, dark_im, bool(verbose))


def compute_contrast(ni_im, mask, verbose=True):
    """Mean(ni_im[mask]) - the mean intensity inside the dark-hole mask."""
    return _core.compute_contrast(
        ensure_np_array(ni_im).astype(np.float64, copy=False),
        ensure_np_array(mask).astype(np.uint8, copy=False),
        bool(verbose),
    )


# ---------------------------------------------------------------------------
# Hardware control -- re-exported unchanged from lina.coro_utils. These will
# be replaced with native C++ INDI calls in a follow-up port.
# ---------------------------------------------------------------------------

from lina.coro_utils import (  # noqa: E402, F401
    add_dm,
    get_cam_exp_time,
    get_cam_gain,
    get_fiber_atten,
    get_im_params,
    home_block,
    home_filter_stage,
    measure_waffle_center_and_angle,
    move_block_in,
    move_block_out,
    move_psf,
    set_cam_blacklevel,
    set_cam_exp_time,
    set_cam_gain,
    set_cam_roi,
    set_dm,
    set_fiber_atten,
    set_filter_stage_position,
    set_filter_stage_velocity,
    set_zwo_bin,
    set_zwo_roi,
    snap_ni,
    switch_filter_stage,
    zero_dm,
)
