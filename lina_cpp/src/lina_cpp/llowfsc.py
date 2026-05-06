"""lina_cpp.llowfsc - C++-backed math kernels + Python loop orchestration.

The math hot path of low-order WFS&C lives in ``cpp/src/llowfsc.cpp``:
* ``acquire_ref`` -- camera reference image (dark subtract, mask, flux normalize)
* ``reconstruct`` -- delta-image projection onto a control-matrix slice
* ``compute_zpo`` -- DM-command-driven zero-path-offset image accumulation
* ``loop_step``   -- one full closed-loop math iteration (returns delta DM cmd)

The closed-loop ``run()`` function stays in Python -- it orchestrates
camera reads, DM writes, and per-iteration logging via callables passed
in by the caller. This is the same orchestration as
``lina.llowfsc.run`` and is reused unchanged; only the math primitives
swap to the C++ implementations.

When you are ready to migrate this loop into a MagAOX C++ app, the
kernels in ``cpp/src/llowfsc.cpp`` are already written to be called
directly without Python in the loop -- you wire shmim reads/writes to
``lina::llowfsc::loop_step`` in the MagAOX app and you have a fully
native real-time controller.
"""

from __future__ import annotations

import copy
import time

import numpy as np

from . import _core
from .math_module import ensure_np_array


# ---------------------------------------------------------------------------
# Math kernels (delegate to C++)
# ---------------------------------------------------------------------------

def acquire_ref(take_im_fun, take_im_params, wfs_mask, camlo_dark=0.0, flux_norm=True):
    """Acquire a reference image: take a frame, dark-subtract, mask, normalize.

    Matches lina.llowfsc.acquire_ref signature. The math (subtract,
    mask multiply, sum, divide) runs in C++ via ``_core.llowfsc_acquire_ref``;
    the camera read goes through the user-supplied ``take_im_fun``.
    """
    camlo_ref_im = take_im_fun(**take_im_params)
    camlo_ref_im = ensure_np_array(camlo_ref_im).astype(np.float64, copy=False)
    mask = ensure_np_array(wfs_mask).astype(np.uint8, copy=False)

    # Dark can be a scalar OR an image; pass through unchanged.
    if isinstance(camlo_dark, (int, float)):
        dark_arg = float(camlo_dark)
    else:
        dark_arg = ensure_np_array(camlo_dark).astype(np.float64, copy=False)

    ref_image, flux_norm_coeff = _core.llowfsc_acquire_ref(
        camlo_ref_im, mask, dark_arg, bool(flux_norm)
    )
    if flux_norm:
        return ref_image, flux_norm_coeff
    return ref_image


def reconstruct(
    camlo_im,
    ref_im,
    wfs_mask,
    control_matrix,
    dark_im=0.0,
    modes=(0, 10),
    flux_norm=True,
    return_del_im=False,
):
    """Project a delta-image onto the [modes[0], modes[1]) rows of `control_matrix`.

    Matches lina.llowfsc.reconstruct signature. All math in C++.
    """
    camlo_im = ensure_np_array(camlo_im).astype(np.float64, copy=False)
    ref_im = ensure_np_array(ref_im).astype(np.float64, copy=False)
    mask = ensure_np_array(wfs_mask).astype(np.uint8, copy=False)
    C = ensure_np_array(control_matrix).astype(np.float64, copy=False)

    if isinstance(dark_im, (int, float)):
        dark_arg = float(dark_im)
    else:
        dark_arg = ensure_np_array(dark_im).astype(np.float64, copy=False)

    return _core.llowfsc_reconstruct(
        camlo_im, ref_im, mask, C,
        mode_lo=int(modes[0]), mode_hi=int(modes[1]),
        dark_im=dark_arg, flux_norm=bool(flux_norm),
        return_del_im=bool(return_del_im),
    )


def compute_zpo(
    DM_STREAMS,
    dm_mask,
    wfs_mask,
    response_matrix,
    dm_modal_matrix,
    ZPO_STREAM,
):
    """Accumulate ZPO image from DM command streams. Matches lina.llowfsc.compute_zpo.

    Each entry of ``DM_STREAMS`` is expected to expose a ``grab_latest()``
    method returning the current DM command as a 2D array; we extract
    the masked entries and pass them to the C++ kernel as a list of
    1D vectors.
    """
    dm_mask = ensure_np_array(dm_mask).astype(bool)
    wfs_mask_u8 = ensure_np_array(wfs_mask).astype(np.uint8, copy=False)
    R = ensure_np_array(response_matrix).astype(np.float64, copy=False)
    M = ensure_np_array(dm_modal_matrix).astype(np.float64, copy=False)

    cmds_masked = []
    for stream in DM_STREAMS:
        cmd = ensure_np_array(stream.grab_latest()).astype(np.float64, copy=False)
        cmds_masked.append(np.ascontiguousarray(cmd[dm_mask]))

    zpo = _core.llowfsc_compute_zpo(cmds_masked, wfs_mask_u8, R, M)
    if ZPO_STREAM is not None:
        ZPO_STREAM.write(zpo)
    return zpo


def loop_step(
    camlo_im,
    ref_plus_zpo,
    wfs_mask,
    control_matrix,
    dm_modes,
    gains,
    modes=(0, 10),
    ffo=None,
    dark_im=0.0,
    flux_norm=True,
):
    """One full per-iteration math step. Returns the delta DM command (2D).

    `dm_modes` may be either:
      - a (Nmodes, dm_rows, dm_cols) 3D array (lina convention), OR
      - a (Nmodes, dm_rows*dm_cols) 2D array (already flattened).

    `gains` is indexed relative to mode_lo (so gains[0] applies to
    mode `modes[0]`). `ffo` defaults to zero if None.
    """
    camlo_im = ensure_np_array(camlo_im).astype(np.float64, copy=False)
    ref_plus_zpo = ensure_np_array(ref_plus_zpo).astype(np.float64, copy=False)
    mask = ensure_np_array(wfs_mask).astype(np.uint8, copy=False)
    C = ensure_np_array(control_matrix).astype(np.float64, copy=False)
    M = ensure_np_array(dm_modes).astype(np.float64, copy=False)

    if M.ndim == 3:
        Nmodes, dm_rows, dm_cols = M.shape
        M_flat = M.reshape(Nmodes, dm_rows * dm_cols)
    else:
        Nmodes, Ndm = M.shape
        # We can't recover dm_rows/dm_cols from a 2D modes array; assume square.
        side = int(np.sqrt(Ndm))
        if side * side != Ndm:
            raise ValueError(
                "loop_step: cannot infer DM shape from flat modes; "
                "pass dm_modes as a 3D (Nmodes, rows, cols) array instead."
            )
        dm_rows = dm_cols = side
        M_flat = np.ascontiguousarray(M)

    g = ensure_np_array(gains).astype(np.float64, copy=False)
    nrange = int(modes[1]) - int(modes[0])
    if g.size == 1:
        g = np.full(nrange, float(g))
    if ffo is None:
        f = np.zeros(nrange)
    else:
        f = ensure_np_array(ffo).astype(np.float64, copy=False)

    if isinstance(dark_im, (int, float)):
        dark_arg = float(dark_im)
    else:
        dark_arg = ensure_np_array(dark_im).astype(np.float64, copy=False)

    return _core.llowfsc_loop_step(
        camlo_im, ref_plus_zpo, mask, C, np.ascontiguousarray(M_flat),
        int(dm_rows), int(dm_cols), g,
        int(modes[0]), int(modes[1]), f,
        dark_im=dark_arg, flux_norm=bool(flux_norm),
    )


# ---------------------------------------------------------------------------
# Closed-loop run() -- Python orchestration that calls the C++ kernels.
#
# Signature kept identical to lina.llowfsc.run so user code (notebooks,
# scripts) doesn't need to change. The math hot path now goes through
# the C++ extension via loop_step / reconstruct.
# ---------------------------------------------------------------------------

def run(
    take_im_fun,
    take_im_params,
    set_dm_fun,
    set_dm_params,
    get_dm_fun,
    get_dm_params,
    get_gains,
    ref_im,
    control_matrix,
    dm_modes,
    wfs_mask,
    dark_im=0.0,
    get_zpo=None,
    get_zpo_params=None,
    get_ffo=None,
    get_ffo_params=None,
):
    """One iteration of the closed loop. Use in a TimedThread / loop driver.

    Identical to lina.llowfsc.run but with the math hot path delegated
    to C++ via loop_step. Math: same. Signature: same. Behaviour: same.
    """
    if get_zpo_params is None: get_zpo_params = {}
    if get_ffo_params is None: get_ffo_params = {}

    camlo_im = take_im_fun(**take_im_params)

    zpo = get_zpo(**get_zpo_params) if get_zpo is not None else 0.0
    ref_plus_zpo = ref_im + zpo

    ffo = get_ffo(**get_ffo_params) if get_ffo is not None else None

    gains = get_gains()
    # The Python lina.llowfsc.run hardcodes modes=(0,10); preserve that.
    modes = (0, 10)
    del_dm_command = loop_step(
        camlo_im, ref_plus_zpo, wfs_mask, control_matrix, dm_modes,
        gains, modes=modes, ffo=ffo, dark_im=dark_im, flux_norm=True,
    )

    total_dm_command = get_dm_fun(**get_dm_params) + del_dm_command
    set_dm_fun(total_dm_command, **set_dm_params)


# ---------------------------------------------------------------------------
# calibrate_dm_modes still uses the user-supplied callables for camera /
# DM I/O; the math is just trivial differencing and dividing. We delegate
# to lina.llowfsc.calibrate_dm_modes (it's a thin loop, not a hot path).
# ---------------------------------------------------------------------------

from lina.llowfsc import (  # noqa: E402, F401
    ContinuousProcess,
    TimedThread,
    calibrate_dm_modes,
    plot_responses,
)
