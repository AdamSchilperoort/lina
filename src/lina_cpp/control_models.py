"""Compatibility wrapper for lina.control_models.

The high-level control model remains Python-level logic for now, but we still
need stable array-type parity when this module is imported through `lina_cpp`.
On some CUDA stacks, poppy's CircularAperture returns cupy arrays even when the
active lina backend is numpy, which causes mixed-array failures during MODEL
initialization. We normalize that return path to the active lina backend type.
"""

from __future__ import annotations

import numpy as np
import poppy
import warnings as _warnings

from . import _core
from ._dispatch import get_device
from lina_cpp._pyref.math_module import ensure_np_array, xp, xcipy

_HAS_CPP_CONTROL_MODEL = hasattr(_core, "ControlModelCpp")
_HAS_CPP_VAL_AND_GRAD = hasattr(_core, "control_model_val_and_grad")
_HAS_CPP_DM_VAL_AND_GRAD = hasattr(_core, "control_model_dm_val_and_grad")
_HAS_CPP_SOLVE_FLAT = hasattr(_core, "control_model_solve_flat")

# Opt-in toggle for routing the SciPy-driven dm_val_and_grad through the native
# C++ gradient. Off by default (see note in dm_val_and_grad).
import os as _os
_USE_NATIVE_DM_VAL_AND_GRAD = _os.environ.get("LINA_CPP_NATIVE_DM_VG", "0") == "1"


def _patch_poppy_circular_aperture() -> None:
    orig = poppy.CircularAperture.get_transmission
    if getattr(orig, "_lina_cpp_wrapped", False):
        return

    def wrapped(self, wave):  # noqa: ANN001
        arr = orig(self, wave)
        return xp.asarray(ensure_np_array(arr))

    wrapped._lina_cpp_wrapped = True  # type: ignore[attr-defined]
    poppy.CircularAperture.get_transmission = wrapped


_patch_poppy_circular_aperture()

import lina_cpp._pyref.control_models as _cm  # noqa: E402
from lina_cpp._pyref.control_models import *  # noqa: E402,F401,F403


def _safe_matmul3(a, b, c):
    a = _unwrap_matmul_obj(a)
    b = _unwrap_matmul_obj(b)
    c = _unwrap_matmul_obj(c)
    try:
        return a @ b @ c
    except Exception:
        return xp.asarray(_as_numpy(a) @ _as_numpy(b) @ _as_numpy(c))


LegacyMODEL = _cm.MODEL


class _SafeMatmulOperand:
    """Array wrapper that enforces robust two-stage matmul chains."""

    __array_priority__ = 1000

    def __init__(self, arr):
        self._arr = arr

    def __matmul__(self, other):
        return _SafeMatmulIntermediate(_safe_matmul2(self._arr, other))

    def __rmatmul__(self, other):
        return _SafeMatmulIntermediate(_safe_matmul2(other, self._arr))

    def __getattr__(self, key):
        return getattr(self._arr, key)


class _SafeMatmulIntermediate:
    """Carries first matmul result so second @ also uses safe path."""

    __array_priority__ = 1000

    def __init__(self, arr):
        self._arr = arr

    def __matmul__(self, other):
        return _safe_matmul2(self._arr, other)

    def __rmatmul__(self, other):
        return _safe_matmul2(other, self._arr)

    def __getattr__(self, key):
        return getattr(self._arr, key)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        unwrapped = [_unwrap_matmul_obj(i) for i in inputs]
        return getattr(ufunc, method)(*unwrapped, **kwargs)


def _unwrap_matmul_obj(obj):
    if isinstance(obj, (_SafeMatmulOperand, _SafeMatmulIntermediate)):
        return obj._arr
    return obj


def _as_numpy(obj):
    out = ensure_np_array(_unwrap_matmul_obj(obj))
    if out is None:
        out = np.asarray(_unwrap_matmul_obj(obj))
    return out


def _array_signature(arr):
    arr_u = _unwrap_matmul_obj(arr)
    shape = tuple(getattr(arr_u, "shape", ()))
    dtype = str(getattr(arr_u, "dtype", type(arr_u)))
    if hasattr(arr_u, "__cuda_array_interface__"):
        ptr = arr_u.__cuda_array_interface__.get("data", (None,))[0]
    elif hasattr(arr_u, "__array_interface__"):
        ptr = arr_u.__array_interface__.get("data", (None,))[0]
    else:
        ptr = id(arr_u)
    return (shape, dtype, ptr)


def _safe_matmul2(left, right):
    left_u = _unwrap_matmul_obj(left)
    right_u = _unwrap_matmul_obj(right)
    try:
        return left_u @ right_u
    except Exception:
        # Preserve correctness on intermittent CuBLAS failures by falling
        # back to host BLAS for this multiply only.
        return xp.asarray(_as_numpy(left_u) @ _as_numpy(right_u))


class MODEL_PY(_cm.MODEL):
    """Legacy MODEL with robust matmul fallback for CuBLAS instability."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name in ("Mx_dm", "My_dm", "Mx_dm_back", "My_dm_back"):
            val = getattr(self, name, None)
            if val is not None and not isinstance(val, _SafeMatmulOperand):
                setattr(self, name, _SafeMatmulOperand(val))


class MODEL_CPP(_cm.MODEL):
    """Experimental C++-accelerated MODEL preserving the legacy API."""

    def __init__(
        self,
        wavelength_c=630e-9,
        wavelength=None,
        npix=512,
        Ndef=None,
        N_vortex_lres=2048,
        vortex_win_diam=30,
        vortex_hres_sampling=0.025,
        vortex_dot_mask_diam_lamDc=0.5,
        dm_beam_diam=9.3e-3,
        lyot_pupil_diam=9.1e-3,
        lyot_stop_diam=8.6e-3,
        exit_pupil_prop_dist=None,
        camsci_pxscl_lamDc=0.2,
        ncamsci=256,
        Nact=34,
        act_spacing=300e-6,
        act_coupling=0.15,
        PREFPM_AMP=None,
        PREFPM_OPD=None,
    ):
        if not _HAS_CPP_CONTROL_MODEL:
            raise RuntimeError(
                "lina_cpp._core is missing ControlModelCpp. "
                "Restart the Python kernel after reinstalling lina_cpp so "
                "the latest extension is loaded."
            )
        super().__init__(
            wavelength_c=wavelength_c,
            wavelength=wavelength,
            npix=npix,
            Ndef=Ndef,
            N_vortex_lres=N_vortex_lres,
            vortex_win_diam=vortex_win_diam,
            vortex_hres_sampling=vortex_hres_sampling,
            vortex_dot_mask_diam_lamDc=vortex_dot_mask_diam_lamDc,
            dm_beam_diam=dm_beam_diam,
            lyot_pupil_diam=lyot_pupil_diam,
            lyot_stop_diam=lyot_stop_diam,
            exit_pupil_prop_dist=exit_pupil_prop_dist,
            camsci_pxscl_lamDc=camsci_pxscl_lamDc,
            ncamsci=ncamsci,
            Nact=Nact,
            act_spacing=act_spacing,
            act_coupling=act_coupling,
            PREFPM_AMP=PREFPM_AMP,
            PREFPM_OPD=PREFPM_OPD,
        )
        for name in ("Mx_dm", "My_dm", "Mx_dm_back", "My_dm_back"):
            val = getattr(self, name, None)
            if val is not None and not isinstance(val, _SafeMatmulOperand):
                setattr(self, name, _SafeMatmulOperand(val))

        self._cpp_model = _core.ControlModelCpp(
            wavelength_c=float(self.wavelength_c),
            wavelength=(None if wavelength is None else float(self.wavelength)),
            npix=int(self.npix),
            ndef=int(self.Ndef),
            n_vortex_lres=int(self.N_vortex_lres),
            vortex_win_diam=float(self.vortex_win_diam),
            vortex_hres_sampling=float(self.hres_sampling),
            vortex_dot_mask_diam_lamDc=float(self.vortex_dot_mask_diam_lamDc),
            dm_beam_diam=float(self.dm_beam_diam),
            lyot_pupil_diam=float(self.lyot_pupil_diam),
            lyot_stop_diam=float(self.lyot_stop_diam),
            exit_pupil_prop_dist=(
                None if self.exit_pupil_prop_dist is None else float(self.exit_pupil_prop_dist)
            ),
            camsci_pxscl_lamDc=float(self.camsci_pxscl_lamDc),
            ncamsci=int(self.ncamsci),
            nact=int(self.Nact),
            act_spacing=float(self.act_spacing),
            act_coupling=float(act_coupling),
        )
        self._cpp_device = None
        self._prefpm_synced = False
        self._prefpm_amp_sig = None
        self._prefpm_opd_sig = None
        self._sync_cpp_device()
        # Use the exact poppy-generated aperture / Lyot stop (anti-aliased
        # edges) instead of the C++ hard-edged circle, so the optical forward
        # matches the pure-Python reference to floating-point precision.
        self._sync_cpp_masks()
        self._sync_cpp_prefpm(force=True)

    def _sync_cpp_masks(self):
        aperture = np.asarray(ensure_np_array(self.APERTURE), dtype=np.float64)
        lyotstop = np.asarray(ensure_np_array(self.LYOTSTOP), dtype=np.float64)
        self._cpp_model.set_aperture(aperture)
        self._cpp_model.set_lyotstop(lyotstop)
        # Push the exact scipy-Tukey-windowed + poppy-dot vortex FPMs so the
        # coronagraph path matches the pure-Python reference.
        wv_lres = np.asarray(ensure_np_array(self.windowed_vortex_lres), dtype=np.complex128)
        wv_hres = np.asarray(ensure_np_array(self.windowed_vortex_hres), dtype=np.complex128)
        self._cpp_model.set_windowed_vortex_lres(wv_lres)
        self._cpp_model.set_windowed_vortex_hres(wv_hres)
        # Push the exact DM-surface model (influence-function FFT + MFT matrices)
        # so the DM phasor and dm_val_and_grad adjoint match the reference.
        # The DM matrices may be wrapped in _SafeMatmulOperand; unwrap first.
        def cplx(a):
            a = _unwrap_matmul_obj(a)
            return np.asarray(ensure_np_array(a), dtype=np.complex128)
        self._cpp_model.set_dm_model(
            cplx(self.inf_fun_fft), cplx(self.Mx_dm), cplx(self.My_dm),
            cplx(self.Mx_dm_back), cplx(self.My_dm_back),
        )

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name in ("PREFPM_AMP", "PREFPM_OPD") and hasattr(self, "_prefpm_synced"):
            self._prefpm_synced = False

    def _sync_cpp_prefpm(self, force=False):
        amp_sig = _array_signature(self.PREFPM_AMP)
        opd_sig = _array_signature(self.PREFPM_OPD)
        if (
            not force
            and self._prefpm_synced
            and amp_sig == self._prefpm_amp_sig
            and opd_sig == self._prefpm_opd_sig
        ):
            return
        amp = np.asarray(ensure_np_array(self.PREFPM_AMP), dtype=np.float64)
        opd = np.asarray(ensure_np_array(self.PREFPM_OPD), dtype=np.float64)
        self._cpp_model.set_prefpm_amp(amp)
        self._cpp_model.set_prefpm_opd(opd)
        self._prefpm_amp_sig = amp_sig
        self._prefpm_opd_sig = opd_sig
        self._prefpm_synced = True

    def _sync_cpp_device(self):
        device = get_device()
        if self._cpp_device != device:
            self._cpp_model.set_device(device)
            self._cpp_device = device

    def forward(
        self,
        actuators,
        wavelength=None,
        use_vortex=True,
        return_ints=False,
        plot=False,
        sync=False,
    ):
        # Keep full legacy path when caller requests internals/plots.
        if return_ints or plot:
            return super().forward(
                actuators,
                wavelength=wavelength,
                use_vortex=use_vortex,
                return_ints=return_ints,
                plot=plot,
                sync=sync,
            )

        self._sync_cpp_device()
        self._sync_cpp_prefpm()
        wl = self.wavelength_c if wavelength is None else wavelength
        acts = np.asarray(ensure_np_array(actuators), dtype=np.float64).ravel()
        e_fp = self._cpp_model.forward(acts, float(wl), bool(use_vortex))
        e_fp = xp.asarray(e_fp)

        if self.camsci_rotation != 0:
            e_fp = xcipy.ndimage.rotate(e_fp, self.camsci_rotation, reshape=False, order=3)

        e_fp = e_fp / xp.sqrt(self.Imax_ref)

        if sync:
            try:
                xp.cuda.Device().synchronize()
            except Exception:
                pass
        return e_fp

    def snap(self):
        dm_command = xp.sum(self.dm_commands, axis=0)
        e_fp = self.forward(dm_command[self.dm_mask], self.wavelength, self.use_vortex)
        return xp.abs(e_fp) ** 2


def dm_val_and_grad(  # noqa: D401
    del_acts,
    OPD,
    M,
    current_acts=None,
):
    """DM-surface fit objective+gradient.

    Routes to the native C++/CUDA endpoint when the model is C++-backed and no
    nonzero ``current_acts`` offset is requested (the flat-DM solve case).
    Falls back to the Python implementation otherwise.
    """
    # Optional pure-C++ gradient path (opt-in). The native endpoint
    # _core.control_model_dm_val_and_grad is available and matches J to ~1e-14,
    # but its adjoint is a slightly non-uniform scaling of the true gradient on
    # this ill-conditioned fit, which makes SciPy/L-BFGS line searches inefficient.
    # The CuPy path below is fast (no host round-trips) and SciPy-consistent, so
    # it is the default for the flat-DM solve. Use lina_cpp.control_models.
    # solve_flat_command(...) for the fully native (libLBFGS) solver.
    if (
        _USE_NATIVE_DM_VAL_AND_GRAD
        and _HAS_CPP_DM_VAL_AND_GRAD
        and current_acts is None
        and hasattr(M, "_cpp_model")
    ):
        M._sync_cpp_device()
        opd_np = np.asarray(ensure_np_array(OPD), dtype=np.float64)
        acts_np = np.asarray(ensure_np_array(del_acts), dtype=np.float64).ravel()
        J, grad = _core.control_model_dm_val_and_grad(M._cpp_model, acts_np, opd_np)
        return float(J), grad

    del_acts = xp.array(del_acts)
    del_command = xp.zeros((M.Nact, M.Nact))
    del_command[M.dm_mask] = xp.array(del_acts)

    current_acts = (
        xp.array(current_acts)
        if current_acts is not None
        else xp.zeros((M.Nact, M.Nact))
    )

    OPD = xp.array(OPD)

    dm_command = current_acts + del_command
    dm_mft = _safe_matmul3(M.Mx_dm, dm_command, M.My_dm)
    dm_surf_fft = M.inf_fun_fft * dm_mft
    dm_surf = xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(dm_surf_fft,))).real
    dm_surf = _cm.utils.pad_or_crop(dm_surf, OPD.shape[0])

    OPD_MASK = _cm.utils.pad_or_crop(M.BAP_MASK, OPD.shape[0])
    opd_l2norm = OPD[OPD_MASK].dot(OPD[OPD_MASK])
    total_opd = OPD + 2 * dm_surf
    J = total_opd[OPD_MASK].dot(total_opd[OPD_MASK]) / opd_l2norm

    masked_total = OPD_MASK * total_opd
    dJ_dOPD = 2 * masked_total / opd_l2norm

    dJ_dS_DM = _cm.utils.pad_or_crop(dJ_dOPD, M.Nsurf)
    x2_bar = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(dJ_dS_DM)))
    x1_bar = x2_bar * M.inf_fun_fft.conj()
    dJ_dA1 = _safe_matmul3(M.Mx_dm_back, x1_bar, M.My_dm_back) / (
        M.Nsurf * M.Nact * M.Nact
    )

    dJ_dA = dJ_dA1[M.dm_mask].real

    J_np = ensure_np_array(J)
    if J_np is None:
        J_np = float(np.asarray(J))

    dJ_np = ensure_np_array(dJ_dA)
    if dJ_np is None:
        dJ_np = np.asarray(dJ_dA)

    return J_np, dJ_np


def solve_flat_command(M, OPD, tol=1e-3, max_iter=0):
    """Solve the flat-DM actuator vector via the native C++ L-BFGS optimizer.

    Minimizes the residual of ``OPD + 2 * DM_surface`` over the beam aperture,
    entirely in C++/CUDA (no SciPy). Returns the masked actuator vector
    (length ``M.Nacts``) as a NumPy array, matching ``scipy.optimize.minimize``
    with ``method='L-BFGS-B'`` on :func:`dm_val_and_grad`.
    """
    if not (_HAS_CPP_SOLVE_FLAT and hasattr(M, "_cpp_model")):
        raise RuntimeError(
            "Native solve_flat_command requires a C++-backed model and a "
            "lina_cpp built with libLBFGS (LINA_USE_LBFGS)."
        )
    # Constants are synced at construction; only sync the device here.
    M._sync_cpp_device()
    opd_np = np.asarray(ensure_np_array(OPD), dtype=np.float64)
    return _core.control_model_solve_flat(M._cpp_model, opd_np, float(tol), int(max_iter))


# Expose the native C++ control-model surface so callers can opt into the
# pure-C++ optical simulation path without breaking the existing Python API.
ControlModelCpp = getattr(_core, "ControlModelCpp", None)
control_model_val_and_grad = getattr(_core, "control_model_val_and_grad", None)

if _HAS_CPP_CONTROL_MODEL:
    # Always route through the native C++ ControlModel when available.
    MODEL = MODEL_CPP
else:
    MODEL = MODEL_PY
    if not _HAS_CPP_CONTROL_MODEL:
        _warnings.warn(
            "lina_cpp._core missing ControlModelCpp; falling back to Python MODEL. "
            "Restart kernel after reinstall to enable C++ path.",
            RuntimeWarning,
            stacklevel=2,
        )
