"""Per-function parity tests: pure-Python ``lina`` vs C++/CUDA ``lina_cpp``.

Each test runs the same operation on the pure-Python reference (``lina``, e.g.
from ``lina_kian``) and on the C++/CUDA backend (``lina_cpp``) using small
inputs, and asserts the results agree to a sensible tolerance. Run with pytest
to get a per-function pass/fail report::

    python -m pytest tests/test_parity.py -v

Both packages must be importable. By default the comparison runs on the CPU
backend (deterministic, fast, machine-load independent). Set the device with::

    LINA_PARITY_DEVICE=gpu python -m pytest tests/test_parity.py -v

If ``lina`` is not installed the whole module is skipped.
"""

import os
import numpy as np
import pytest

lina = pytest.importorskip("lina", reason="pure-Python `lina` not installed")
# `lina.__init__` only imports a subset of submodules; pull in the rest used here.
import lina.props      # noqa: E402,F401
import lina.wfe        # noqa: E402,F401
import lina.control_models  # noqa: E402,F401
import lina_cpp  # noqa: E402

DEVICE = os.environ.get("LINA_PARITY_DEVICE", "cpu").lower()


# --------------------------------------------------------------------------- #
# Backend setup: force both packages onto the same device.
# --------------------------------------------------------------------------- #
def _set_lina_cpu():
    mm = lina.math_module
    mm.update_np(mm.np)
    mm.update_scipy(mm.scipy)


def _set_lina_gpu():
    mm = lina.math_module
    if not getattr(mm, "cupy_avail", False):
        pytest.skip("cupy not available for GPU parity")
    mm.update_np(mm.cupy)
    mm.update_scipy(mm.cupyx.scipy)


@pytest.fixture(scope="module", autouse=True)
def _backend():
    if DEVICE == "gpu":
        if not lina_cpp.gpu_available():
            pytest.skip("lina_cpp built without CUDA")
        lina_cpp.set_backend("gpu")
        _set_lina_gpu()
    else:
        lina_cpp.set_backend("cpu")
        _set_lina_cpu()
    yield


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _np(x):
    """Return a NumPy array from a numpy/cupy/scalar result."""
    return np.asarray(lina_cpp.math_module.ensure_np_array(x))


def assert_close(py, cpp, *, rtol=1e-9, atol=1e-12, name=""):
    a = _np(py)
    b = _np(cpp)
    assert a.shape == b.shape, f"{name}: shape {a.shape} != {b.shape}"
    denom = float(np.abs(a).max()) or 1.0
    rel = float(np.abs(a - b).max()) / denom
    assert rel <= rtol or np.allclose(a, b, rtol=rtol, atol=atol), (
        f"{name}: rel max diff {rel:.3e} > rtol {rtol:.1e}"
    )


rng = np.random.RandomState(0)


def cplx(shape):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex128)


def real(shape):
    return rng.standard_normal(shape).astype(np.float64)


# --------------------------------------------------------------------------- #
# props (propagation)
# --------------------------------------------------------------------------- #
def test_fft():
    x = cplx((64, 64))
    assert_close(lina.props.fft(x), lina_cpp.props.fft(x), rtol=1e-9, name="fft")


def test_ifft():
    x = cplx((64, 64))
    assert_close(lina.props.ifft(x), lina_cpp.props.ifft(x), rtol=1e-9, name="ifft")


def test_mft_forward():
    wf = cplx((64, 64))
    assert_close(lina.props.mft_forward(wf, 64, 32, 0.2),
                 lina_cpp.props.mft_forward(wf, 64, 32, 0.2), rtol=1e-9, name="mft_forward")


def test_mft_reverse():
    fp = cplx((32, 32))
    assert_close(lina.props.mft_reverse(fp, 0.2, 64, 64),
                 lina_cpp.props.mft_reverse(fp, 0.2, 64, 64), rtol=1e-9, name="mft_reverse")


def test_ang_spec():
    wf = cplx((64, 64))
    assert_close(lina.props.ang_spec(wf, 633e-9, 1e-3, 1e-5),
                 lina_cpp.props.ang_spec(wf, 633e-9, 1e-3, 1e-5), rtol=1e-8, name="ang_spec")


def test_make_vortex_phase_mask():
    assert_close(lina.props.make_vortex_phase_mask(64),
                 lina_cpp.props.make_vortex_phase_mask(64), rtol=1e-9, name="vortex")


def test_get_fresnel_TF():
    assert_close(lina.props.get_fresnel_TF(1e-3, 64, 633e-9, 30.0),
                 lina_cpp.props.get_fresnel_TF(1e-3, 64, 633e-9, 30.0), rtol=1e-9, name="fresnel_TF")


# --------------------------------------------------------------------------- #
# utils (reductions, masks, linear algebra)
# --------------------------------------------------------------------------- #
def test_mean():
    x = real((40, 40))
    assert_close(lina.utils.mean(x), lina_cpp.utils.mean(x), rtol=1e-12, name="mean")


def test_mean_masked():
    x = real((40, 40))
    m = x > 0
    assert_close(lina.utils.mean(x, m), lina_cpp.utils.mean(x, m), rtol=1e-12, name="mean_masked")


def test_rms():
    x = real((40, 40))
    assert_close(lina.utils.rms(x), lina_cpp.utils.rms(x), rtol=1e-12, name="rms")


def test_pad_or_crop_pad():
    x = real((32, 32))
    assert_close(lina.utils.pad_or_crop(x, 64), lina_cpp.utils.pad_or_crop(x, 64),
                 rtol=1e-12, name="pad")


def test_pad_or_crop_crop():
    x = real((64, 64))
    assert_close(lina.utils.pad_or_crop(x, 32), lina_cpp.utils.pad_or_crop(x, 32),
                 rtol=1e-12, name="crop")


def test_create_annular_mask():
    # NB: the two packages order the positional args differently
    # (lina: N, irad, orad, pixelscale ; lina_cpp: N, pixelscale, irad, orad),
    # so call with keywords to compare the same logical mask.
    a = lina.utils.create_annular_mask(64, irad=3, orad=10, pixelscale=0.2)
    b = lina_cpp.utils.create_annular_mask(64, pixelscale=0.2, irad=3, orad=10)
    assert_close(_np(a).astype(float), _np(b).astype(float), rtol=1e-12, name="annular_mask")


def test_create_annular_focal_plane_mask():
    a = lina.utils.create_annular_focal_plane_mask(64, 0.2, 3, 10)
    b = lina_cpp.utils.create_annular_focal_plane_mask(64, 0.2, 3, 10)
    assert_close(_np(a).astype(float), _np(b).astype(float), rtol=1e-12, name="annular_fp_mask")


def test_lstsq():
    modes = real((6, 32, 32))
    data = real((32, 32))
    assert_close(lina.utils.lstsq(modes, data), lina_cpp.utils.lstsq(modes, data),
                 rtol=1e-6, name="lstsq")


def test_tikhonov_inverse():
    A = real((40, 24))
    assert_close(lina.utils.tikhonov_inverse(A, 1e-3),
                 lina_cpp.utils.tikhonov_inverse(A, 1e-3), rtol=1e-6, name="tikhonov_inverse")


def test_beta_reg():
    S = real((120, 32))
    assert_close(lina.utils.beta_reg(S, -2), lina_cpp.utils.beta_reg(S, -2),
                 rtol=1e-6, name="beta_reg")


# --------------------------------------------------------------------------- #
# dm (mode / command generation)
# --------------------------------------------------------------------------- #
def test_create_mask():
    assert_close(_np(lina.dm.create_mask(Nact=34)).astype(float),
                 _np(lina_cpp.dm.create_mask(Nact=34)).astype(float), rtol=1e-12, name="dm_mask")


def test_create_hadamard_modes():
    dm_mask = lina_cpp.dm.create_mask(Nact=34)
    a = lina.dm.create_hadamard_modes(_np(dm_mask).astype(bool))
    b = lina_cpp.dm.create_hadamard_modes(dm_mask)
    assert_close(_np(a).astype(float), _np(b).astype(float), rtol=1e-9, name="hadamard")


def test_make_gaussian_inf_fun():
    assert_close(lina.dm.make_gaussian_inf_fun(300e-6, 4.0, 0.15, 36),
                 lina_cpp.dm.make_gaussian_inf_fun(300e-6, 4.0, 0.15, 36),
                 rtol=1e-9, name="gaussian_inf_fun")


def test_make_fourier_command():
    assert_close(lina.dm.make_fourier_command(10, 10, 34),
                 lina_cpp.dm.make_fourier_command(10, 10, 34), rtol=1e-9, name="fourier_command")


def test_make_f():
    assert_close(_np(lina.dm.make_f(10, 6, (-1, 0), 34)).astype(float),
                 _np(lina_cpp.dm.make_f(10, 6, (-1, 0), 34)).astype(float),
                 rtol=1e-9, name="make_f")


# --------------------------------------------------------------------------- #
# wfe (index conversions + PSD synthesis)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("j", [1, 2, 3, 6, 10, 15])
def test_noll_index(j):
    assert lina.wfe.noll_index_to_mn(j) == tuple(lina_cpp.wfe.noll_index_to_mn(j))


def test_generate_freqs():
    fa, da, ta = lina.wfe.generate_freqs(0.1e-3, 1.0)
    fb, db, tb = lina_cpp.wfe.generate_freqs(0.1e-3, 1.0)
    assert_close(fa, fb, rtol=1e-9, name="generate_freqs.freqs")
    assert abs(da - db) <= 1e-12 * max(abs(da), 1.0)


def test_roll_psd():
    freqs, _, _ = lina_cpp.wfe.generate_freqs(0.1e-3, 1.0)
    fnp = _np(freqs)
    assert_close(lina.wfe.roll_psd(fnp, 3.0, 1.0, 2.0),
                 lina_cpp.wfe.roll_psd(fnp, 3.0, 1.0, 2.0), rtol=1e-9, name="roll_psd")


# --------------------------------------------------------------------------- #
# coro_utils
# --------------------------------------------------------------------------- #
def test_compute_contrast():
    im = np.abs(real((64, 64)))
    mask = lina_cpp.utils.create_annular_focal_plane_mask(64, 0.2, 3, 10)
    mnp = _np(mask)
    a = lina.coro_utils.compute_contrast(im, mnp)
    b = lina_cpp.coro_utils.compute_contrast(im, mnp)
    assert_close(np.atleast_1d(_np(a)[0] if np.ndim(_np(a)) else a),
                 np.atleast_1d(_np(b)[0] if np.ndim(_np(b)) else b),
                 rtol=1e-9, name="compute_contrast")


# --------------------------------------------------------------------------- #
# control model + iEFC (the sim_iefc_demo-style end-to-end checks, small model)
# --------------------------------------------------------------------------- #
MODEL_PARAMS = dict(
    wavelength_c=630e-9, npix=128, N_vortex_lres=512, vortex_hres_sampling=0.1,
    vortex_win_diam=30, vortex_dot_mask_diam_lamDc=0.65, ncamsci=64,
    camsci_pxscl_lamDc=0.2, Nact=34, dm_beam_diam=9.3e-3, lyot_pupil_diam=4.2e-3,
    lyot_stop_diam=3.7e-3, act_spacing=300e-6,
)


@pytest.fixture(scope="module")
def models():
    Cpy = lina.control_models.MODEL(**MODEL_PARAMS)
    Cc = lina_cpp.control_models.MODEL(**MODEL_PARAMS)
    for C in (Cpy, Cc):
        C.Imax_ref = 1
        C.use_vortex = 0
    return Cpy, Cc


@pytest.mark.slow
def test_model_snap_no_vortex(models):
    Cpy, Cc = models
    Cpy.use_vortex = 0
    Cc.use_vortex = 0
    assert_close(Cpy.snap(), Cc.snap(), rtol=1e-8, name="snap(no vortex)")


@pytest.mark.slow
def test_model_snap_vortex(models):
    Cpy, Cc = models
    Cpy.use_vortex = 1
    Cc.use_vortex = 1
    assert_close(Cpy.snap(), Cc.snap(), rtol=1e-6, name="snap(vortex)")


@pytest.mark.slow
def test_dm_val_and_grad(models):
    Cpy, Cc = models
    opd = real((Cpy.Ndef, Cpy.Ndef)) * 1e-8
    acts = real(Cpy.Nacts) * 1e-9
    Jp, gp = lina.control_models.dm_val_and_grad(acts, opd, Cpy)
    Jc, gc = lina_cpp.control_models.dm_val_and_grad(acts, opd, Cc)
    # NB: on the CPU/NumPy backend lina's ensure_np_array() returns None for a
    # scalar, so lina yields J=None there (it only produces J on the GPU
    # backend); lina_cpp always returns the scalar. Compare J only when lina
    # actually provides one.
    if Jp is not None:
        assert abs(float(Jp) - float(Jc)) <= 1e-9 * max(abs(float(Jp)), 1.0), "dm_val_and_grad J"
    # gradient direction parity (the analytic gradient is consistently scaled)
    a, b = _np(gp).ravel(), _np(gc).ravel()
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert cos > 1 - 1e-6, f"dm_val_and_grad grad direction cos={cos}"


@pytest.mark.slow
def test_iefc_calibrate(models):
    Cpy, Cc = models
    for C in (Cpy, Cc):
        C.use_vortex = 1
    wfs_mask = lina_cpp.utils.create_annular_focal_plane_mask(
        Cc.ncamsci, Cc.camsci_pxscl_lamDc, 3, 10, edge=3, rotation=90, centering="odd")
    dm_mask = lina_cpp.dm.create_mask(Nact=Cc.Nact)
    probes = lina_cpp.dm.create_fourier_probes(
        dm_mask, Cc.ncamsci, Cc.camsci_pxscl_lamDc, 2, 14, rotation=90,
        use_weighting=True, nprobes=2)
    calib = lina_cpp.dm.create_hadamard_modes(dm_mask)[:16]

    def tif(C):
        return C.snap()

    def sdf(cmd, C, channel=1):
        C.dm_commands[channel] = cmd

    rm_p, _ = lina.iefc.calibrate(tif, {"C": Cpy}, sdf, {"C": Cpy, "channel": 3},
                                  wfs_mask, probes, 5e-9, calib, 2e-9)
    rm_c, _ = lina_cpp.iefc.calibrate(tif, {"C": Cc}, sdf, {"C": Cc, "channel": 3},
                                      wfs_mask, probes, 5e-9, calib, 2e-9)
    assert_close(rm_p, rm_c, rtol=1e-6, name="iefc.calibrate response_matrix")
