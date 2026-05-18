"""Per-method parity tests: every C++ math function vs its Python original.

This module compares each numerical function in the C++ port against its
Python counterpart, using direct `lina_cpp` pybind bindings (so we exercise
the actual library, not a stdout-parsed CLI).

Build prerequisites:

    cmake -S cpp -B cpp/build-native \\
          -DLINA_BUILD_PYBIND=ON \\
          -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir) \\
          -DLINA_FORCE_LAPACKE=ON
    cmake --build cpp/build-native
    export PYTHONPATH=$PWD/cpp/build-native:$PYTHONPATH

Then run:

    python3 -m unittest lina.tests.test_per_method_parity -v

Tests are self-skipping if `lina_cpp` (the pybind module) is not importable.

Each test documents the expected behavior. Tests that target known
divergences are marked accordingly with the relevant findings from the
audit (see cpp/AUDIT.md).
"""
import math
import os
import sys
import unittest

import numpy as np
import scipy
import scipy.linalg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _try_import_lina_cpp():
    try:
        import lina_cpp  # noqa: F401
        return True
    except Exception:  # pragma: no cover - environment dependent
        return False


def _to_np(arr):
    """Convert lina output (numpy or cupy) to numpy for comparison.

    Uses ``lina.math_module.ensure_np_array`` when available so cupy
    arrays produced by pure-Python ``lina`` calls don't break
    ``np.testing.assert_allclose``. Falls back to ``np.asarray`` when
    ``lina`` itself failed to import (in which case the input is
    already numpy by construction).
    """
    try:
        from lina.math_module import ensure_np_array
        return ensure_np_array(arr)
    except Exception:
        return np.asarray(arr)


def _has_lina_cpp_attr(name: str) -> bool:
    """True iff this lina_cpp build exposes the named binding."""
    try:
        import lina_cpp
        return hasattr(lina_cpp, name)
    except Exception:
        return False


def _mask_to_uint8(mask):
    """Convert a boolean ndarray to flat C-contiguous uint8 (for pybind)."""
    return np.ascontiguousarray(mask, dtype=np.bool_).astype(np.uint8).ravel()


# ---------------------------------------------------------------------------
# Base class — sets up numpy backend on the Python side and skips tests if
# lina_cpp is unavailable.
# ---------------------------------------------------------------------------


class _LinaCppTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _try_import_lina_cpp():
            raise unittest.SkipTest(
                "lina_cpp pybind module not importable. Build with "
                "-DLINA_BUILD_PYBIND=ON and add the build dir to PYTHONPATH."
            )
        # Force the CPU/numpy backend on the Python side. The parity
        # tests compare lina (Python) to lina_cpp (C++) by value, and
        # both sides must agree to bit-for-bit numpy semantics. The new
        # set_backend("cpu") call propagates to every lina submodule.
        import lina
        lina.set_backend("cpu")


# ---------------------------------------------------------------------------
# utils.py
# ---------------------------------------------------------------------------


class TestUtils(_LinaCppTest):

    def test_mean_unmasked(self):
        import lina_cpp
        rng = np.random.default_rng(0)
        for shape in [(8, 8), (33, 33), (64, 32)]:
            arr = rng.standard_normal(shape)
            cpp = lina_cpp.mean(arr)
            py = float(np.mean(arr))
            self.assertAlmostEqual(cpp, py, places=12,
                                   msg=f"mean failed for shape {shape}")

    def test_mean_masked(self):
        import lina_cpp
        rng = np.random.default_rng(1)
        arr = rng.standard_normal((16, 16))
        mask_2d = arr > 0
        cpp = lina_cpp.mean_masked(arr, _mask_to_uint8(mask_2d))
        py = float(np.mean(arr[mask_2d]))
        self.assertAlmostEqual(cpp, py, places=12)

    def test_rms(self):
        import lina_cpp
        rng = np.random.default_rng(2)
        for shape in [(8, 8), (33, 33), (64, 32)]:
            arr = rng.standard_normal(shape)
            cpp = lina_cpp.rms(arr)
            py = float(np.sqrt(np.mean(arr * arr)))
            self.assertAlmostEqual(cpp, py, places=12,
                                   msg=f"rms failed for shape {shape}")

    def test_pad_or_crop_pad(self):
        import lina_cpp
        rng = np.random.default_rng(3)
        for n_in, n_out in [(8, 16), (16, 32), (33, 64)]:
            arr = rng.standard_normal((n_in, n_in))
            cpp = lina_cpp.pad_or_crop(arr, n_out)
            from lina import utils
            py = utils.pad_or_crop(arr.copy(), n_out)
            np.testing.assert_allclose(cpp, py, rtol=0, atol=0,
                                       err_msg=f"pad {n_in}->{n_out}")

    def test_pad_or_crop_crop(self):
        import lina_cpp
        rng = np.random.default_rng(4)
        for n_in, n_out in [(16, 8), (32, 16), (64, 33)]:
            arr = rng.standard_normal((n_in, n_in))
            cpp = lina_cpp.pad_or_crop(arr, n_out)
            from lina import utils
            py = utils.pad_or_crop(arr.copy(), n_out)
            np.testing.assert_allclose(cpp, py, rtol=0, atol=0,
                                       err_msg=f"crop {n_in}->{n_out}")

    def test_make_grid_even(self):
        """make_grid agrees for even npix."""
        import lina_cpp
        from lina import utils
        for npix in [8, 16, 32]:
            for half_shift in (False, True):
                cpp_x, cpp_y = lina_cpp.make_grid(npix, 1.0, half_shift)
                py_x, py_y = utils.make_grid(npix, 1.0, half_shift=half_shift)
                np.testing.assert_allclose(cpp_x, py_x, atol=1e-12)
                np.testing.assert_allclose(cpp_y, py_y, atol=1e-12)

    def test_make_grid_odd_documents_known_divergence(self):
        """For odd npix the C++ and Python grids differ by half a pixel.

        Python uses npix//2 (integer floor); C++ uses npix/2.0 (float).
        For npix=33, Python centers on pixel 16 (value 0); C++ has no zero
        pixel (16 -> -0.5, 17 -> 0.5).

        This test asserts the *expected* divergence so it acts as a
        regression check until the C++ side is fixed.
        """
        import lina_cpp
        from lina import utils
        npix = 33
        cpp_x, _ = lina_cpp.make_grid(npix, 1.0, False)
        py_x, _ = utils.make_grid(npix, 1.0, half_shift=False)
        # Python's center column is at 0; C++'s is at -0.5.
        self.assertAlmostEqual(float(py_x[0, npix // 2]), 0.0)
        self.assertAlmostEqual(float(cpp_x[0, npix // 2]), -0.5)
        # They differ by exactly +0.5 everywhere (Python = C++ + 0.5).
        np.testing.assert_allclose(py_x - cpp_x, 0.5, atol=1e-12)

    def test_create_annular_mask_no_edge_filter(self):
        """create_annular_mask with edge=None matches Python edge=None
        (full annulus, no half-plane cut)."""
        import lina_cpp
        from lina import utils
        for n in [16, 32, 33]:
            cpp = lina_cpp.create_annular_mask(
                n, 1.0, 2.5, 7.5, edge=None, x_shift=0.0, y_shift=0.0,
                rotation=0.0)
            py = utils.create_annular_mask(
                n, 1.0, 2.5, 7.5, edge=None, x_shift=0, y_shift=0,
                return_np=True).astype(np.uint8)
            np.testing.assert_array_equal(cpp, py, err_msg=f"n={n}")

    def test_create_annular_mask_edge_zero_filters_right_half(self):
        """create_annular_mask with edge=0 matches Python edge=0
        (filter at xr > 0, right half of annulus)."""
        import lina_cpp
        from lina import utils
        for n in [16, 32, 33]:
            cpp = lina_cpp.create_annular_mask(
                n, 1.0, 2.5, 7.5, edge=0.0, x_shift=0.0, y_shift=0.0,
                rotation=0.0)
            py = utils.create_annular_mask(
                n, 1.0, 2.5, 7.5, edge=0, x_shift=0, y_shift=0,
                return_np=True).astype(np.uint8)
            np.testing.assert_array_equal(cpp, py, err_msg=f"n={n}")


# ---------------------------------------------------------------------------
# props.py
# ---------------------------------------------------------------------------


class TestProps(_LinaCppTest):

    def test_fft_random(self):
        """fft matches numpy's centered-FFT convention."""
        import lina_cpp
        rng = np.random.default_rng(10)
        for n in [8, 16, 33, 64]:
            arr = (rng.standard_normal((n, n))
                   + 1j * rng.standard_normal((n, n))).astype(np.complex128)
            cpp = lina_cpp.fft_cpu(arr)
            py = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(arr)))
            np.testing.assert_allclose(cpp, py, rtol=1e-10, atol=1e-9,
                                       err_msg=f"fft size {n}")

    def test_ifft_random(self):
        """ifft matches numpy's centered-IFFT convention."""
        import lina_cpp
        rng = np.random.default_rng(11)
        for n in [8, 16, 33, 64]:
            arr = (rng.standard_normal((n, n))
                   + 1j * rng.standard_normal((n, n))).astype(np.complex128)
            cpp = lina_cpp.ifft_cpu(arr)
            py = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(arr)))
            np.testing.assert_allclose(cpp, py, rtol=1e-10, atol=1e-9,
                                       err_msg=f"ifft size {n}")

    def test_fft_ifft_roundtrip_even_n(self):
        """For even n, ifft(fft(x)) == x exactly (numerical precision).

        For odd n the centered convention used by both numpy and the C++
        port composes ifftshift twice in the middle of the pipeline, which
        rolls the result by 1 pixel — this is a property of numpy's shift
        conventions, not a port bug. See test_fft_ifft_matches_python.
        """
        import lina_cpp
        rng = np.random.default_rng(12)
        for n in [8, 16, 32, 64]:
            arr = (rng.standard_normal((n, n))
                   + 1j * rng.standard_normal((n, n))).astype(np.complex128)
            roundtrip = lina_cpp.ifft_cpu(lina_cpp.fft_cpu(arr))
            np.testing.assert_allclose(roundtrip, arr,
                                       rtol=1e-9, atol=1e-9,
                                       err_msg=f"n={n}")

    def test_fft_ifft_matches_python_for_odd_n(self):
        """For odd n the centered FFT pipeline does not roundtrip in
        either Python or C++ (same shift-convention quirk). What MUST
        hold is that the C++ produces the same numerical result as
        Python's lina.props.fft / lina.props.ifft on the same input."""
        import lina_cpp
        from lina import math_module, props
        import scipy
        math_module.update_np(np); math_module.update_scipy(scipy)
        rng = np.random.default_rng(42)
        for n in [33, 65]:
            arr = (rng.standard_normal((n, n))
                   + 1j * rng.standard_normal((n, n))).astype(np.complex128)
            cpp_fwd = lina_cpp.fft_cpu(arr)
            py_fwd = _to_np(props.fft(arr))
            np.testing.assert_allclose(cpp_fwd, py_fwd,
                                       rtol=1e-10, atol=1e-9,
                                       err_msg=f"fft n={n}")
            cpp_inv = lina_cpp.ifft_cpu(arr)
            py_inv = _to_np(props.ifft(arr))
            np.testing.assert_allclose(cpp_inv, py_inv,
                                       rtol=1e-10, atol=1e-9,
                                       err_msg=f"ifft n={n}")

    def test_make_vortex_phase_mask_even(self):
        """vortex mask matches Python for even npix (where conventions align)."""
        import lina_cpp
        from lina import props
        for n in [8, 16, 32]:
            cpp = lina_cpp.make_vortex_phase_mask(n, 6, "odd")
            py = props.make_vortex_phase_mask(n, charge=6, grid='odd')
            np.testing.assert_allclose(cpp, py, atol=1e-12,
                                       err_msg=f"vortex npix={n}")

    @unittest.expectedFailure
    def test_make_vortex_phase_mask_odd_known_divergence(self):
        """Documents that Python's linspace convention differs for odd npix.

        Python uses linspace(-npix//2, npix//2-1, npix) which produces a
        non-unit-spaced grid for odd npix. The C++ port assumes unit
        spacing. They cannot agree for odd npix without fixing one side.
        """
        import lina_cpp
        from lina import props
        n = 17
        cpp = lina_cpp.make_vortex_phase_mask(n, 6, "odd")
        py = props.make_vortex_phase_mask(n, charge=6, grid='odd')
        np.testing.assert_allclose(cpp, py, atol=1e-12)

    def test_mft_forward_against_padded_fft(self):
        """mft_forward with pp_centering='odd', N=npix, npsf=N, scale=1
        is equivalent to a centered DFT with no padding."""
        import lina_cpp
        rng = np.random.default_rng(13)
        for n in [8, 16, 32]:
            arr = (rng.standard_normal((n, n))
                   + 1j * rng.standard_normal((n, n))).astype(np.complex128)
            cpp_mft = lina_cpp.mft_forward(arr, n, n, 1.0, "-", "odd", "odd")
            # Compare to Python's mft_forward directly:
            from lina import props
            py_mft = props.mft_forward(arr, n, n, 1.0,
                                       convention='-',
                                       pp_centering='odd',
                                       fp_centering='odd')
            np.testing.assert_allclose(cpp_mft, py_mft,
                                       rtol=1e-9, atol=1e-9)

    def test_mft_reverse_inverse_of_mft_forward(self):
        """mft_reverse should approximately invert mft_forward."""
        import lina_cpp
        rng = np.random.default_rng(14)
        n, npsf = 16, 16
        arr = (rng.standard_normal((n, n))
               + 1j * rng.standard_normal((n, n))).astype(np.complex128)
        fp = lina_cpp.mft_forward(arr, n, npsf, 1.0, "-", "odd", "odd")
        back = lina_cpp.mft_reverse(fp, 1.0, n, n, "+", "odd", "odd")
        # Round-trip exact for unit pixel scale
        np.testing.assert_allclose(back, arr, rtol=1e-7, atol=1e-7)

    def test_ang_spec_zero_distance_identity(self):
        """ang_spec with distance=0 should be identity."""
        import lina_cpp
        rng = np.random.default_rng(15)
        n = 32
        wf = (rng.standard_normal((n, n))
              + 1j * rng.standard_normal((n, n))).astype(np.complex128)
        out = lina_cpp.ang_spec(wf, wavelength=633e-9, distance=0.0,
                                pixelscale=10e-6)
        np.testing.assert_allclose(out, wf, rtol=1e-8, atol=1e-8)


# ---------------------------------------------------------------------------
# linalg
# ---------------------------------------------------------------------------


class TestLinalg(_LinaCppTest):

    def test_gemm(self):
        import lina_cpp
        rng = np.random.default_rng(20)
        for shape in [(4, 5, 3), (16, 8, 12), (33, 17, 5)]:
            m, k, n = shape
            A = rng.standard_normal((m, k))
            B = rng.standard_normal((k, n))
            cpp = lina_cpp.gemm(A, B, False, False)
            np.testing.assert_allclose(cpp, A @ B, rtol=1e-10, atol=1e-10)

    def test_gemm_transpose(self):
        import lina_cpp
        rng = np.random.default_rng(21)
        A = rng.standard_normal((6, 4))
        B = rng.standard_normal((6, 5))
        cpp = lina_cpp.gemm(A, B, True, False)  # A.T @ B
        np.testing.assert_allclose(cpp, A.T @ B, rtol=1e-10, atol=1e-10)

    def test_gemv(self):
        import lina_cpp
        if not _has_lina_cpp_attr("gemv"):
            self.skipTest("lina_cpp.gemv binding not present in this build; "
                          "rebuild with -DLINA_USE_OPENBLAS=ON.")
        rng = np.random.default_rng(22)
        A = rng.standard_normal((10, 7))
        x = rng.standard_normal(7)
        # Permissive tolerance: different OpenBLAS builds use different
        # FMA orderings, so the last 1-2 ULP of the dot products differ.
        cpp = lina_cpp.gemv(A, x, False)
        np.testing.assert_allclose(cpp, A @ x, rtol=1e-10, atol=1e-10,
                                   err_msg=f"gemv: cpp[:3]={np.asarray(cpp)[:3]}, "
                                           f"ref[:3]={(A @ x)[:3]}")

    def test_svd_float_singular_values(self):
        """Singular values from svd_float_cpu agree with NumPy.

        Note: the double-precision `lina_cpp.svd` is currently broken when
        the library is built with `LINA_FORCE_LAPACKE=ON` because that
        code path passes `nullptr` for the LAPACKE `superb` workspace
        (see cpp/AUDIT.md). Use the float path here, which is correct.
        """
        import lina_cpp
        if not _has_lina_cpp_attr("svd_float_cpu"):
            self.skipTest("lina_cpp.svd_float_cpu binding not present; "
                          "rebuild with -DLINA_USE_LAPACKE=ON.")
        rng = np.random.default_rng(23)
        for shape in [(6, 4), (4, 6), (10, 10), (32, 16)]:
            A = rng.standard_normal(shape).astype(np.float32)
            U, s, Vt = lina_cpp.svd_float_cpu(A)
            _, s_py, _ = np.linalg.svd(A.astype(np.float64), full_matrices=True)
            # Allow ~1e-3 absolute since this is a single-precision SVD
            # and different LAPACK builds can shift singular values by
            # a few ULPs.
            np.testing.assert_allclose(s, s_py, rtol=2e-3, atol=2e-3,
                                       err_msg=f"svd shape={shape}: "
                                               f"cpp s={np.asarray(s)[:4]}, "
                                               f"ref s={s_py[:4]}")

    def test_svd_float_reconstruction(self):
        """U @ diag(s) @ Vt reconstructs A (handles sign ambiguity)."""
        import lina_cpp
        if not _has_lina_cpp_attr("svd_float_cpu"):
            self.skipTest("lina_cpp.svd_float_cpu binding not present; "
                          "rebuild with -DLINA_USE_LAPACKE=ON.")
        rng = np.random.default_rng(24)
        for shape in [(6, 4), (4, 6), (8, 8)]:
            A = rng.standard_normal(shape).astype(np.float32)
            U, s, Vt = lina_cpp.svd_float_cpu(A)
            S = np.zeros(shape, dtype=np.float32)
            for i, v in enumerate(s):
                S[i, i] = v
            recon = U @ S @ Vt
            np.testing.assert_allclose(recon, A, rtol=2e-3, atol=2e-3)


# ---------------------------------------------------------------------------
# dm.py
# ---------------------------------------------------------------------------


class TestDm(_LinaCppTest):

    def test_create_mask_even(self):
        import lina_cpp
        from lina import dm
        for nact in [4, 16, 34]:
            cpp = lina_cpp.create_mask(nact)
            py = dm.create_mask(Nact=nact, return_np=True).astype(np.uint8)
            np.testing.assert_array_equal(cpp, py,
                                          err_msg=f"dm_mask nact={nact}")

    def test_make_gaussian_inf_fun(self):
        import lina_cpp
        from lina import dm
        cpp = lina_cpp.make_gaussian_inf_fun(
            act_spacing=300e-6, sampling=10.0, coupling=0.15, nact=4)
        py = dm.make_gaussian_inf_fun(
            act_spacing=300e-6, sampling=10, coupling=0.15, Nact=4)
        np.testing.assert_allclose(cpp, py, rtol=1e-10, atol=1e-12)

    def test_make_fourier_command_even(self):
        import lina_cpp
        from lina import dm
        cpp = lina_cpp.make_fourier_command(5, 7, 34, 0.0)
        py = dm.make_fourier_command(x_cpa=5, y_cpa=7, Nact=34,
                                     phase=0, return_np=True)
        np.testing.assert_allclose(cpp, py, rtol=1e-12, atol=1e-12)

    def test_make_ring_even(self):
        import lina_cpp
        from lina import dm
        cpp = lina_cpp.make_ring(5.0, 32, 0.5)
        py = dm.make_ring(rad=5, Nact=32, thresh=0.5)
        np.testing.assert_array_equal(cpp.astype(int), np.asarray(py).astype(int))

    def test_create_hadamard_modes_content_match(self):
        """C++ returns 2D (np2, Nact*Nact); Python returns 3D (np2, Nact, Nact).

        Both should hold the same Hadamard mode amplitudes after reshaping.
        """
        import lina_cpp
        from lina import dm
        nact = 16  # produces some inactive actuators
        mask_py = dm.create_mask(Nact=nact, return_np=True)
        cpp = lina_cpp.create_hadamard_modes(mask_py.astype(np.uint8))
        py = dm.create_hadamard_modes(mask_py, return_np=True)
        # Reshape Python from (np2, Nact, Nact) -> (np2, Nact*Nact)
        py_flat = py.reshape(py.shape[0], -1)
        np.testing.assert_allclose(cpp, py_flat, atol=1e-12)


# ---------------------------------------------------------------------------
# coro_utils.py
# ---------------------------------------------------------------------------


class TestCoroUtils(_LinaCppTest):

    def test_normalize_coro_im_unit_params(self):
        """With unit exposures/gains/atten, normalize_coro_im just subtracts
        dark and divides by Imax."""
        import lina_cpp
        rng = np.random.default_rng(30)
        raw = 100.0 + 10.0 * rng.standard_normal((8, 8))
        Imax = 1000.0
        cpp = lina_cpp.normalize_coro_im(
            raw,
            exp_time_im=1.0, gain_im=0.0, atten_im=0.0,
            exp_time_ref=1.0, gain_ref=0.0, atten_ref=0.0,
            Imax=Imax,
            dark_im=None,
        )
        py = (raw - 0.0) / Imax
        np.testing.assert_allclose(cpp, py, rtol=1e-12, atol=1e-12)

    def test_compute_contrast_positive_pixels_only(self):
        import lina_cpp
        from lina import coro_utils
        rng = np.random.default_rng(31)
        ni = rng.standard_normal((10, 10))  # mix of pos/neg
        mask = np.ones_like(ni, dtype=bool)
        contrast_cpp, n_mask, n_pos = lina_cpp.compute_contrast(
            ni, mask.astype(np.uint8))
        py = coro_utils.compute_contrast(ni, mask, verbose=False)
        self.assertAlmostEqual(contrast_cpp, py, places=12)
        self.assertEqual(n_mask, ni.size)
        self.assertEqual(n_pos, int((ni > 0).sum()))


# ---------------------------------------------------------------------------
# wfe.py
# ---------------------------------------------------------------------------


class TestWfe(_LinaCppTest):
    """Parity tests for wfe.py math primitives.

    The poppy-dependent generate_opd / generate_wfe live in Python; we
    only test the pure-math functions that are now C++.
    """

    def test_noll_index_to_mn_matches_python(self):
        import lina_cpp
        from lina.wfe import noll_index_to_mn
        for j in range(1, 50):
            self.assertEqual(tuple(lina_cpp.noll_index_to_mn(j)),
                             tuple(noll_index_to_mn(j)),
                             msg=f"j={j}")

    def test_mn_to_noll_index_matches_python(self):
        import lina_cpp
        from lina.wfe import mn_to_noll_index
        for n in range(8):
            for m in range(-n, n + 1, 2):
                self.assertEqual(lina_cpp.mn_to_noll_index(m, n),
                                 mn_to_noll_index(m, n),
                                 msg=f"(m={m},n={n})")

    def test_fringe_index_to_mn_matches_python(self):
        import lina_cpp
        from lina.wfe import fringe_index_to_mn
        for j in range(1, 50):
            self.assertEqual(tuple(lina_cpp.fringe_index_to_mn(j)),
                             tuple(fringe_index_to_mn(j)),
                             msg=f"j={j}")

    def test_mn_to_fringe_index_matches_python(self):
        import lina_cpp
        from lina.wfe import mn_to_fringe_index
        # Cover (m, n) pairs that are reachable from fringe_index_to_mn(j) for j in [1, 50).
        from lina.wfe import fringe_index_to_mn
        for j in range(1, 50):
            m, n = fringe_index_to_mn(j)
            self.assertEqual(lina_cpp.mn_to_fringe_index(m, n),
                             mn_to_fringe_index(m, n),
                             msg=f"j={j} -> (m={m},n={n})")

    def test_generate_freqs_matches_python(self):
        import lina_cpp
        from lina import wfe as pwfe
        py_f, py_df, py_t = pwfe.generate_freqs(delt=0.5e-3, tmax=8.0)
        cpp_f, cpp_df, cpp_t = lina_cpp.wfe_generate_freqs(0.5e-3, 8.0)
        np.testing.assert_allclose(_to_np(py_f), cpp_f, rtol=0, atol=1e-12)
        self.assertEqual(len(py_f), len(cpp_f))
        self.assertAlmostEqual(py_df, cpp_df, places=12)
        np.testing.assert_allclose(_to_np(py_t), cpp_t, rtol=0, atol=1e-12)

    def test_roll_psd_matches_python(self):
        import lina_cpp
        from lina import wfe as pwfe
        freqs = np.linspace(0, 500, 1001)
        py = _to_np(pwfe.roll_psd(freqs, beta=2.0, f_roll=15.0, alpha=2.5,
                                    normalized=True, verbose=False))
        cpp = lina_cpp.wfe_roll_psd(freqs, beta=2.0, f_roll=15.0, alpha=2.5,
                                     normalized=True)
        np.testing.assert_allclose(py, cpp, rtol=1e-12, atol=1e-13)

        # Non-normalized branch.
        py2 = _to_np(pwfe.roll_psd(freqs, beta=1.5, f_roll=5.0, alpha=3.0,
                                     normalized=False, verbose=False))
        cpp2 = lina_cpp.wfe_roll_psd(freqs, beta=1.5, f_roll=5.0, alpha=3.0,
                                      normalized=False)
        np.testing.assert_allclose(py2, cpp2, rtol=1e-12, atol=1e-13)

    def test_compute_cumulative_psd_matches_python(self):
        """The C++ Simpson integration agrees with scipy.integrate.simpson
        on uniform grids to within scipy's non-uniform-spacing correction
        term (~1e-6 absolute for typical PSD curves)."""
        import lina_cpp
        from lina import wfe as pwfe
        freqs = np.linspace(0, 500, 5001)
        psd = _to_np(pwfe.roll_psd(freqs, beta=1.0, f_roll=10.0, alpha=2.5,
                                     normalized=True, verbose=False))
        cum_p, _ = pwfe.compute_cumulative_psd(freqs, psd)
        cum_c, _ = lina_cpp.wfe_compute_cumulative_psd(freqs, psd)
        np.testing.assert_allclose(_to_np(cum_p), cum_c, rtol=1e-5, atol=1e-5)

    def test_generate_time_series_psd_statistics(self):
        """The C++ generate_time_series uses a different PRNG than numpy
        (Mersenne Twister vs numpy's default), so individual samples
        differ. But the time-series RMS, which is a deterministic function
        of the PSD, must match the Python output to better than ~1%."""
        import lina_cpp
        from lina import wfe as pwfe
        freqs, _delf, _times = pwfe.generate_freqs(delt=1e-3, tmax=10.0)
        freqs = _to_np(freqs)
        psd = _to_np(pwfe.roll_psd(freqs, beta=1.0, f_roll=10.0, alpha=2.5,
                                     normalized=True, verbose=False))

        ts_p, _ = pwfe.generate_time_series(psd, freqs, seed=42, verbose=False)
        ts_c, _ = lina_cpp.wfe_generate_time_series(psd, freqs, seed=42)
        ts_p = _to_np(ts_p)
        ts_c = _to_np(ts_c)

        rms_p = float(np.sqrt(np.mean(ts_p ** 2)))
        rms_c = float(np.sqrt(np.mean(ts_c ** 2)))
        # Both backends produce time series that integrate to the same
        # PSD power; the empirical RMS agrees up to PRNG noise.
        self.assertAlmostEqual(rms_p, rms_c, delta=0.02 * rms_p,
                               msg=f"RMS mismatch: py={rms_p}, cpp={rms_c}")

        # Both must be real-valued: imag-leakage tests
        self.assertEqual(ts_c.dtype, np.float64)


# ---------------------------------------------------------------------------
# llowfsc.py
# ---------------------------------------------------------------------------


class TestLlowfsc(_LinaCppTest):
    """Parity tests for the math kernels of lina.llowfsc:
    acquire_ref math, reconstruct, compute_zpo, loop_step.
    """

    def _setup_random(self, seed=0, H=32, W=32, Nmodes=10):
        rng = np.random.default_rng(seed)
        camlo = 5.0 + 0.1 * rng.standard_normal((H, W))
        mask = (rng.standard_normal((H, W)) > 0.0).astype(np.uint8)
        ref = 0.001 * rng.standard_normal((H, W))
        Nmask = int(mask.sum())
        C = rng.standard_normal((Nmodes, Nmask))
        return camlo, mask, ref, C

    def test_acquire_ref_scalar_dark_matches_python(self):
        import lina_cpp
        from lina.llowfsc import acquire_ref as py_acquire_ref
        camlo, mask, _ref, _C = self._setup_random(seed=1)
        # Python's acquire_ref takes a callable for the camera read.
        py_ref, py_coeff = py_acquire_ref(
            take_im_fun=lambda: camlo, take_im_params={},
            wfs_mask=mask.astype(bool), camlo_dark=0.5, flux_norm=True)
        cpp_ref, cpp_coeff = lina_cpp.llowfsc_acquire_ref(
            camlo, mask, dark_im=0.5, flux_norm=True)
        np.testing.assert_allclose(_to_np(py_ref), cpp_ref,
                                    rtol=1e-10, atol=1e-12)
        # flux_norm_coeff is a sum over masked pixels; FP summation order
        # differs between the two backends, allow ~ULP relative tolerance.
        self.assertAlmostEqual(float(py_coeff), float(cpp_coeff),
                               delta=1e-9 * abs(float(py_coeff)) + 1e-12)

    def test_acquire_ref_image_dark_matches_python(self):
        import lina_cpp
        from lina.llowfsc import acquire_ref as py_acquire_ref
        camlo, mask, _ref, _C = self._setup_random(seed=2)
        rng = np.random.default_rng(99)
        dark = 0.05 * rng.standard_normal(camlo.shape)
        py_ref, py_coeff = py_acquire_ref(
            take_im_fun=lambda: camlo, take_im_params={},
            wfs_mask=mask.astype(bool), camlo_dark=dark, flux_norm=True)
        cpp_ref, cpp_coeff = lina_cpp.llowfsc_acquire_ref(
            camlo, mask, dark_im=dark, flux_norm=True)
        np.testing.assert_allclose(_to_np(py_ref), cpp_ref,
                                    rtol=1e-10, atol=1e-12)
        self.assertAlmostEqual(float(py_coeff), float(cpp_coeff),
                               delta=1e-9 * abs(float(py_coeff)) + 1e-12)

    def test_acquire_ref_no_flux_norm(self):
        import lina_cpp
        from lina.llowfsc import acquire_ref as py_acquire_ref
        camlo, mask, _ref, _C = self._setup_random(seed=3)
        py_ref = py_acquire_ref(
            take_im_fun=lambda: camlo, take_im_params={},
            wfs_mask=mask.astype(bool), camlo_dark=0.0, flux_norm=False)
        cpp_ref, cpp_coeff = lina_cpp.llowfsc_acquire_ref(
            camlo, mask, dark_im=0.0, flux_norm=False)
        np.testing.assert_allclose(_to_np(py_ref), cpp_ref,
                                    rtol=1e-10, atol=1e-12)
        self.assertEqual(float(cpp_coeff), 0.0)

    def test_reconstruct_matches_python(self):
        import lina_cpp
        from lina.llowfsc import reconstruct as py_reconstruct
        for seed in (4, 5, 6):
            camlo, mask, ref, C = self._setup_random(seed=seed)
            py_coeff = py_reconstruct(
                camlo, ref, mask.astype(bool), C,
                dark_im=0.0, modes=(0, C.shape[0]),
                flux_norm=True, return_del_im=False)
            cpp_coeff = lina_cpp.llowfsc_reconstruct(
                camlo, ref, mask, C,
                mode_lo=0, mode_hi=C.shape[0],
                dark_im=0.0, flux_norm=True, return_del_im=False)
            np.testing.assert_allclose(_to_np(py_coeff), cpp_coeff,
                                        rtol=1e-10, atol=1e-12,
                                        err_msg=f"seed={seed}")

    def test_reconstruct_mode_subset(self):
        """[mode_lo, mode_hi) slicing matches Python's modes=(lo, hi)."""
        import lina_cpp
        from lina.llowfsc import reconstruct as py_reconstruct
        camlo, mask, ref, C = self._setup_random(seed=7, Nmodes=12)
        py_coeff = py_reconstruct(
            camlo, ref, mask.astype(bool), C,
            modes=(2, 8), flux_norm=True, return_del_im=False)
        cpp_coeff = lina_cpp.llowfsc_reconstruct(
            camlo, ref, mask, C,
            mode_lo=2, mode_hi=8, flux_norm=True, return_del_im=False)
        np.testing.assert_allclose(_to_np(py_coeff), cpp_coeff,
                                    rtol=1e-10, atol=1e-12)
        self.assertEqual(len(cpp_coeff), 6)

    def test_reconstruct_return_del_im(self):
        """Verify the return_del_im=True branch returns the same del_im
        as the Python implementation."""
        import lina_cpp
        from lina.llowfsc import reconstruct as py_reconstruct
        camlo, mask, ref, C = self._setup_random(seed=8)
        py_coeff, py_del_im = py_reconstruct(
            camlo, ref, mask.astype(bool), C,
            modes=(0, C.shape[0]), flux_norm=True, return_del_im=True)
        cpp_coeff, cpp_del_im = lina_cpp.llowfsc_reconstruct(
            camlo, ref, mask, C,
            mode_lo=0, mode_hi=C.shape[0],
            flux_norm=True, return_del_im=True)
        np.testing.assert_allclose(_to_np(py_coeff), cpp_coeff,
                                    rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(_to_np(py_del_im), cpp_del_im,
                                    rtol=1e-10, atol=1e-12)

    def test_compute_zpo_matches_python(self):
        """Match the math of lina.llowfsc.compute_zpo: sum-project a list
        of masked DM commands through (R . M . cmd) into a 2D image at the
        wfs_mask pixels."""
        import lina_cpp
        rng = np.random.default_rng(9)
        H = W = 24
        wfs_mask = (rng.standard_normal((H, W)) > 0.0).astype(np.uint8)
        Nmask = int(wfs_mask.sum())
        Nmodes = 6
        Ndm_side = 12
        Ndm = Ndm_side * Ndm_side

        R = rng.standard_normal((Nmask, Nmodes))
        M = rng.standard_normal((Nmodes, Ndm))

        # Three "DM commands" as 1D masked vectors.
        cmds = [rng.standard_normal(Ndm) for _ in range(3)]

        # Reference implementation -- direct numpy expression of compute_zpo.
        py_zpo = np.zeros((H, W))
        py_acc = np.zeros(Nmask)
        for cmd in cmds:
            py_acc += R.dot(M.dot(cmd))
        py_zpo[wfs_mask.astype(bool)] = py_acc

        cpp_zpo = lina_cpp.llowfsc_compute_zpo(cmds, wfs_mask, R, M)
        np.testing.assert_allclose(py_zpo, np.asarray(cpp_zpo),
                                    rtol=1e-12, atol=1e-13)

    def test_loop_step_matches_manual_recompute(self):
        """loop_step end-to-end: reconstruct + (-gain * coeff) + sum_i delta DM."""
        import lina_cpp
        from lina.llowfsc import reconstruct as py_reconstruct
        rng = np.random.default_rng(10)
        H = W = 32
        Nmodes = 8
        camlo = 5.0 + 0.1 * rng.standard_normal((H, W))
        mask = (rng.standard_normal((H, W)) > 0.0).astype(np.uint8)
        ref = 0.001 * rng.standard_normal((H, W))
        Nmask = int(mask.sum())
        C = rng.standard_normal((Nmodes, Nmask))

        dm_rows = dm_cols = 16
        Ndm = dm_rows * dm_cols
        gains = 0.4 + 0.1 * rng.standard_normal(Nmodes)
        ffo = 0.01 * rng.standard_normal(Nmodes)
        dm_modes = rng.standard_normal((Nmodes, Ndm))

        cpp_del_dm = lina_cpp.llowfsc_loop_step(
            camlo, ref, mask, C, dm_modes,
            dm_rows, dm_cols, gains,
            0, Nmodes, ffo,
            dark_im=0.0, flux_norm=True)

        # Reference path: Python reconstruct + manual gain/sum.
        py_coeff = _to_np(py_reconstruct(
            camlo, ref, mask.astype(bool), C,
            modes=(0, Nmodes), flux_norm=True, return_del_im=False))
        py_coeff -= ffo
        py_modal = -gains * py_coeff
        py_del_dm = np.einsum('i,ij->j', py_modal, dm_modes).reshape(dm_rows, dm_cols)

        np.testing.assert_allclose(_to_np(cpp_del_dm), py_del_dm,
                                    rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# Known divergences (regression checks; expected to fail until C++ is fixed)
# ---------------------------------------------------------------------------


class TestKnownDivergences(_LinaCppTest):
    """Tests that document and lock down the bugs identified in cpp/AUDIT.md.

    Each test is decorated with @expectedFailure: it asserts the *correct*
    Python behavior. When the C++ side is fixed, these tests will start
    passing — at which point remove the decorator.
    """

    def test_create_fourier_modes_match_python(self):
        """After fixing C++ dm::create_fourier_modes to call
        create_annular_mask with edge=0 (matching Python's current source),
        the mode set must match Python's mode set bit-exactly (modulo
        permutation of modes)."""
        import lina_cpp
        from lina import dm
        nact = 34
        mask = dm.create_mask(Nact=nact, return_np=True).astype(np.uint8)

        from scipy.spatial.distance import cdist
        configs = [
            dict(iwa=2.5, owa=10.0, fourier_sampling=0.75,
                 npsf=64, psf_pixelscale_lamD=0.354),
            dict(iwa=3.0, owa=12.0, fourier_sampling=1.0,
                 npsf=64, psf_pixelscale_lamD=0.5),
            dict(iwa=2.0, owa=8.0, fourier_sampling=0.5,
                 npsf=128, psf_pixelscale_lamD=0.3),
        ]
        for cfg in configs:
            for which in ('cos', 'sin', 'both'):
                cpp = lina_cpp.create_fourier_modes(
                    mask, rotation=0.0, which=which, **cfg)
                py = dm.create_fourier_modes(
                    mask.astype(bool), rotation=0, which=which, **cfg)
                py_flat = np.asarray(py).reshape(py.shape[0], -1)
                self.assertEqual(cpp.shape, py_flat.shape,
                                 msg=f"shape mismatch for {cfg=} {which=}")
                D = cdist(py_flat, cpp)
                worst = float(D.min(axis=1).max())
                self.assertLess(worst, 1e-10,
                                msg=f"worst mode residual {worst} "
                                    f"for {cfg=} {which=}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
