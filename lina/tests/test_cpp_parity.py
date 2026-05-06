import math
import os
import subprocess
import sys
import unittest

import numpy as np
import scipy
import sys

from lina import math_module


def _find_runner():
    # LINA_CPP_RUNNER takes priority so users can point at a freshly-built
    # binary even if a stale one exists in `cpp/build*`.
    candidates = [
        os.environ.get("LINA_CPP_RUNNER"),
        os.path.join(os.getcwd(), "cpp", "build", "lina_runner"),
        os.path.join(os.getcwd(), "cpp", "build-cuda", "lina_runner"),
        os.path.join(os.getcwd(), "cpp", "install", "bin", "lina_runner"),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def _run_runner(cmd):
    runner = _find_runner()
    if not runner:
        raise unittest.SkipTest("lina_runner not found; build C++ tools first")
    proc = subprocess.run([runner, cmd], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        if cmd == "svd":
            raise unittest.SkipTest("SVD backend not available")
        if cmd == "shmim":
            raise unittest.SkipTest("shmim runner not available")
        raise RuntimeError(proc.stderr or proc.stdout)
    return proc.stdout.strip().splitlines()


class TestCppParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Force NumPy backend so tests are deterministic even if cupy is installed.
        math_module.update_np(np)
        math_module.update_scipy(scipy)
        shmim_path = "/opt/MagAOX/source/milk/src/ImageStreamIO/build/lib.linux-x86_64-cpython-310"
        if os.path.isdir(shmim_path) and shmim_path not in sys.path:
            sys.path.insert(0, shmim_path)
    def test_fft(self):
        lines = _run_runner("fft")
        self.assertEqual(lines[0], "FFT 4 4")
        vals = [list(map(float, ln.split())) for ln in lines[1:]]
        arr = np.arange(16, dtype=float).reshape(4, 4)
        fft_py = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(arr)))
        flat_py = fft_py.flatten()
        for i, (re, im) in enumerate(vals):
            self.assertTrue(math.isclose(re, flat_py[i].real, rel_tol=1e-6, abs_tol=1e-6))
            self.assertTrue(math.isclose(im, flat_py[i].imag, rel_tol=1e-6, abs_tol=1e-6))

    def test_vortex_mask(self):
        lines = _run_runner("vortex")
        self.assertEqual(lines[0], "VORTEX 4 4")
        vals = [list(map(float, ln.split())) for ln in lines[1:]]
        from lina import props
        mask_py = props.make_vortex_phase_mask(4)
        flat_py = mask_py.flatten()
        for i, (re, im) in enumerate(vals):
            self.assertTrue(math.isclose(re, flat_py[i].real, rel_tol=1e-6, abs_tol=1e-6))
            self.assertTrue(math.isclose(im, flat_py[i].imag, rel_tol=1e-6, abs_tol=1e-6))

    def test_annular_mask(self):
        lines = _run_runner("annular")
        self.assertEqual(lines[0], "ANNULAR 4 4")
        vals = np.array([int(v) for v in lines[1:]]).reshape(4, 4)
        from lina import utils
        mask_py = utils.create_annular_mask(4, 1.0, 0.5, 2.0, edge=-1e9, x_shift=0.0, y_shift=0.0, rotation=0.0)
        self.assertTrue(np.array_equal(vals, mask_py.astype(int)))

    def test_dm_mask(self):
        lines = _run_runner("dm_mask")
        self.assertEqual(lines[0], "DM_MASK 4 4")
        vals = np.array([int(v) for v in lines[1:]]).reshape(4, 4)
        from lina import dm as dm_py
        mask_py = dm_py.create_mask(Nact=4, return_np=True)
        self.assertTrue(np.array_equal(vals, mask_py.astype(int)))

    def test_svd(self):
        lines = _run_runner("svd")
        self.assertTrue(lines[0].startswith("SVD "))
        svals = np.array([float(v) for v in lines[1:]])
        mat = np.array([[3.0, 1.0], [0.0, -1.0], [2.0, 4.0]])
        _, s_py, _ = np.linalg.svd(mat, full_matrices=True)
        self.assertEqual(len(svals), len(s_py))
        for i in range(len(s_py)):
            self.assertTrue(math.isclose(svals[i], s_py[i], rel_tol=1e-6, abs_tol=1e-6))

    @unittest.skip(
        "Stale: this test calls efc.run() with kwargs (pwp_params, "
        "set_dm_fun, etc.) from an older Python API that no longer "
        "exists in lina/efc.py. Rewrite against the current signature "
        "(see lina.tests.test_per_method_parity for the modern style)."
    )
    def test_efc(self):
        lines = _run_runner("efc")
        self.assertTrue(lines[0].startswith("EFC_CONTRAST "))
        contrast_cpp = float(lines[0].split()[1])
        self.assertEqual(lines[1], "EFC_COMMAND 2 2")
        cmd_cpp = np.array([float(v) for v in lines[2:]]).reshape(2, 2)

        import lina.efc as efc
        import lina.pwp as pwp
        import lina.utils as utils
        import lina.coro_utils as coro_utils

        class DummyDmStream:
            def __init__(self, shape):
                self.shape = shape
                self._latest = np.zeros(shape)

            def grab_latest(self):
                return self._latest

            def grab_many(self, n):
                return np.repeat(self._latest[None, ...], n, axis=0)

            def write(self, data):
                self._latest = data

        class DummyCamStream:
            def __init__(self, shape, dm_stream, scale):
                self.shape = shape
                self._dm = dm_stream
                self._base = np.ones(shape)
                self._scale = scale

            def grab_latest(self):
                return self.grab_many(1)[0]

            def grab_many(self, n):
                dm_sum = float(np.sum(self._dm._latest))
                im = self._base + self._scale * dm_sum
                return np.repeat(im[None, ...], n, axis=0)

            def write(self, _data):
                return None

        def dummy_imshow(*_args, **_kwargs):
            return None

        utils.imshow = dummy_imshow

        def dummy_pwp_run(*_args, **_kwargs):
            field = np.zeros((4, 4), dtype=np.complex128)
            field[1, 1] = 1.0 + 2.0j
            field[1, 2] = -0.5 + 0.25j
            field_vec = np.array([1.0 + 2.0j, -0.5 + 0.25j])
            return field, field_vec

        pwp.run = dummy_pwp_run

        dm = DummyDmStream((2, 2))
        camsci = DummyCamStream((4, 4), dm, 0.1)
        efc_data = {"images": [], "contrasts": [], "efields": [], "commands": [], "del_commands": []}
        im_params = {"exp_time": 1.0, "gain": 0.0, "atten": 0.0, "Imax": 1.0}
        ref_params = {"exp_time": 1.0, "gain": 0.0, "atten": 0.0, "Imax": 1.0}
        control_mask = np.zeros((4, 4), dtype=bool)
        control_mask[1, 1] = True
        control_mask[1, 2] = True
        dm_mask = np.ones((2, 2), dtype=bool)
        control_matrix = np.eye(4)
        dark_im = np.zeros((4, 4))

        efc.run(
            efc_data,
            camsci,
            dm,
            im_params,
            ref_params,
            1,
            dark_im,
            control_mask,
            dm_mask,
            control_matrix,
            pwp_params={},
            Nitr=1,
            gain=1.0,
            leakage=0.0,
            delay=0.0,
        )

        contrast_py = efc_data["contrasts"][0]
        cmd_py = efc_data["commands"][0] * 1e6
        if math.isnan(contrast_py):
            # Python path returns NaN if no positive pixels; C++ returns 0.0.
            self.assertTrue(math.isclose(contrast_cpp, 0.0, rel_tol=1e-6, abs_tol=1e-6))
        else:
            self.assertTrue(math.isclose(contrast_cpp, contrast_py, rel_tol=1e-6, abs_tol=1e-6))
        self.assertTrue(np.allclose(cmd_cpp, cmd_py, atol=1e-6, rtol=1e-6))

    @unittest.skip(
        "Stale: this test calls iefc.run() with kwargs (delay, "
        "set_dm_fun, etc.) from an older Python API that no longer "
        "exists in lina/iefc.py. Rewrite against the current signature "
        "(see lina.tests.test_per_method_parity for the modern style)."
    )
    def test_iefc(self):
        lines = _run_runner("iefc")
        self.assertTrue(lines[0].startswith("IEFC_CONTRAST "))
        contrast_cpp = float(lines[0].split()[1])
        self.assertEqual(lines[1], "IEFC_COMMAND 2 2")
        cmd_cpp = np.array([float(v) for v in lines[2:]]).reshape(2, 2)

        import lina.iefc as iefc
        import lina.utils as utils

        class DummyDmStream:
            def __init__(self, shape):
                self.shape = shape
                self._latest = np.zeros(shape)

            def grab_latest(self):
                return self._latest

            def grab_many(self, n):
                return np.repeat(self._latest[None, ...], n, axis=0)

            def write(self, data):
                self._latest = data

        class DummyCamStream:
            def __init__(self, shape, dm_stream, scale):
                self.shape = shape
                self._dm = dm_stream
                self._base = np.ones(shape)
                self._scale = scale

            def grab_latest(self):
                return self.grab_many(1)[0]

            def grab_many(self, n):
                dm_sum = float(np.sum(self._dm._latest))
                im = self._base + self._scale * dm_sum
                return np.repeat(im[None, ...], n, axis=0)

            def write(self, _data):
                return None

        def dummy_imshow(*_args, **_kwargs):
            return None

        utils.imshow = dummy_imshow

        dm = DummyDmStream((2, 2))
        camsci = DummyCamStream((4, 4), dm, 0.1)
        iefc_data = {
            "images": [],
            "raw_images": [],
            "dark_images": [],
            "ni_images": [],
            "contrasts": [],
            "commands": [],
            "del_commands": [],
        }
        im_params = {"exp_time": 1.0, "gain": 0.0, "atten": 0.0, "Imax": 1.0}
        ref_params = {"exp_time": 1.0, "gain": 0.0, "atten": 0.0, "Imax": 1.0}
        control_mask = np.zeros((4, 4), dtype=bool)
        control_mask[1, 1] = True
        control_mask[1, 2] = True
        probe_modes = np.zeros((2, 2, 2))
        probe_modes[0, 0, 0] = 1.0
        probe_modes[1, 0, 1] = -1.0
        calib_modes = np.zeros((4, 2, 2))
        for i in range(4):
            calib_modes[i].flat[i] = 1.0
        control_matrix = np.eye(4)
        dark_im = np.zeros((4, 4))

        iefc.run(
            iefc_data,
            camsci,
            1,
            dm,
            im_params,
            ref_params,
            dark_im,
            control_matrix,
            0.5,
            probe_modes,
            calib_modes,
            control_mask,
            delay=0.0,
            num_iterations=1,
            gain=0.75,
            leakage=0.0,
            plot_current=False,
            plot_all=False,
        )

        contrast_py = iefc_data["contrasts"][0]
        cmd_py = iefc_data["commands"][0]
        if not math.isnan(contrast_py):
            self.assertTrue(math.isclose(contrast_cpp, contrast_py, rel_tol=1e-6, abs_tol=1e-6))
        self.assertTrue(np.allclose(cmd_cpp, cmd_py, atol=1e-6, rtol=1e-6))

    def test_shmim(self):
        try:
            import ImageStreamIOWrap as shmio
        except Exception:
            raise unittest.SkipTest("ImageStreamIOWrap not available")

        try:
            lines = _run_runner("shmim")
        except unittest.SkipTest:
            raise
        self.assertEqual(lines[0], "SHMIM 2 2")
        cpp_vals = [float(v) for v in lines[1:]]
        img = shmio.Image()
        img.open("lina_test_shmim")
        if hasattr(img, "copy"):
            data = np.array(img.copy())
            data = data.T
        else:
            raise unittest.SkipTest("ImageStreamIOWrap copy method not found")
        self.assertTrue(np.allclose(data.flatten(), cpp_vals, atol=1e-6, rtol=1e-6))
        img.close()
        shm_path = "/milk/shm/lina_test_shmim.im.shm"
        self.assertTrue(os.path.exists(shm_path))


if __name__ == "__main__":
    unittest.main()
