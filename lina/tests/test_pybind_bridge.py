import os
import time
import unittest

import numpy as np

REPORT_PATH = os.environ.get(
    "LINA_BENCH_REPORT",
    os.path.join(os.getcwd(), "lina", "tests", "bench_report.md"),
)

class TestPybindBridge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("LINA_RUN_PYBIND") == "0":
            raise unittest.SkipTest("Pybind tests disabled")

    def test_fft_cpu_parity(self):
        try:
            import lina_cpp
        except Exception as exc:
            raise unittest.SkipTest(f"lina_cpp unavailable: {exc}") from exc

        rng = np.random.default_rng(42)
        arr = rng.random((64, 64)) + 1j * rng.random((64, 64))
        arr = arr.astype(np.complex128, copy=False)

        # Warmup
        for _ in range(2):
            _ = lina_cpp.fft_cpu(arr)
            _ = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(arr)))

        start = time.perf_counter()
        out_cpp = lina_cpp.fft_cpu(arr)
        cpp_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        out_py = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(arr)))
        py_ms = (time.perf_counter() - start) * 1000.0

        np.testing.assert_allclose(out_cpp, out_py, rtol=1e-6, atol=1e-6)
        print(f"PYBIND_FFT_CPU {cpp_ms:.3f} {py_ms:.3f}")

    def test_svd_cpu_parity(self):
        try:
            import lina_cpp
        except Exception as exc:
            raise unittest.SkipTest(f"lina_cpp unavailable: {exc}") from exc

        rng = np.random.default_rng(7)
        mat = rng.random((128, 64), dtype=np.float32)

        for _ in range(2):
            _ = lina_cpp.svd_float_cpu(mat)
            _ = np.linalg.svd(mat, full_matrices=True)

        start = time.perf_counter()
        u_cpp, s_cpp, vt_cpp = lina_cpp.svd_float_cpu(mat)
        cpp_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        u_py, s_py, vt_py = np.linalg.svd(mat, full_matrices=True)
        py_ms = (time.perf_counter() - start) * 1000.0

        np.testing.assert_allclose(s_cpp, s_py, rtol=1e-4, atol=1e-4)
        self.assertEqual(u_cpp.shape, u_py.shape)
        self.assertEqual(vt_cpp.shape, vt_py.shape)
        print(f"PYBIND_SVD_CPU {cpp_ms:.3f} {py_ms:.3f}")

    def test_three_group_benchmarks(self):
        raise unittest.SkipTest("Group benchmarks are now reported in bench_report.md")
