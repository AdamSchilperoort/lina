import os
import subprocess
import tempfile
import time
import unittest

import numpy as np


REPORT_PATH = os.environ.get(
    "LINA_BENCH_REPORT",
    os.path.join(os.getcwd(), "lina", "tests", "bench_report.md"),
)


def _find_runner():
    candidates = [
        os.path.join(os.getcwd(), "cpp", "build", "lina_runner"),
        os.path.join(os.getcwd(), "cpp", "build-cuda", "lina_runner"),
        os.environ.get("LINA_CPP_RUNNER"),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def _run_runner(cmd, env=None):
    runner = _find_runner()
    if not runner:
        raise unittest.SkipTest("lina_runner not found; build C++ tools first")
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    proc = subprocess.run([runner, cmd], capture_output=True, text=True, check=False, env=merged_env)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout)
    return proc.stdout.strip()

def _parse_bench_ms(line, prefix):
    parts = line.strip().split()
    if len(parts) >= 3 and parts[0] == prefix:
        try:
            return float(parts[-1])
        except ValueError:
            return None
    return None


class TestBenchmarks(unittest.TestCase):
    report_rows = []
    fft_rows = []
    svd_rows = []
    has_lina_cpp = False
    lina_cpp = None

    @classmethod
    def _record(cls, section, size_label, cpp_ms, py_ms, pybind_ms):
        entries = {
            "C++": cpp_ms,
            "Python": py_ms,
            "Pybind": pybind_ms,
        }
        fastest = "N/A"
        best_ms = None
        for name, val in entries.items():
            if val is None:
                continue
            if best_ms is None or val < best_ms:
                best_ms = val
                fastest = name
        cpp_text = f"{cpp_ms:.3f}" if cpp_ms is not None else "N/A"
        py_text = f"{py_ms:.3f}" if py_ms is not None else "N/A"
        pybind_text = f"{pybind_ms:.3f}" if pybind_ms is not None else "N/A"
        cls.report_rows.append(
            f"| {section} | {size_label} | {cpp_text} | {py_text} | {pybind_text} | {fastest} |"
        )
        if section.startswith("FFT"):
            cls.fft_rows.append((size_label, section, cpp_ms))
        if section.startswith("SVD"):
            cls.svd_rows.append((size_label, section, cpp_ms))
        print(f"{section}_FASTEST {size_label} {fastest}")

    @classmethod
    def tearDownClass(cls):
        if not cls.report_rows:
            return
        header = [
            "# Lina Benchmark Report",
            "",
            "| Test | Size | C++ ms | Python ms | Pybind ms | Fastest |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        extras = []
        fft_cpu = {}
        fft_gpu = {}
        for size_label, section, cpp_ms in cls.fft_rows:
            if section == "FFT-CPU":
                fft_cpu[size_label] = cpp_ms
            if section == "FFT-GPU":
                fft_gpu[size_label] = cpp_ms
        svd_cpu = {}
        svd_gpu = {}
        for size_label, section, cpp_ms in cls.svd_rows:
            if section == "SVD-CPU":
                svd_cpu[size_label] = cpp_ms
            if section == "SVD-GPU":
                svd_gpu[size_label] = cpp_ms

        def find_crossover(cpu_map, gpu_map):
            sizes = []
            for key in cpu_map.keys():
                dims = key.split("x")
                if len(dims) == 2:
                    sizes.append((int(dims[0]), key))
            for _, key in sorted(sizes):
                if key in gpu_map and gpu_map[key] < cpu_map[key]:
                    return key
            return "N/A"

        fft_cross = find_crossover(fft_cpu, fft_gpu)
        svd_cross = find_crossover(svd_cpu, svd_gpu)
        extras.append("")
        extras.append("## CPU/GPU Crossover")
        extras.append(f"- FFT crossover: {fft_cross}")
        extras.append(f"- SVD crossover: {svd_cross}")

        contents = "\n".join(header + cls.report_rows + extras) + "\n"
        os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
        with open(REPORT_PATH, "w", encoding="utf-8") as handle:
            handle.write(contents)
        print(f"BENCH_REPORT_WRITTEN {REPORT_PATH}")

    @classmethod
    def setUpClass(cls):
        if os.environ.get("LINA_RUN_BENCHMARKS") != "1":
            raise unittest.SkipTest("Benchmarks disabled (set LINA_RUN_BENCHMARKS=1)")
        try:
            import lina_cpp
            cls.has_lina_cpp = True
            cls.lina_cpp = lina_cpp
        except Exception:
            cls.has_lina_cpp = False
        try:
            import cupy as cp
            cls.cupy = cp
            cls.has_cupy = True
        except Exception:
            cls.cupy = None
            cls.has_cupy = False
        from lina import math_module as math_module
        from lina import props as lina_props
        cls.math_module = math_module
        cls.lina_props = lina_props

    def test_fft_benchmark(self):
        sizes = [16, 32, 64, 128, 256, 512, 1024]
        iters_map = {16: 10, 32: 10, 64: 10, 128: 10, 256: 10, 512: 5, 1024: 3}
        for n in sizes:
            iters = iters_map[n]
            rng = np.random.default_rng(1234 + n)
            arr = rng.random((n, n), dtype=np.float64)
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                arr.tofile(tmp)
                data_path = tmp.name
            out_cpu = _run_runner(
                "bench_fft_cpu",
                env={
                    "LINA_BENCH_FFT_N": str(n),
                    "LINA_BENCH_FFT_ITERS": str(iters),
                    "LINA_BENCH_DATA_PATH": data_path,
                },
            )
            self.assertTrue(out_cpu.startswith("BENCH_FFT_CPU"))
            print(out_cpu)
            out_gpu = _run_runner(
                "bench_fft_gpu",
                env={
                    "LINA_BENCH_FFT_N": str(n),
                    "LINA_BENCH_FFT_ITERS": str(iters),
                    "LINA_BENCH_DATA_PATH": data_path,
                },
            )
            self.assertTrue(out_gpu.startswith("BENCH_FFT_GPU"))
            print(out_gpu)
            self.math_module.update_np(np)
            for _ in range(2):
                _ = self.lina_props.fft(arr)
            start = time.perf_counter()
            for _ in range(iters):
                _ = self.lina_props.fft(arr)
            elapsed = (time.perf_counter() - start) * 1000.0
            avg_ms = elapsed / iters
            print(f"PY_BENCH_FFT {n} {iters} {avg_ms:.3f}")
            pybind_cpu = None
            pybind_gpu = None
            if self.has_lina_cpp:
                for _ in range(2):
                    _ = self.lina_cpp.fft_cpu(arr)
                start = time.perf_counter()
                for _ in range(iters):
                    _ = self.lina_cpp.fft_cpu(arr)
                pybind_cpu = (time.perf_counter() - start) * 1000.0 / iters
                for _ in range(2):
                    _ = self.lina_cpp.fft_gpu(arr)
                start = time.perf_counter()
                for _ in range(iters):
                    _ = self.lina_cpp.fft_gpu(arr)
                pybind_gpu = (time.perf_counter() - start) * 1000.0 / iters

            cpp_ms_cpu = _parse_bench_ms(out_cpu, "BENCH_FFT_CPU")
            if cpp_ms_cpu is not None:
                self._record("FFT-CPU", f"{n}x{n}", cpp_ms_cpu, avg_ms, pybind_cpu)
            cpp_ms_gpu = _parse_bench_ms(out_gpu, "BENCH_FFT_GPU")
            py_gpu_ms = None
            if self.has_cupy:
                self.math_module.update_np(self.cupy)
                arr_gpu = self.cupy.asarray(arr)
                for _ in range(2):
                    _ = self.lina_props.fft(arr_gpu)
                self.cupy.cuda.Stream.null.synchronize()
                start = time.perf_counter()
                for _ in range(iters):
                    _ = self.lina_props.fft(arr_gpu)
                self.cupy.cuda.Stream.null.synchronize()
                py_gpu_ms = (time.perf_counter() - start) * 1000.0 / iters
            if cpp_ms_gpu is not None:
                self._record("FFT-GPU", f"{n}x{n}", cpp_ms_gpu, py_gpu_ms, pybind_gpu)
            os.unlink(data_path)

    def test_svd_benchmark(self):
        sizes = [(100, 50), (400, 200), (2000, 1000), (5000, 2000)]
        for m, n in sizes:
            rng = np.random.default_rng(5678 + m + n)
            mat = rng.random((m, n), dtype=np.float32)
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                mat.tofile(tmp)
                data_path = tmp.name
            iters = 10
            out_cpu = _run_runner(
                "bench_svd_size_cpu",
                env={
                    "LINA_BENCH_SVD_M": str(m),
                    "LINA_BENCH_SVD_N": str(n),
                    "LINA_BENCH_SVD_ITERS": str(iters),
                    "LINA_BENCH_DATA_PATH": data_path,
                },
            )
            self.assertTrue(out_cpu.startswith("BENCH_SVD_SIZE_CPU"))
            print(out_cpu)
            out_gpu = _run_runner(
                "bench_svd_size_gpu",
                env={
                    "LINA_BENCH_SVD_M": str(m),
                    "LINA_BENCH_SVD_N": str(n),
                    "LINA_BENCH_SVD_ITERS": str(iters),
                    "LINA_BENCH_DATA_PATH": data_path,
                },
            )
            self.assertTrue(out_gpu.startswith("BENCH_SVD_SIZE_GPU"))
            print(out_gpu)
            for _ in range(2):
                np.linalg.svd(mat, full_matrices=True)
            start = time.perf_counter()
            for _ in range(iters):
                np.linalg.svd(mat, full_matrices=True)
            elapsed = (time.perf_counter() - start) * 1000.0
            avg_ms = elapsed / iters
            print(f"PY_BENCH_SVD_SIZE {m} {n} {avg_ms:.3f}")
            pybind_cpu = None
            pybind_gpu = None
            if self.has_lina_cpp:
                for _ in range(2):
                    _ = self.lina_cpp.svd_float_cpu(mat)
                start = time.perf_counter()
                for _ in range(iters):
                    _ = self.lina_cpp.svd_float_cpu(mat)
                pybind_cpu = (time.perf_counter() - start) * 1000.0 / iters
                for _ in range(2):
                    _ = self.lina_cpp.svd_float_gpu(mat)
                start = time.perf_counter()
                for _ in range(iters):
                    _ = self.lina_cpp.svd_float_gpu(mat)
                pybind_gpu = (time.perf_counter() - start) * 1000.0 / iters

            cpp_ms_cpu = _parse_bench_ms(out_cpu, "BENCH_SVD_SIZE_CPU")
            if cpp_ms_cpu is not None:
                self._record("SVD-CPU", f"{m}x{n}", cpp_ms_cpu, avg_ms, pybind_cpu)
            cpp_ms_gpu = _parse_bench_ms(out_gpu, "BENCH_SVD_SIZE_GPU")
            py_gpu_ms = None
            if self.has_cupy:
                mat_gpu = self.cupy.asarray(mat)
                for _ in range(2):
                    _ = self.cupy.linalg.svd(mat_gpu, full_matrices=True)
                self.cupy.cuda.Stream.null.synchronize()
                start = time.perf_counter()
                for _ in range(iters):
                    _ = self.cupy.linalg.svd(mat_gpu, full_matrices=True)
                self.cupy.cuda.Stream.null.synchronize()
                py_gpu_ms = (time.perf_counter() - start) * 1000.0 / iters
            if cpp_ms_gpu is not None:
                self._record("SVD-GPU", f"{m}x{n}", cpp_ms_gpu, py_gpu_ms, pybind_gpu)
            os.unlink(data_path)

    def test_svd_large_benchmark(self):
        if os.environ.get("LINA_RUN_LARGE_SVD") != "1":
            raise unittest.SkipTest("Large SVD benchmark disabled (set LINA_RUN_LARGE_SVD=1)")
        # Covered by size sweep in test_svd_benchmark; keep for backward compatibility.
        self.assertTrue(True)

    def test_efc_benchmark(self):
        out = _run_runner("bench_efc")
        self.assertTrue(out.startswith("BENCH_EFC"))

    def test_iefc_benchmark(self):
        out = _run_runner("bench_iefc")
        self.assertTrue(out.startswith("BENCH_IEFC"))
