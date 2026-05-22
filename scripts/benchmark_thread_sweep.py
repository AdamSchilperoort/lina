#!/usr/bin/env python3
"""Sweep CPU thread counts for lina_cpp math kernels."""

from __future__ import annotations

import argparse
import os
import time
import numpy as np


def avg_ms(fn, iters=10, warmup=2):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) * 1000.0 / max(1, iters)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", default="1,2,4,8,12,16")
    ap.add_argument("--iters", type=int, default=8)
    args = ap.parse_args()

    import lina_cpp
    ths = [int(x.strip()) for x in args.threads.split(",") if x.strip()]

    rng = np.random.default_rng(0)
    fft_arr = (rng.standard_normal((1024, 1024)) + 1j * rng.standard_normal((1024, 1024))).astype(np.complex128)
    svd_arr = rng.standard_normal((2000, 1000), dtype=np.float32)
    A = rng.standard_normal((4096, 1024))
    x = rng.standard_normal(1024)

    print(f"OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS')}")
    print(f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")
    print("| threads | fft_cpu_1024 | svd_cpu_2000x1000 | gemv_4096x1024 |")
    print("| ---: | ---: | ---: | ---: |")
    for t in ths:
        lina_cpp.set_num_threads(t)
        actual = lina_cpp.get_num_threads()
        fft_ms = avg_ms(lambda: lina_cpp.fft_cpu(fft_arr), args.iters)
        svd_ms = avg_ms(lambda: lina_cpp.svd_float_cpu(svd_arr), args.iters)
        gemv_ms = avg_ms(lambda: lina_cpp.gemv(A, x, False), args.iters)
        print(f"| {actual} | {fft_ms:.3f} | {svd_ms:.3f} | {gemv_ms:.3f} |")


if __name__ == "__main__":
    main()
