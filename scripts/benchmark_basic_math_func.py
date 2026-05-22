#!/usr/bin/env python3
"""Basic math benchmark split: kernel-only, transfer-only, end-to-end."""

from __future__ import annotations

import argparse
import json
import time
import numpy as np

FFT_SIZES_FULL = [1024, 2048, 4096]
FFT_SIZES_SMALL = [1024]

SVD_SIZES_FULL = [
    (2000, 1000),
    (2000, 2000),
    (2000, 4000),
    (2000, 7500),
    (2000, 15000),
    (5000, 1000),
    (5000, 2000),
    (5000, 4000),
    (5000, 7500),
    (10000, 1000),
    (10000, 2000),
    (10000, 4000),
    (10000, 7500),
    (10000, 15000),
    (20000, 1000),
    (20000, 2000),
    (20000, 4000),
    (20000, 7500),
    (20000, 15000),
    (50000, 1000),
    (50000, 2000),
    (50000, 4000),
    (50000, 7500),
]
SVD_SIZES_SMALL = [
    (2000, 1000),
    (5000, 2000),
    (10000, 2000),
]


def _load_grid_config(path):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_fft_sizes(values):
    out = []
    for v in values:
        n = int(v)
        if n > 0:
            out.append(n)
    if not out:
        raise ValueError("FFT size list is empty after parsing.")
    return out


def _normalize_svd_sizes(values):
    out = []
    for item in values:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"Invalid SVD size entry: {item!r} (expected [m, n])")
        m, n = int(item[0]), int(item[1])
        if m <= 0 or n <= 0:
            raise ValueError(f"Invalid SVD size entry: {item!r} (dimensions must be positive)")
        out.append((m, n))
    if not out:
        raise ValueError("SVD size list is empty after parsing.")
    return out


def avg_ms(fn, iters=20, warmup=2):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) * 1000.0 / max(1, iters)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument(
        "--full-compute",
        action="store_true",
        help="Run full FFT/SVD dimension sets (default runs capped subset).",
    )
    ap.add_argument(
        "--sizes-file",
        type=str,
        default="",
        help="Optional JSON file with fft/svd size grids.",
    )
    args = ap.parse_args()

    import lina_cpp
    try:
        import cupy as cp
        has_cupy = True
    except Exception:
        has_cupy = False
        cp = None

    print("| op | size | mode | ms |")
    print("| --- | --- | --- | ---:|")

    cfg = _load_grid_config(args.sizes_file)
    if args.full_compute:
        fft_raw = cfg.get("fft_sizes_full", cfg.get("fft_sizes", FFT_SIZES_FULL))
        svd_raw = cfg.get("svd_sizes_full", cfg.get("svd_sizes", SVD_SIZES_FULL))
    else:
        fft_raw = cfg.get("fft_sizes_small", FFT_SIZES_SMALL)
        svd_raw = cfg.get("svd_sizes_small", SVD_SIZES_SMALL)
    fft_sizes = _normalize_fft_sizes(fft_raw)
    svd_sizes = _normalize_svd_sizes(svd_raw)

    fft_iters_map = {1024: min(args.iters, 3), 2048: min(args.iters, 1), 4096: min(args.iters, 1)}

    # FFT split
    for n in fft_sizes:
        iters = max(1, fft_iters_map.get(n, args.iters))
        arr = np.random.default_rng(1234 + n).random((n, n)) + 1j * np.random.default_rng(5678 + n).random((n, n))
        arr = arr.astype(np.complex128)
        print(f"| FFT | {n}x{n} | lina_cpp_cpu_e2e | {avg_ms(lambda: lina_cpp.fft_cpu(arr), iters):.3f} |")
        print(f"| FFT | {n}x{n} | lina_cpp_gpu_e2e | {avg_ms(lambda: lina_cpp.fft_gpu(arr), iters):.3f} |")
        if has_cupy:
            arr_gpu = cp.asarray(arr)

            def xfer_only():
                t = cp.asarray(arr)
                _ = t.get()
                cp.cuda.Stream.null.synchronize()

            fft_kernel = avg_ms(lambda: (cp.fft.ifftshift(cp.fft.fft2(cp.fft.fftshift(arr_gpu))), cp.cuda.Stream.null.synchronize()), iters)
            fft_xfer = avg_ms(xfer_only, iters)
            fft_e2e = avg_ms(lambda: (cp.fft.ifftshift(cp.fft.fft2(cp.fft.fftshift(cp.asarray(arr)))).get(), cp.cuda.Stream.null.synchronize()), iters)
            print(f"| FFT | {n}x{n} | cupy_gpu_kernel | {fft_kernel:.3f} |")
            print(f"| FFT | {n}x{n} | cupy_gpu_xfer | {fft_xfer:.3f} |")
            print(f"| FFT | {n}x{n} | cupy_gpu_e2e | {fft_e2e:.3f} |")

    # SVD split
    for m, n in svd_sizes:
        # Full-compute sizes are expensive; keep large problems to one pass.
        iters = max(1, min(args.iters, 1 if (m * n) >= 8_000_000 else 2))
        try:
            mat = np.random.default_rng(42 + m + n).random((m, n), dtype=np.float32)
        except Exception as exc:
            print(f"| SVD | {m}x{n} | alloc | ERROR ({exc}) |")
            continue
        try:
            print(f"| SVD | {m}x{n} | numpy_cpu_e2e | {avg_ms(lambda: np.linalg.svd(mat, full_matrices=True), iters):.3f} |")
        except Exception as exc:
            print(f"| SVD | {m}x{n} | numpy_cpu_e2e | ERROR ({exc}) |")
        try:
            print(f"| SVD | {m}x{n} | lina_cpp_cpu_e2e | {avg_ms(lambda: lina_cpp.svd_float_cpu(mat), iters):.3f} |")
        except Exception as exc:
            print(f"| SVD | {m}x{n} | lina_cpp_cpu_e2e | ERROR ({exc}) |")
        try:
            print(f"| SVD | {m}x{n} | lina_cpp_gpu_e2e | {avg_ms(lambda: lina_cpp.svd_float_gpu(mat), iters):.3f} |")
        except Exception as exc:
            print(f"| SVD | {m}x{n} | lina_cpp_gpu_e2e | ERROR ({exc}) |")
        if has_cupy:
            try:
                mat_gpu = cp.asarray(mat)
            except Exception as exc:
                print(f"| SVD | {m}x{n} | cupy_gpu_alloc | ERROR ({exc}) |")
                continue

            def xfer_only():
                t = cp.asarray(mat)
                _ = t.get()
                cp.cuda.Stream.null.synchronize()

            try:
                svd_kernel = avg_ms(lambda: (cp.linalg.svd(mat_gpu, full_matrices=True), cp.cuda.Stream.null.synchronize()), iters)
                print(f"| SVD | {m}x{n} | cupy_gpu_kernel | {svd_kernel:.3f} |")
            except Exception as exc:
                print(f"| SVD | {m}x{n} | cupy_gpu_kernel | ERROR ({exc}) |")
            try:
                svd_xfer = avg_ms(xfer_only, iters)
                print(f"| SVD | {m}x{n} | cupy_gpu_xfer | {svd_xfer:.3f} |")
            except Exception as exc:
                print(f"| SVD | {m}x{n} | cupy_gpu_xfer | ERROR ({exc}) |")
            try:
                svd_e2e = avg_ms(lambda: (tuple(x.get() for x in cp.linalg.svd(cp.asarray(mat), full_matrices=True)), cp.cuda.Stream.null.synchronize()), iters)
                print(f"| SVD | {m}x{n} | cupy_gpu_e2e | {svd_e2e:.3f} |")
            except Exception as exc:
                print(f"| SVD | {m}x{n} | cupy_gpu_e2e | ERROR ({exc}) |")


if __name__ == "__main__":
    main()
