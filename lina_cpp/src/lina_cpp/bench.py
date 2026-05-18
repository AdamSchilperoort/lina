"""Direct speed comparison: lina_cpp (C++) vs pure-Python lina (numpy/cupy).

Usage::

    python -m lina_cpp.bench
    python -m lina_cpp.bench --sizes 512 1024 2048 --repeats 5

For each math primitive (fft, ifft, mft_forward, ang_spec,
make_vortex_phase_mask, get_fresnel_TF) we time the same logical
operation across as many of the following stacks as are available on
the system:

* ``numpy``                     -- raw numpy/numpy.fft (single-threaded)
* ``cupy``                      -- raw cupy/cupy.fft on the GPU
* ``lina``                      -- pure-Python lina, using whichever xp
                                   it resolved at import time
* ``lina_cpp CPU``              -- our C++ extension with FFTW/OpenBLAS
* ``lina_cpp GPU``              -- our C++ extension with cuFFT/cuBLAS

Each call is run once for warmup (so JIT, kernel build, plan caches,
and first-touch page allocations don't pollute timings), then timed
over ``--repeats`` runs; the best time is reported. For GPU stacks we
explicitly synchronise the device before stopping the timer so we're
measuring real GPU compute, not kernel-launch overhead.

The output is a table per-operation with absolute milliseconds and a
speedup relative to a stable baseline (numpy when available, else the
first available stack).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import warnings
from typing import Callable, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def _detect_backends():
    info = {
        "numpy": True,
        "cupy": False,
        "lina": False,
        "lina_xp_name": None,
        "lina_cpp": False,
        "lina_cpp_gpu": False,
    }
    try:
        import cupy as cp  # noqa: F401
        info["cupy"] = True
    except BaseException:
        pass
    try:
        import lina  # noqa: F401
        from lina.math_module import xp as lina_xp
        info["lina"] = True
        info["lina_xp_name"] = getattr(lina_xp, "__name__", type(lina_xp).__name__)
    except BaseException:
        pass
    try:
        import lina_cpp  # noqa: F401
        info["lina_cpp"] = True
        info["lina_cpp_gpu"] = bool(lina_cpp.gpu_available())
    except BaseException:
        pass
    return info


def _cupy_sync():
    """Block until all queued GPU work finishes. No-op if cupy is missing."""
    try:
        import cupy as cp
        cp.cuda.runtime.deviceSynchronize()
    except BaseException:
        pass


def _time_call(fn: Callable, sync: Optional[Callable] = None,
               repeats: int = 3) -> float:
    """Return the best wall-clock seconds across ``repeats`` calls.

    A warmup call is performed first. If ``sync`` is given it is called
    between every measurement and *before* stopping the timer; this is
    how we make GPU timings honest.
    """
    fn()
    if sync is not None:
        sync()
    best = math.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        if sync is not None:
            sync()
        dt = time.perf_counter() - t0
        if dt < best:
            best = dt
    return best


# ---------------------------------------------------------------------------
# Per-operation runners
# ---------------------------------------------------------------------------
#
# Each runner returns a dict {stack_name: best_seconds_or_None}. Missing
# stacks (e.g. cupy on a CPU box) come back as None and are skipped at
# print time.

def bench_fft(N: int, repeats: int, info: dict) -> dict:
    rng = np.random.default_rng(N)
    arr_np = (rng.standard_normal((N, N))
              + 1j * rng.standard_normal((N, N))).astype(np.complex128)

    out: dict = {}

    # numpy: emulate lina's ifftshift -> fft2 -> fftshift convention so
    # we're timing the same logical operation across all stacks.
    def _np_fft():
        return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(arr_np)))
    out["numpy"] = _time_call(_np_fft, sync=None, repeats=repeats)

    if info["cupy"]:
        import cupy as cp
        arr_cp = cp.asarray(arr_np)

        def _cp_fft():
            return cp.fft.fftshift(cp.fft.fft2(cp.fft.ifftshift(arr_cp)))
        out["cupy"] = _time_call(_cp_fft, sync=_cupy_sync, repeats=repeats)

    if info["lina"]:
        import lina.props as lprops
        from lina.math_module import xp as lina_xp
        on_gpu = info["lina_xp_name"] == "cupy"
        arr_lina = lina_xp.asarray(arr_np) if hasattr(lina_xp, "asarray") else arr_np

        def _lina_fft():
            return lprops.fft(arr_lina)
        out["lina"] = _time_call(
            _lina_fft,
            sync=_cupy_sync if on_gpu else None,
            repeats=repeats,
        )

    if info["lina_cpp"]:
        import lina_cpp

        def _cpp_cpu():
            return lina_cpp.props.fft(arr_np, device="cpu")
        out["lina_cpp CPU"] = _time_call(_cpp_cpu, sync=None, repeats=repeats)

        if info["lina_cpp_gpu"]:
            def _cpp_gpu():
                return lina_cpp.props.fft(arr_np, device="gpu")
            # lina_cpp's GPU functions return host arrays, so the cudaMemcpy
            # in the binding already synchronises -- no extra sync needed.
            out["lina_cpp GPU"] = _time_call(_cpp_gpu, sync=None, repeats=repeats)

    return out


def _mft_matrices_numpy(npix: int, npsf: int, du: float):
    """Build the two MFT matrices in pure numpy with the lina convention.

    Matches lina.props.make_mft_forward_matrices but without depending
    on whatever array backend lina happens to be using.
    """
    # Pupil coords (odd-grid centring): centred on (npix-1)/2.
    x = np.arange(npix) - (npix - 1) / 2.0
    u = (np.arange(npsf) - (npsf - 1) / 2.0) * du
    arg = -2.0j * np.pi * np.outer(u, x) / npix  # convention='-'
    M_pre = np.exp(arg) / np.sqrt(npix * npsf)
    M_post = M_pre.T
    return M_pre, M_post


def bench_mft_forward(npix: int, npsf: int, repeats: int, info: dict) -> dict:
    rng = np.random.default_rng(npix + npsf)
    wf_np = (rng.standard_normal((npix, npix))
             + 1j * rng.standard_normal((npix, npix))).astype(np.complex128)

    out: dict = {}

    # numpy baseline via explicit MFT matrices. Builds once, times only
    # the two matmuls -- which is what an optimised MFT really is.
    M_pre_np, M_post_np = _mft_matrices_numpy(npix, npsf, 0.5)

    def _np_mft():
        return M_pre_np @ wf_np @ M_post_np
    out["numpy"] = _time_call(_np_mft, sync=None, repeats=repeats)

    if info["cupy"]:
        try:
            import cupy as cp
            wf_cp = cp.asarray(wf_np)
            M_pre_cp = cp.asarray(M_pre_np)
            M_post_cp = cp.asarray(M_post_np)

            def _cp_mft():
                return M_pre_cp @ wf_cp @ M_post_cp
            out["cupy"] = _time_call(_cp_mft, sync=_cupy_sync, repeats=repeats)
        except BaseException:
            pass

    if info["lina"]:
        import lina.props as lprops
        from lina.math_module import xp as lina_xp
        on_gpu = info["lina_xp_name"] == "cupy"
        wf_lina = lina_xp.asarray(wf_np) if hasattr(lina_xp, "asarray") else wf_np

        def _lina_mft():
            return lprops.mft_forward(wf_lina, npix, npsf, 0.5)
        out["lina"] = _time_call(
            _lina_mft,
            sync=_cupy_sync if on_gpu else None,
            repeats=repeats,
        )

    if info["lina_cpp"]:
        import lina_cpp

        def _cpp_cpu():
            return lina_cpp.props.mft_forward(wf_np, npix, npsf, 0.5, device="cpu")
        out["lina_cpp CPU"] = _time_call(_cpp_cpu, sync=None, repeats=repeats)

        if info["lina_cpp_gpu"]:
            def _cpp_gpu():
                return lina_cpp.props.mft_forward(
                    wf_np, npix, npsf, 0.5, device="gpu"
                )
            out["lina_cpp GPU"] = _time_call(_cpp_gpu, sync=None, repeats=repeats)

    return out


def bench_make_vortex(npix: int, repeats: int, info: dict) -> dict:
    """Vortex phase mask: exp(i * charge * theta) on an odd-grid mesh."""
    out: dict = {}
    charge = 6

    # numpy baseline -- identical math to lina_cpp's CPU kernel.
    grid = np.arange(npix) - (npix - 1) / 2.0
    yy_np, xx_np = np.meshgrid(grid, grid, indexing="ij")

    def _np_vortex():
        theta = np.arctan2(yy_np, xx_np)
        return np.exp(1j * charge * theta)
    out["numpy"] = _time_call(_np_vortex, sync=None, repeats=repeats)

    if info["cupy"]:
        try:
            import cupy as cp
            yy_cp = cp.asarray(yy_np)
            xx_cp = cp.asarray(xx_np)

            def _cp_vortex():
                theta = cp.arctan2(yy_cp, xx_cp)
                return cp.exp(1j * charge * theta)
            out["cupy"] = _time_call(_cp_vortex, sync=_cupy_sync, repeats=repeats)
        except BaseException:
            pass

    if info["lina"]:
        import lina.props as lprops
        on_gpu = info["lina_xp_name"] == "cupy"

        def _lina_vortex():
            return lprops.make_vortex_phase_mask(npix, charge=charge)
        out["lina"] = _time_call(
            _lina_vortex,
            sync=_cupy_sync if on_gpu else None,
            repeats=repeats,
        )

    if info["lina_cpp"]:
        import lina_cpp

        def _cpp_cpu():
            return lina_cpp.props.make_vortex_phase_mask(
                npix, charge=charge, device="cpu"
            )
        out["lina_cpp CPU"] = _time_call(_cpp_cpu, sync=None, repeats=repeats)

        if info["lina_cpp_gpu"]:
            def _cpp_gpu():
                return lina_cpp.props.make_vortex_phase_mask(
                    npix, charge=charge, device="gpu"
                )
            out["lina_cpp GPU"] = _time_call(_cpp_gpu, sync=None, repeats=repeats)

    return out


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------

# Stable ordering of stacks in the output table.
STACK_ORDER = ("numpy", "cupy", "lina", "lina_cpp CPU", "lina_cpp GPU")


def _pick_baseline(timings: dict) -> tuple[str, float]:
    """Choose the speedup baseline: prefer numpy, else first available."""
    if "numpy" in timings and timings["numpy"] is not None:
        return "numpy", timings["numpy"]
    for name in STACK_ORDER:
        v = timings.get(name)
        if v is not None:
            return name, v
    return "(none)", float("nan")


def _print_table(title: str, timings: dict) -> None:
    print()
    print(f"{title}")
    if not any(v is not None for v in timings.values()):
        print("  (no available backends)")
        return
    base_name, base_t = _pick_baseline(timings)
    print(f"  {'stack':<20}{'best (ms)':>12}{'vs ' + base_name:>14}")
    print(f"  {'-'*20}{'-'*12:>12}{'-'*14:>14}")
    for name in STACK_ORDER:
        t = timings.get(name)
        if t is None:
            continue
        speedup = base_t / t if t > 0 else float("inf")
        speedup_s = f"{speedup:>6.2f}x"
        marker = "  <- baseline" if name == base_name else ""
        print(f"  {name:<20}{t*1000:>12.3f}{speedup_s:>14}{marker}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare lina_cpp vs lina/numpy/cupy speeds."
    )
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[512, 1024, 2048],
        help="Square sizes N for the fft / vortex benchmarks.",
    )
    parser.add_argument(
        "--mft-sizes", type=int, nargs="+", default=[512, 1024],
        help="Pupil sizes (npix) for the MFT benchmarks. npsf=npix/4.",
    )
    parser.add_argument(
        "--repeats", type=int, default=3,
        help="Number of timed repetitions per stack (best is reported).",
    )
    parser.add_argument(
        "--no-fft", action="store_true", help="Skip the FFT benchmark.",
    )
    parser.add_argument(
        "--no-mft", action="store_true", help="Skip the MFT benchmark.",
    )
    parser.add_argument(
        "--no-vortex", action="store_true", help="Skip the vortex mask benchmark.",
    )
    args = parser.parse_args(argv)

    warnings.filterwarnings("ignore", category=ImportWarning)

    info = _detect_backends()

    print("lina_cpp benchmark")
    print("==================")
    print(f"  numpy             : {info['numpy']}")
    print(f"  cupy              : {info['cupy']}")
    print(f"  lina              : {info['lina']}"
          + (f"  (xp={info['lina_xp_name']})" if info['lina'] else ""))
    print(f"  lina_cpp          : {info['lina_cpp']}")
    print(f"  lina_cpp GPU build: {info['lina_cpp_gpu']}")

    if not args.no_fft:
        for N in args.sizes:
            t = bench_fft(N, args.repeats, info)
            _print_table(f"FFT (N x N complex)   N = {N}", t)

    if not args.no_mft:
        for npix in args.mft_sizes:
            npsf = max(64, npix // 4)
            t = bench_mft_forward(npix, npsf, args.repeats, info)
            _print_table(
                f"MFT forward           npix={npix}, npsf={npsf}", t,
            )

    if not args.no_vortex:
        for N in args.sizes:
            t = bench_make_vortex(N, args.repeats, info)
            _print_table(f"Vortex phase mask     N = {N}", t)

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
