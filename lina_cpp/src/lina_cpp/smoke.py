"""Smoke test for the lina_cpp install.

Run this immediately after ``pip install -e ./lina_cpp/`` to confirm:
* the compiled extension loaded
* CPU FFT/MFT/propagation primitives round-trip
* CPU agrees with pure-Python ``lina`` to ~1e-12
* GPU primitives produce the same numbers as CPU (when GPU build is on)
* timings of CPU vs GPU on a representative 1k x 1k FFT and MFT

Usage::

    python -m lina_cpp.smoke

Exit code is 0 on success, 1 on any failure.
"""

from __future__ import annotations

import math
import sys
import time
import traceback
import warnings

import numpy as np


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

PASS = "  [PASS]"
FAIL = "  [FAIL]"
INFO = "  [info]"

results: list[tuple[str, bool, str]] = []  # (name, ok, message)


def _record(name: str, ok: bool, msg: str = "") -> bool:
    results.append((name, ok, msg))
    tag = PASS if ok else FAIL
    if msg:
        print(f"{tag} {name}: {msg}")
    else:
        print(f"{tag} {name}")
    return ok


def _section(title: str) -> None:
    print()
    print(f"=== {title} ===")


def _err(name: str, exc: BaseException) -> bool:
    return _record(name, False, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------

def _max_abs(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def _time_call(fn, *args, repeats: int = 3, **kwargs) -> tuple[float, object]:
    """Return (best_seconds, last_result) over ``repeats`` calls.

    A warmup call is performed first so JIT / first-use kernel compilation
    on the GPU doesn't pollute timings.
    """
    fn(*args, **kwargs)  # warmup
    best = math.inf
    out = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        dt = time.perf_counter() - t0
        if dt < best:
            best = dt
    return best, out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_import() -> bool:
    _section("Importing lina_cpp")
    try:
        import lina_cpp
    except BaseException as exc:
        return _err("import lina_cpp", exc)
    print(f"{INFO} lina_cpp.__file__ = {lina_cpp.__file__}")
    print(f"{INFO} gpu_available()   = {lina_cpp.gpu_available()}")
    print(f"{INFO} get_device()      = {lina_cpp.get_device()}")
    ncore_syms = len(
        [s for s in dir(lina_cpp._core) if not s.startswith("_")]
    )
    print(f"{INFO} _core exports     = {ncore_syms} symbols")
    return _record("import lina_cpp", True)


def test_fft_roundtrip() -> bool:
    _section("FFT round-trip (CPU)")
    import lina_cpp
    from lina_cpp.props import fft, ifft

    rng = np.random.default_rng(0)
    N = 256
    arr = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    try:
        back = ifft(fft(arr, device="cpu"), device="cpu")
        err = _max_abs(arr, back)
    except BaseException as exc:
        return _err("CPU FFT round-trip", exc)
    return _record(
        f"CPU FFT round-trip N={N}", err < 1e-10,
        f"max|arr - ifft(fft(arr))| = {err:.2e}",
    )


def test_cpu_parity_with_lina() -> bool:
    """Compare every CPU primitive against pure-Python lina.

    `lina.props.*` runs on whatever ``lina.math_module.xp`` resolved to at
    import time (numpy when CuPy is unavailable, CuPy otherwise). We
    convert both inputs and outputs through xp/ensure_np_array so the
    comparison works regardless of which backend lina picked.
    """
    _section("CPU parity with lina (pure-Python)")

    import lina_cpp
    try:
        import lina.props as lprops
        from lina.math_module import xp, ensure_np_array
    except BaseException as exc:
        return _err("import lina.props for parity comparison", exc)

    backend_name = getattr(xp, "__name__", type(xp).__name__)
    print(f"{INFO} lina.math_module.xp = {backend_name}")

    def _to_xp(arr):
        return xp.asarray(arr) if hasattr(xp, "asarray") else arr

    rng = np.random.default_rng(42)
    ok_all = True

    # FFT parity
    arr = rng.standard_normal((128, 128)) + 1j * rng.standard_normal((128, 128))
    try:
        a = ensure_np_array(lina_cpp.props.fft(arr, device="cpu"))
        b = ensure_np_array(lprops.fft(_to_xp(arr)))
        err = _max_abs(a, b)
        ok_all &= _record("fft  (CPU vs lina)", err < 1e-10,
                          f"max diff = {err:.2e}")
    except BaseException as exc:
        ok_all &= _err("fft (CPU vs lina)", exc)

    # MFT forward
    try:
        npix, npsf, du = 256, 64, 0.5
        wf = (rng.standard_normal((npix, npix))
              + 1j * rng.standard_normal((npix, npix))).astype(np.complex128)
        fp_cpp = ensure_np_array(
            lina_cpp.props.mft_forward(wf, npix, npsf, du, device="cpu")
        )
        fp_py = ensure_np_array(lprops.mft_forward(_to_xp(wf), npix, npsf, du))
        err = _max_abs(fp_cpp, fp_py)
        ok_all &= _record(
            "mft_forward (CPU vs lina)", err < 1e-9,
            f"max diff = {err:.2e}  (shape={fp_cpp.shape})",
        )
    except BaseException as exc:
        ok_all &= _err("mft_forward (CPU vs lina)", exc)

    # Vortex phase mask
    try:
        a = ensure_np_array(
            lina_cpp.props.make_vortex_phase_mask(64, charge=6, device="cpu")
        )
        b = ensure_np_array(lprops.make_vortex_phase_mask(64, charge=6))
        err = _max_abs(a, b)
        ok_all &= _record(
            "make_vortex_phase_mask (CPU vs lina)", err < 1e-12,
            f"max diff = {err:.2e}",
        )
    except BaseException as exc:
        ok_all &= _err("make_vortex_phase_mask (CPU vs lina)", exc)

    return ok_all


def test_dm_mask() -> bool:
    """One of the very first calls in the SCoOB notebook."""
    _section("dm.create_mask (used by run_iefc.ipynb cell 1)")
    try:
        import lina_cpp
        mask = lina_cpp.dm.create_mask(Nact=34)
        ok = mask.shape == (34, 34) and mask.sum() > 0
        return _record(
            "lina_cpp.dm.create_mask(Nact=34)", ok,
            f"shape={mask.shape}, active={int(mask.sum())} actuators",
        )
    except BaseException as exc:
        return _err("lina_cpp.dm.create_mask", exc)


def test_gpu_parity_with_cpu() -> bool:
    """When CUDA was compiled in, GPU primitives must match CPU."""
    _section("GPU parity with CPU")
    import lina_cpp
    if not lina_cpp.gpu_available():
        print(f"{INFO} GPU not available in this build; skipping.")
        return True

    rng = np.random.default_rng(7)
    ok_all = True

    # FFT
    try:
        arr = (rng.standard_normal((512, 512))
               + 1j * rng.standard_normal((512, 512))).astype(np.complex128)
        c = lina_cpp.props.fft(arr, device="cpu")
        g = lina_cpp.props.fft(arr, device="gpu")
        err = _max_abs(c, g)
        ok_all &= _record(
            "fft  GPU vs CPU (N=512)", err < 1e-7,
            f"max diff = {err:.2e}",
        )
    except BaseException as exc:
        ok_all &= _err("fft GPU vs CPU", exc)

    # MFT
    try:
        npix, npsf, du = 256, 64, 0.5
        wf = (rng.standard_normal((npix, npix))
              + 1j * rng.standard_normal((npix, npix))).astype(np.complex128)
        c = lina_cpp.props.mft_forward(wf, npix, npsf, du, device="cpu")
        g = lina_cpp.props.mft_forward(wf, npix, npsf, du, device="gpu")
        err = _max_abs(c, g)
        ok_all &= _record(
            f"mft_forward GPU vs CPU (npix={npix},npsf={npsf})",
            err < 1e-8, f"max diff = {err:.2e}",
        )
    except BaseException as exc:
        ok_all &= _err("mft_forward GPU vs CPU", exc)

    # Vortex mask
    try:
        c = lina_cpp.props.make_vortex_phase_mask(128, charge=6, device="cpu")
        g = lina_cpp.props.make_vortex_phase_mask(128, charge=6, device="gpu")
        err = _max_abs(c, g)
        ok_all &= _record(
            "make_vortex_phase_mask GPU vs CPU", err < 1e-12,
            f"max diff = {err:.2e}",
        )
    except BaseException as exc:
        ok_all &= _err("make_vortex_phase_mask GPU vs CPU", exc)

    return ok_all


def test_cpu_vs_gpu_timings() -> bool:
    """Print CPU and GPU timings so the user can see the speedup."""
    _section("Microbenchmarks  (best of 3, after warmup)")
    import lina_cpp
    have_gpu = lina_cpp.gpu_available()
    if not have_gpu:
        print(f"{INFO} GPU not available; CPU timings only.")

    rng = np.random.default_rng(1)

    sizes_fft = (512, 1024, 2048) if have_gpu else (512, 1024)
    for N in sizes_fft:
        arr = (rng.standard_normal((N, N))
               + 1j * rng.standard_normal((N, N))).astype(np.complex128)
        cpu_t, _ = _time_call(lina_cpp.props.fft, arr, device="cpu")
        line = f"  fft  N={N:5d}   CPU {cpu_t*1000:8.2f} ms"
        if have_gpu:
            gpu_t, _ = _time_call(lina_cpp.props.fft, arr, device="gpu")
            speedup = cpu_t / gpu_t if gpu_t > 0 else float("inf")
            line += f"   GPU {gpu_t*1000:8.2f} ms   speedup x{speedup:5.1f}"
        print(line)

    mft_sizes = ((512, 128), (1024, 256))
    if have_gpu:
        mft_sizes = mft_sizes + ((2048, 256),)
    for npix, npsf in mft_sizes:
        wf = (rng.standard_normal((npix, npix))
              + 1j * rng.standard_normal((npix, npix))).astype(np.complex128)
        cpu_t, _ = _time_call(
            lina_cpp.props.mft_forward, wf, npix, npsf, 0.5, device="cpu"
        )
        line = (f"  mft  npix={npix:5d} npsf={npsf:4d}   "
                f"CPU {cpu_t*1000:8.2f} ms")
        if have_gpu:
            gpu_t, _ = _time_call(
                lina_cpp.props.mft_forward, wf, npix, npsf, 0.5, device="gpu"
            )
            speedup = cpu_t / gpu_t if gpu_t > 0 else float("inf")
            line += f"   GPU {gpu_t*1000:8.2f} ms   speedup x{speedup:5.1f}"
        print(line)

    return _record("microbenchmarks completed", True)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    print("lina_cpp smoke test")
    print("===================")

    # Suppress optional-dep warnings so the smoke output stays readable.
    warnings.filterwarnings("ignore", category=ImportWarning)

    ok = True
    try:
        ok &= test_import()
        ok &= test_fft_roundtrip()
        ok &= test_cpu_parity_with_lina()
        ok &= test_dm_mask()
        ok &= test_gpu_parity_with_cpu()
        ok &= test_cpu_vs_gpu_timings()
    except BaseException:
        traceback.print_exc()
        ok = False

    _section("Summary")
    n_pass = sum(1 for _, p, _ in results if p)
    n_fail = sum(1 for _, p, _ in results if not p)
    print(f"  {n_pass} passed, {n_fail} failed")
    if n_fail:
        for name, p, msg in results:
            if not p:
                print(f"   - FAIL {name}: {msg}")
    return 0 if ok and n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
