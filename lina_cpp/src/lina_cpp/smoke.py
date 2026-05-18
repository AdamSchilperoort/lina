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
    # Toolchain info -- useful when a miscompilation creeps in.
    try:
        import numpy
        print(f"{INFO} numpy version     = {numpy.__version__}")
    except Exception:  # noqa: BLE001
        pass
    try:
        import pybind11  # type: ignore
        print(f"{INFO} pybind11 version  = {pybind11.__version__}")
    except Exception:  # noqa: BLE001
        pass
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


def test_llowfsc_reconstruct_sanity() -> bool:
    """Catch the 'collapsed gemv loop' miscompilation.

    The C++ ``llowfsc_reconstruct`` computes ``coeff[k] = C[mode_lo+k]
    @ del_im[mask]``. For random inputs each coefficient must differ.
    A binary that returns ``[c0, c0, c0, ...]`` (every entry equal to
    coeff[0]) means the per-row pointer is being hoisted out of the
    inner loop -- observed on gcc 12 with CUDA-enabled -O3 builds. We
    rewrote the loop to use the safe ``Array2D::operator()(r, c)``
    accessor specifically to dodge that miscompilation; this test
    catches any regression that brings it back.
    """
    _section("llowfsc_reconstruct sanity (catches miscompiled gemv loop)")
    try:
        import lina_cpp
    except BaseException as exc:
        return _err("import lina_cpp", exc)

    rng = np.random.default_rng(8)
    H = W = 32
    Nmodes = 10
    camlo = 5.0 + 0.1 * rng.standard_normal((H, W))
    mask = (rng.standard_normal((H, W)) > 0.0).astype(np.uint8)
    ref = 0.001 * rng.standard_normal((H, W))
    Nmask = int(mask.sum())
    C = rng.standard_normal((Nmodes, Nmask))

    try:
        coeff = np.asarray(lina_cpp.llowfsc_reconstruct(
            camlo, ref, mask, C,
            mode_lo=0, mode_hi=Nmodes,
            dark_im=0.0, flux_norm=True, return_del_im=False,
        ))
    except BaseException as exc:
        return _err("llowfsc_reconstruct call", exc)

    if coeff.shape != (Nmodes,):
        return _record(
            "llowfsc_reconstruct shape", False,
            f"expected ({Nmodes},), got {coeff.shape}",
        )

    # Cross-check against the pure-Python lina, when available. If the
    # backends disagree we dump enough state that a remote user can
    # send the smoke output back and we can pinpoint the cause.
    try:
        import lina  # noqa: F401
        from lina.llowfsc import reconstruct as py_reconstruct
        lina.set_backend("cpu")
        py_coeff = np.asarray(py_reconstruct(
            camlo, ref, mask.astype(bool), C,
            dark_im=0.0, modes=(0, Nmodes),
            flux_norm=True, return_del_im=False,
        ))
        py_ref = py_coeff
    except BaseException:
        py_ref = None

    spread = float(coeff.max() - coeff.min())
    n_unique = int(np.unique(np.round(coeff, 12)).size)
    is_constant = n_unique <= 1 or spread < 1e-12

    if is_constant:
        msg_lines = [
            f"C++ output is constant ({coeff[0]:+.9f} x {Nmodes}).",
            f"  Nmask = {Nmask},  C.shape = {C.shape}",
            f"  cpp[:5] = {coeff[:5]}",
        ]
        if py_ref is not None:
            msg_lines.append(f"  py [:5] = {py_ref[:5]}  (correct reference)")
            msg_lines.append(
                f"  cpp[0] matches py[0]? "
                f"{np.isclose(coeff[0], py_ref[0], rtol=1e-10)}"
            )
        msg_lines.append(
            "This is the 'collapsed gemv loop' bug. The defensive fix "
            "in cpp/src/llowfsc.cpp uses Array2D::operator()(r,c) "
            "specifically to prevent this; if you see this message "
            "after a clean rebuild on a recent checkout, send the "
            "full smoke output, your compiler version (`gcc --version`),"
            " your numpy version, and your pybind11 version "
            "(`python -c 'import pybind11; print(pybind11.__version__)'`) "
            "to debug."
        )
        return _record("llowfsc_reconstruct produces non-constant vector",
                       False, "\n        ".join(msg_lines))

    # Spread looks OK -- also verify parity with Python when we have it.
    parity_msg = f"spread = {spread:.3e}, {n_unique}/{Nmodes} distinct"
    if py_ref is not None:
        max_err = float(np.max(np.abs(coeff - py_ref)))
        parity_msg += f"; max|cpp - py| = {max_err:.2e}"
        if max_err > 1e-10:
            return _record(
                "llowfsc_reconstruct matches Python",
                False, parity_msg,
            )
    return _record(
        "llowfsc_reconstruct produces non-constant vector",
        True, parity_msg,
    )


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
        ok &= test_llowfsc_reconstruct_sanity()
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
