"""lina_cpp -- C++-backed mirror of the lina wavefront sensing & control package.

Goal: ``import lina_cpp as lina`` is a drop-in replacement for ``import lina``,
so a notebook can swap backends with a single line and compare performance and
numerical equivalence.

Two API surfaces are exposed:

1. **Flat C++ symbols** at the package top level (e.g. ``lina_cpp.fft_cpu``,
   ``lina_cpp.create_annular_mask``). This is what the per-method parity test
   suite uses to exercise the bindings directly without touching the Python
   wrappers.

2. **Submodules** that mirror lina's module layout
   (``lina_cpp.utils``, ``.props``, ``.dm``, ``.coro_utils``,
   ``.llowfsc``, ``.rt_utils``, ``.wfe``, ``.efc``, ``.iefc``,
   ``.aefc``, ``.math_module``). User code does
   ``from lina_cpp import props, dm, utils`` -- math hot paths delegate to
   the compiled extension, and the not-yet-ported helpers re-export from
   the vendored pure-Python reference at ``lina_cpp._pyref``.

Self-contained: ``lina_cpp`` does not import the externally installed
``lina`` package. The pure-Python helpers it still needs (poppy-based WFE
generation, plotting, a handful of utilities) are vendored privately under
``lina_cpp._pyref``. The canonical ``lina`` (e.g. from ``lina_kian``) is
left untouched so it remains available for side-by-side parity comparison
(see ``lina_cpp.bench`` / ``lina_cpp.smoke``).
"""

from __future__ import annotations

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# 0. Initialize NumPy *before* loading the C++ extension.
#
#    Why this matters: the lina_cpp pybind extension may be linked
#    against a different libopenblas than numpy uses (typical when the
#    C++ side was built against the system OpenBLAS and Python is in a
#    conda env). If we load the C++ extension first, the system
#    libopenblas wins the dynamic-linker race. NumPy's startup sanity
#    check (numpy/__init__.py:_sanity_check) then SIGSEGVs the process
#    when it later tries to call its expected BLAS routine.
#
#    Importing numpy here triggers its sanity check while only conda's
#    libopenblas has been dlopened, so both BLAS implementations can
#    safely coexist in the same process afterwards.
# ---------------------------------------------------------------------------

import numpy as _numpy  # noqa: F401  (intentional side-effect: sanity-check first)


# ---------------------------------------------------------------------------
# 1. Import the compiled extension. Two locations are supported:
#    - lina_cpp/_core.so when the package was installed via setuptools
#      (the canonical layout from `pip install -e ./lina_cpp/`).
#    - A standalone `_core.so` / `lina_cpp.so` placed on PYTHONPATH by a
#      raw CMake build. This is the path used by the parity-test
#      runners that build the extension directly into cpp/build/.
# ---------------------------------------------------------------------------

try:
    from . import _core as _ext  # type: ignore
except ImportError:
    try:
        import importlib

        _ext = importlib.import_module("_core")
    except ImportError as e:
        raise ImportError(
            "lina_cpp could not import its compiled extension `_core`. "
            "If you are running from source, build cpp/ with "
            "`-DLINA_BUILD_PYBIND=ON -DLINA_PYBIND_MODULE_NAME=_core` "
            "and place the resulting .so on PYTHONPATH; otherwise "
            "reinstall lina_cpp via `pip install -e ./lina_cpp/`."
        ) from e


# ---------------------------------------------------------------------------
# 2. Flat re-export at the top level: every public symbol from the C
#    extension is also accessible as `lina_cpp.<name>`. This preserves
#    the existing test surface where parity tests do `lina_cpp.fft_cpu(...)`
#    directly without going through the submodule layout.
# ---------------------------------------------------------------------------

_flat_names = []
for _name in dir(_ext):
    if _name.startswith("_"):
        continue
    globals()[_name] = getattr(_ext, _name)
    _flat_names.append(_name)


# ---------------------------------------------------------------------------
# Cap CPU math (OpenBLAS/LAPACKE) thread count.
#
# OpenBLAS defaults to *all* logical CPUs. On a large shared box (e.g. 384
# logical cores) this catastrophically oversubscribes for the matrix sizes
# lina uses (<= a few thousand square): a single 1024x1024 SVD/GEMM with 384
# threads thrashes and can take minutes instead of well under a second,
# especially when other users are loading the machine. Cap to a sane default
# unless the user has explicitly set OMP_NUM_THREADS / OPENBLAS_NUM_THREADS.
# ---------------------------------------------------------------------------
import os as _os  # noqa: E402

if not (
    _os.environ.get("OPENBLAS_NUM_THREADS")
    or _os.environ.get("OMP_NUM_THREADS")
    or _os.environ.get("LINA_CPP_NUM_THREADS")
):
    try:
        _ncpu = _os.cpu_count() or 8
        _ext.set_num_threads(min(16, _ncpu))
    except Exception:
        pass
elif _os.environ.get("LINA_CPP_NUM_THREADS"):
    try:
        _ext.set_num_threads(int(_os.environ["LINA_CPP_NUM_THREADS"]))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 2b. CPU/GPU device dispatch helpers.
#
#    Exposed at the package top level so users can do:
#        lina_cpp.set_device("gpu")    # all subsequent calls go to GPU
#        lina_cpp.gpu_available()      # True if CUDA was compiled in
#        lina_cpp.get_device()         # current default
#
#    Each math wrapper in lina_cpp.props, lina_cpp.utils, etc. also
#    accepts a per-call ``device=`` kwarg that overrides the default
#    for one call.
# ---------------------------------------------------------------------------

from ._dispatch import (  # noqa: E402
    get_device,
    gpu_available,
    set_device,
)

_dispatch_names = ("get_device", "gpu_available", "set_device")
for _name in _dispatch_names:
    _flat_names.append(_name)


# Unified backend toggle: switch BOTH the lina Python backend (numpy/cupy)
# AND the lina_cpp C++ device dispatcher (cpu/gpu) with one call. Handy
# in notebooks where users want one knob for "run everything on the
# GPU" or "run everything on the CPU".
def set_backend(name: str) -> dict:
    """Switch lina_cpp's compute backend (CPU/GPU) with one call.

    This switches both halves of lina_cpp at once:

    * the native C++/CUDA device dispatcher (``set_device``), and
    * the vendored pure-Python reference (``lina_cpp._pyref``) that backs
      the not-yet-ported helpers (poppy-based WFE generation, plotting,
      a handful of utilities). Its numpy/cupy proxy is switched too so
      those helpers run on the same device.

    The externally-installed ``lina`` package (the canonical pure-Python
    baseline from ``lina_kian``) is intentionally left untouched so it
    remains available for side-by-side parity comparisons.

    Parameters
    ----------
    name : {'cpu', 'gpu'}
        Target backend for the entire stack.

    Returns
    -------
    dict
        ``{'lina_cpp': <new device>, 'pyref': <new pyref backend or None>}``.

    Raises
    ------
    RuntimeError
        If ``'gpu'`` is requested but lina_cpp was built without CUDA.
    """
    n = str(name).lower()
    if n not in ("cpu", "gpu"):
        raise ValueError(f"backend must be 'cpu' or 'gpu', got {name!r}")

    # Native C++/CUDA device first -- this is the source of truth.
    lina_cpp_state = set_device(n)

    # Switch the vendored pure-Python reference (numpy/cupy + poppy) used by
    # the wrapper layer for un-ported helpers, so they run on the same device.
    pyref_state = None
    try:
        from . import _pyref
        if hasattr(_pyref, "set_backend"):
            pyref_state = _pyref.set_backend(n)
    except Exception as e:  # noqa: BLE001
        _warnings.warn(
            f"lina_cpp device set to {n!r}, but switching the vendored "
            f"pure-Python reference backend failed: {e}",
            RuntimeWarning,
            stacklevel=2,
        )

    _set_poppy_backend(n == "gpu")
    return {"lina_cpp": lina_cpp_state, "pyref": pyref_state}


def _set_poppy_backend(use_cupy: bool) -> None:
    """Point poppy at the same numpy/cupy backend as lina_cpp."""
    try:
        import poppy
        if hasattr(poppy, "conf") and hasattr(poppy.conf, "use_cupy"):
            poppy.conf.use_cupy = bool(use_cupy)
        if hasattr(poppy, "accel_math") and hasattr(poppy.accel_math, "_USE_CUPY"):
            poppy.accel_math._USE_CUPY = bool(use_cupy)
    except Exception:
        pass


_flat_names.append("set_backend")


# ---------------------------------------------------------------------------
# 3. Submodule layout: thin Python wrappers that mirror lina's modules.
#    Math hot paths delegate to the compiled extension (lina_cpp._core);
#    un-ported helpers (poppy WFE, plotting, a few utils) are re-exported
#    from the *vendored* pure-Python reference at lina_cpp._pyref, so the
#    package is fully self-contained and does not import the externally
#    installed `lina`. If a submodule wrapper fails to load we still
#    expose the flat top-level C++ symbols and log a warning.
# ---------------------------------------------------------------------------

_submodules = (
    "math_module",
    "utils",
    "props",
    "dm",
    "control_models",
    "coro_utils",
    "llowfsc",
    "rt_utils",
    "wfe",
    "efc",
    "iefc",
    "aefc",
    "experimental",
)

from importlib import import_module as _import_module
import importlib.util as _importlib_util
import sys as _sys
import warnings as _warnings

_pkg = _sys.modules[__name__]
_failed_submodules: dict[str, BaseException] = {}

# Try each submodule individually so a single bad one (e.g. a busted
# cupy install when loading the rt_utils / math_module wrappers)
# doesn't take down the rest. The flat top-level C++ symbols are
# already in place by the point we get here, so even total submodule
# failure leaves the package usable for the parity-test surface.
for _sub in _submodules:
    _mod_name = f"{__name__}.{_sub}"

    # Recover from stale partially-imported submodules (e.g. prior failed import
    # that left a module object without a usable spec, which then breaks
    # importlib.reload(...)).
    _stale = _sys.modules.get(_mod_name)
    if _stale is not None and getattr(_stale, "__spec__", None) is None:
        _sys.modules.pop(_mod_name, None)

    try:
        _mod = _import_module(_mod_name)
        if getattr(_mod, "__spec__", None) is None:
            try:
                _mod.__spec__ = _importlib_util.find_spec(_mod_name)
            except Exception:
                pass
        setattr(_pkg, _sub, _mod)
    except BaseException as _err:  # noqa: BLE001
        # We catch BaseException (not just ImportError / Exception) because
        # cupy in particular can raise AttributeError, RuntimeError, or
        # SystemExit from inside its CUDA discovery code on broken envs.
        _failed_submodules[_sub] = _err
        _sys.modules.pop(_mod_name, None)
        if hasattr(_pkg, _sub):
            delattr(_pkg, _sub)

if _failed_submodules:
    _names = ", ".join(sorted(_failed_submodules))
    _details = "; ".join(
        f"{name}: {type(err).__name__}: {err}"
        for name, err in _failed_submodules.items()
    )
    _warnings.warn(
        f"lina_cpp: the following submodule wrappers failed to load and "
        f"are unavailable as attributes of the package: {_names}. "
        f"Flat top-level C++ symbols (e.g. lina_cpp.fft_cpu) still work. "
        f"Details: {_details}",
        ImportWarning,
        stacklevel=2,
    )

__all__ = list(_flat_names) + list(_submodules) + ["__version__"]
