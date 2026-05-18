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
   ``.aefc``, ``.math_module``). User code that does
   ``from lina_cpp import props, dm, utils`` works identically to
   ``from lina import props, dm, utils`` -- math hot paths delegate to the
   compiled extension, everything else re-exports from the original lina
   modules until ported.

The C++/pybind side is intentionally importable without importing the
original Python ``lina`` package -- the flat symbols at the top level
work standalone. The submodule wrappers do depend on ``lina`` (they
re-export the un-ported helpers), so loading them is wrapped in a
try/except so that running tests against a bare-extension build still
works even if ``lina`` is not installed.
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


# ---------------------------------------------------------------------------
# 3. Submodule layout: thin Python wrappers that mirror lina's modules.
#    These re-export from lina.* for un-ported helpers and delegate
#    math hot paths to lina_cpp._core. The submodules require `lina` to
#    be importable (because they re-export many helpers from it). If
#    `lina` is missing -- e.g. when running standalone smoke tests
#    against the bare pybind extension -- we still expose the flat
#    top-level symbols above, and just log a warning instead of
#    failing the import.
# ---------------------------------------------------------------------------

_submodules = (
    "math_module",
    "utils",
    "props",
    "dm",
    "coro_utils",
    "llowfsc",
    "rt_utils",
    "wfe",
    "efc",
    "iefc",
    "aefc",
)

from importlib import import_module as _import_module
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
    try:
        _import_module(f"{__name__}.{_sub}")
        setattr(_pkg, _sub, _sys.modules[f"{__name__}.{_sub}"])
    except BaseException as _err:  # noqa: BLE001
        # We catch BaseException (not just ImportError / Exception) because
        # cupy in particular can raise AttributeError, RuntimeError, or
        # SystemExit from inside its CUDA discovery code on broken envs.
        _failed_submodules[_sub] = _err

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
