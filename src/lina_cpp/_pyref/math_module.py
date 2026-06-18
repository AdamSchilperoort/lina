"""lina.math_module - runtime-switchable numpy/cupy backend.

The mathematical content of ``lina`` is written in a backend-agnostic
style: every submodule does ``from .math_module import xp, xcipy``
and uses ``xp.fft.fft2(...)`` instead of ``numpy.fft.fft2(...)``.

``xp`` and ``xcipy`` are not the actual numpy/cupy modules. They are
thin proxies (``np_backend`` / ``scipy_backend``) that forward every
attribute lookup to whichever underlying module is currently selected.
Because every submodule receives a *reference* to the same proxy, a
single call to :func:`set_backend` switches the entire package at
runtime without any module reloading::

    import lina
    lina.set_backend("cpu")     # use numpy/scipy for everything
    lina.set_backend("gpu")     # use cupy/cupyx.scipy for everything

The choice is also honoured from the ``LINA_BACKEND`` environment
variable at import time, e.g.::

    LINA_BACKEND=cpu jupyter lab
    LINA_BACKEND=gpu python bench.py

If GPU is requested but cupy is broken or absent, the import emits an
``ImportWarning`` and falls back to numpy; :func:`set_backend('gpu')`
raises ``RuntimeError`` with a clear message.
"""

import os
import warnings

import numpy as np
import scipy


# ---------------------------------------------------------------------------
# Optional cupy import.
#
# cupy is optional. We treat *any* failure during cupy import as
# "cupy is unavailable, fall back to numpy" -- not just plain
# ImportError. In practice cupy raises AttributeError, RuntimeError,
# OSError or similar when its CUDA discovery breaks (e.g. CUDA toolkit
# missing on the host even though the cupy wheel is installed). We
# don't want any of those to kill the whole `import lina` chain.
# ---------------------------------------------------------------------------

try:
    import cupy
    import cupyx.scipy
    cupy_avail = True
    _cupy_import_error = None
except Exception as _cupy_err:  # noqa: BLE001
    cupy_avail = False
    cupy = None  # type: ignore[assignment]
    cupyx = None  # type: ignore[assignment]
    _cupy_import_error = f"{type(_cupy_err).__name__}: {_cupy_err}"
    warnings.warn(
        f"cupy import failed ({_cupy_import_error}); "
        "falling back to numpy backend. GPU paths will be unavailable.",
        ImportWarning,
        stacklevel=2,
    )


# ---------------------------------------------------------------------------
# Proxy shims.
#
# These are deliberately tiny. The important property is that
#     from lina.math_module import xp
# anywhere in the codebase yields a reference to the *same* proxy
# instance, so reassigning ``proxy._srcmodule`` here is observed by
# every submodule that has already imported.
# ---------------------------------------------------------------------------

class np_backend:
    """Proxy for numpy / cupy. Swap with :func:`set_backend`."""

    def __init__(self, src):
        self._srcmodule = src

    def __getattr__(self, key):
        if key == "_srcmodule":
            return self._srcmodule
        return getattr(self._srcmodule, key)


class scipy_backend:
    """Proxy for scipy / cupyx.scipy. Swap with :func:`set_backend`."""

    def __init__(self, src):
        self._srcmodule = src

    def __getattr__(self, key):
        if key == "_srcmodule":
            return self._srcmodule
        return getattr(self._srcmodule, key)


# ---------------------------------------------------------------------------
# Resolve initial backend from LINA_BACKEND env var (defaults to cpu).
#
# We default to *cpu* even when cupy is available, because the most
# common surprise is "tests fail with TypeError: implicit conversion
# from cupy". Users who want GPU can opt in explicitly via
# `LINA_BACKEND=gpu` or `lina.set_backend("gpu")`.
# ---------------------------------------------------------------------------

def _resolve_initial_backend():
    env = os.environ.get("LINA_BACKEND", "").strip().lower()
    if env in ("gpu", "cuda", "cupy"):
        if cupy_avail:
            return "gpu"
        warnings.warn(
            "LINA_BACKEND=gpu requested but cupy is not usable "
            f"({_cupy_import_error or 'cupy not installed'}); "
            "falling back to numpy.",
            ImportWarning,
            stacklevel=2,
        )
        return "cpu"
    if env in ("", "cpu", "numpy"):
        return "cpu"
    warnings.warn(
        f"LINA_BACKEND={env!r} is not recognised "
        "(use 'cpu' or 'gpu'); falling back to cpu.",
        ImportWarning,
        stacklevel=2,
    )
    return "cpu"


_active_backend = _resolve_initial_backend()

# Build the proxies, then bind them to the right underlying modules.
xp = np_backend(np)
xcipy = scipy_backend(scipy)
if _active_backend == "gpu":
    xp._srcmodule = cupy
    xcipy._srcmodule = cupyx.scipy


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------

def gpu_available() -> bool:
    """Return True iff cupy imported cleanly at startup.

    Note: cupy can import successfully and still fail later if CUDA
    discovery breaks at first GPU op. ``set_backend('gpu')`` will catch
    that and raise.
    """
    return cupy_avail


def get_backend() -> str:
    """Return 'cpu' or 'gpu' depending on which backend is currently active."""
    return _active_backend


def set_backend(name: str) -> str:
    """Switch lina's numpy/cupy backend at runtime.

    Parameters
    ----------
    name : {'cpu', 'gpu'}
        ``'cpu'`` selects numpy + scipy. ``'gpu'`` selects cupy +
        cupyx.scipy (requires a working cupy install).

    Returns
    -------
    str
        The new active backend.

    Raises
    ------
    ValueError
        If ``name`` is not ``'cpu'`` or ``'gpu'``.
    RuntimeError
        If ``name == 'gpu'`` but cupy is not importable. We refuse to
        silently downgrade so callers can't accidentally end up on the
        CPU when they explicitly asked for GPU.
    """
    global _active_backend
    n = str(name).lower()
    if n in ("cpu", "numpy"):
        xp._srcmodule = np
        xcipy._srcmodule = scipy
        _active_backend = "cpu"
        return "cpu"
    if n in ("gpu", "cuda", "cupy"):
        if not cupy_avail:
            raise RuntimeError(
                "lina.set_backend('gpu') requested but cupy is not usable: "
                f"{_cupy_import_error or 'cupy not installed'}. "
                "Common fixes:\n"
                "  export CUDA_PATH=/usr/local/cuda\n"
                "  export CUDA_HOME=/usr/local/cuda\n"
                "  pip install --force-reinstall --no-deps cupy-cudaXYx  # "
                "match your CUDA major version"
            )
        xp._srcmodule = cupy
        xcipy._srcmodule = cupyx.scipy
        _active_backend = "gpu"
        return "gpu"
    raise ValueError(f"backend must be 'cpu' or 'gpu', got {name!r}")


# ---------------------------------------------------------------------------
# Backwards-compatible legacy entry points.
# ---------------------------------------------------------------------------

def update_np(module):
    """Legacy: replace numpy/cupy proxy target. Prefer :func:`set_backend`."""
    xp._srcmodule = module


def update_scipy(module):
    """Legacy: replace scipy/cupyx.scipy proxy target. Prefer :func:`set_backend`."""
    xcipy._srcmodule = module


# ---------------------------------------------------------------------------
# Array conversion helper. Works regardless of active backend, so test
# code that builds a tensor under cupy and later wants numpy doesn't
# need to know which backend produced the array.
# ---------------------------------------------------------------------------

def ensure_np_array(arr):
    """Return ``arr`` as a numpy array. Handles cupy ``.get()`` round-trip.

    Identical to ``np.asarray(arr)`` for numpy inputs, returns
    ``arr.get()`` for cupy inputs, and is a no-op for None.
    """
    if arr is None:
        return None
    if isinstance(arr, np.ndarray):
        return arr
    if cupy_avail and isinstance(arr, cupy.ndarray):
        return arr.get()
    # Last resort: rely on the array's __array__ protocol.
    return np.asarray(arr)
