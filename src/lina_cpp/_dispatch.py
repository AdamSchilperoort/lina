"""Device selection for lina_cpp math primitives.

Each math hot path in lina_cpp has two C++ implementations: a CPU variant
backed by FFTW / OpenBLAS / LAPACKE, and (when the extension was compiled
with ``-DLINA_USE_CUDA=ON``) a GPU variant backed by cuFFT / cuBLAS /
cuSOLVER. This module decides which one to call.

Three layers of selection are available:

1. **Environment variable**, set before the package is imported::

       LINA_CPP_DEVICE=gpu python my_script.py

2. **Module-level default**, swappable at runtime::

       import lina_cpp
       lina_cpp.set_device("gpu")     # all subsequent calls go to GPU
       lina_cpp.set_device("cpu")     # back to CPU

3. **Per-call override**, for one-off control::

       lina_cpp.props.fft(arr, device="gpu")
       lina_cpp.props.mft_forward(wf, npix, npsf, du, device="cpu")

If GPU is requested but this build of the extension does not include
the GPU kernels (``LINA_USE_CUDA=OFF`` at compile time), the call falls
back to the CPU implementation and emits a one-shot warning. If a
function has no GPU implementation at all (e.g. mask creation
primitives), the device argument is silently ignored.

The intended pattern for new wrapper functions is::

    from ._dispatch import resolve_device

    def fft(arr, device=None):
        impl = resolve_device("fft", device)   # 'cpu' or 'gpu'
        arr = _prep(arr)
        return _core.fft_gpu(arr) if impl == "gpu" else _core.fft_cpu(arr)
"""

from __future__ import annotations

import os
import warnings
from typing import Literal, Optional

from . import _core

Device = Literal["cpu", "gpu"]
DeviceArg = Optional[Device]


# ---------------------------------------------------------------------------
# GPU availability
# ---------------------------------------------------------------------------

def gpu_available() -> bool:
    """Return True if this build of lina_cpp was compiled with CUDA.

    Detection is structural: we look for a marker GPU function in the
    compiled extension. Newer additions should keep ``fft_gpu`` as the
    canonical "is CUDA in" marker.
    """
    return hasattr(_core, "fft_gpu")


# ---------------------------------------------------------------------------
# Default device (module-level state, swappable at runtime)
# ---------------------------------------------------------------------------

def _initial_device() -> Device:
    env = os.environ.get("LINA_CPP_DEVICE", "").strip().lower()
    if env in ("gpu", "cuda"):
        if gpu_available():
            return "gpu"
        warnings.warn(
            "LINA_CPP_DEVICE=gpu requested but lina_cpp was built without "
            "CUDA support (LINA_USE_CUDA=OFF); defaulting to CPU.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "cpu"
    if env in ("", "cpu"):
        return "cpu"
    warnings.warn(
        f"LINA_CPP_DEVICE={env!r} is not recognised (use 'cpu' or 'gpu'); "
        "defaulting to CPU.",
        RuntimeWarning,
        stacklevel=2,
    )
    return "cpu"


_default_device: Device = _initial_device()


def get_device() -> Device:
    """Current default device for math hot paths ('cpu' or 'gpu')."""
    return _default_device


def set_device(device: Device) -> Device:
    """Set the default device for math hot paths.

    Parameters
    ----------
    device : {'cpu', 'gpu'}
        The new default. If 'gpu' is requested but the build lacks CUDA
        support, a RuntimeError is raised so users can't silently end
        up on CPU when they expected GPU.

    Returns
    -------
    str
        The new active device (the same string passed in, on success).
    """
    global _default_device
    device_l = str(device).lower()
    if device_l not in ("cpu", "gpu"):
        raise ValueError(f"device must be 'cpu' or 'gpu', got {device!r}")
    if device_l == "gpu" and not gpu_available():
        raise RuntimeError(
            "lina_cpp was built without CUDA support (LINA_USE_CUDA=OFF); "
            "cannot set device to 'gpu'. Rebuild with "
            "'cmake -S cpp -B cpp/build-cuda -DLINA_USE_CUDA=ON ...' and "
            "reinstall lina_cpp to enable the GPU backend."
        )
    _default_device = device_l  # type: ignore[assignment]
    return _default_device


# ---------------------------------------------------------------------------
# Per-call resolution
# ---------------------------------------------------------------------------

# Names of functions for which we have logged a "no GPU variant exists"
# warning. Used so each missing-kernel name only warns once per process.
_warned_missing: set[str] = set()


def resolve_device(function_name: str, device: DeviceArg) -> Device:
    """Resolve a per-call ``device`` kwarg to the actual backend to use.

    Parameters
    ----------
    function_name : str
        The wrapper's own name (e.g. ``'fft'``). Used only for clearer
        error/warning messages.
    device : {'cpu', 'gpu', None}
        The user-supplied per-call device. ``None`` means "use the
        module-level default" (see :func:`get_device`).

    Returns
    -------
    {'cpu', 'gpu'}
        The backend the caller should dispatch to. Falls back to CPU
        with a warning if GPU was requested but isn't available in this
        build, or if the requested function has no GPU variant.
    """
    if device is None:
        chosen = _default_device
    else:
        chosen = str(device).lower()  # type: ignore[assignment]
        if chosen not in ("cpu", "gpu"):
            raise ValueError(
                f"device must be 'cpu' or 'gpu', got {device!r}"
            )

    if chosen == "gpu" and not gpu_available():
        warnings.warn(
            f"lina_cpp.{function_name}: GPU requested but lina_cpp was "
            "built without CUDA support; falling back to CPU.",
            RuntimeWarning,
            stacklevel=3,
        )
        return "cpu"
    return chosen  # type: ignore[return-value]


def warn_no_gpu_variant(function_name: str) -> None:
    """Emit a one-shot warning when GPU was requested for a CPU-only kernel.

    Used by wrappers around functions that don't (yet) have a GPU
    implementation in the extension -- DM mode creation, mask
    construction, FITS I/O, etc. Without this notice, users would
    silently get CPU execution and could mistakenly attribute timings
    to a GPU code path.
    """
    if function_name in _warned_missing:
        return
    _warned_missing.add(function_name)
    warnings.warn(
        f"lina_cpp.{function_name} has no GPU implementation; running "
        "on CPU. This warning is shown once per process.",
        RuntimeWarning,
        stacklevel=3,
    )
