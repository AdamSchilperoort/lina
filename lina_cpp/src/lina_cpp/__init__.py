"""lina_cpp - C++-backed mirror of the lina wavefront sensing & control package.

Goal: ``import lina_cpp as lina`` should be a drop-in replacement for ``import
lina``, so a notebook can swap backends with a single line and compare
performance and numerical equivalence.

Coverage at a glance
--------------------
* Math hot paths (utils.fft/ifft/mft/ang_spec/get_fresnel_TF, dm.create_*,
  efc/iefc/aefc, control_models, pwp, linalg.svd) -> native C++ via the
  pybind extension at :mod:`lina_cpp._core`.
* FITS I/O -> native C++ via cfitsio.
* Hardware control (coro_utils.move_psf, set_zwo_*, etc.) -> re-exported
  from :mod:`lina.coro_utils` until the native pcf::IndiClient port lands.
* High-level loop orchestration (llowfsc.run, rt_utils, wfe, telem
  helpers, plotting) -> re-exported from :mod:`lina` until ported.

This split lets users get the C++ speedups today while keeping the
exact-same Python entry points. As more modules get ported, the
re-exports here are swapped for native wrappers without changing any
call site.
"""

from __future__ import annotations

# Import the compiled extension. We try a few names so this works whether:
#  - lina_cpp is pip-installed (extension at lina_cpp/_core.so), or
#  - the user is running against a standalone CMake build whose .so is
#    named lina_cpp.so on PYTHONPATH (common during development).
#
# The flat re-export below is what makes `lina_cpp.fft(...)` work after
# `import lina_cpp` -- the existing parity test suite relies on this.
try:
    from . import _core as _ext  # type: ignore
except ImportError:
    try:
        # Fall back to a top-level import; happens during `cmake --build`
        # workflows where the .so isn't yet inside the package dir.
        import importlib

        _ext = importlib.import_module("_core")
    except ImportError as e:
        raise ImportError(
            "lina_cpp could not import its compiled extension `_core`. "
            "If you are running from source, build cpp/ with "
            "`-DLINA_BUILD_PYBIND=ON -DLINA_PYBIND_MODULE_NAME=_core` "
            "and place the .so on PYTHONPATH; otherwise reinstall "
            "lina_cpp."
        ) from e

# Version mirrors lina.
__version__ = "0.1.0"


def _flat_reexport() -> None:
    """Mirror every public symbol from the C extension at the top level.

    This preserves the existing test surface where parity tests do
    ``lina_cpp.fft(arr)`` directly. New code is encouraged to use the
    submodule layout (``lina_cpp.props.fft``) which mirrors lina.
    """
    g = globals()
    for name in dir(_ext):
        if name.startswith("_"):
            continue
        g.setdefault(name, getattr(_ext, name))


_flat_reexport()


# Submodules. These are thin Python files that mirror lina's module
# layout so user code that does `from lina_cpp import props, dm, utils`
# behaves identically to `from lina import props, dm, utils`.
from . import math_module  # noqa: E402
from . import utils  # noqa: E402
from . import props  # noqa: E402
from . import dm  # noqa: E402
from . import coro_utils  # noqa: E402  (re-exports from lina)
from . import llowfsc  # noqa: E402  (re-exports from lina)
from . import rt_utils  # noqa: E402  (re-exports from lina)
from . import wfe  # noqa: E402  (re-exports from lina)
from . import efc  # noqa: E402
from . import iefc  # noqa: E402
from . import aefc  # noqa: E402

__all__ = [
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
    "__version__",
]
