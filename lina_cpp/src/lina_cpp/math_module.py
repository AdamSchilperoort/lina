"""lina_cpp.math_module - re-export of lina.math_module.

The runtime numpy/cupy switching shim is identical between the two
backends; the C++ code path doesn't read xp/xcipy itself, so we just
re-export so user code that does `from lina_cpp.math_module import xp`
behaves identically to `from lina.math_module import xp`.
"""

from __future__ import annotations

# Re-export everything from lina.math_module so any reference users
# already have continues to work.
from lina.math_module import (  # noqa: F401
    cupy_avail,
    ensure_np_array,
    np_backend,
    scipy_backend,
    update_np,
    update_scipy,
    xcipy,
    xp,
)
