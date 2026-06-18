"""Compatibility shim for ``lina.math_module`` runtime backend switching.

The original ``lina`` package mutates module-level symbols (``xp``, ``xcipy``,
``np``, ``cupy``, etc.) at runtime via ``update_np`` / ``update_scipy``.
``lina_cpp.set_backend()`` relies on those same symbols, so this wrapper must
re-export the full interface instead of only a subset.
"""

from __future__ import annotations

from lina_cpp._pyref import math_module as _mm

# Keep names stable for existing imports.
cupy_avail = _mm.cupy_avail
ensure_np_array = _mm.ensure_np_array
np_backend = _mm.np_backend
scipy_backend = _mm.scipy_backend
update_np = _mm.update_np
update_scipy = _mm.update_scipy
xcipy = _mm.xcipy
xp = _mm.xp

# ``lina_cpp.set_backend`` expects these names to exist.
np = _mm.np
scipy = _mm.scipy
cupy = getattr(_mm, "cupy", None)
cupyx = getattr(_mm, "cupyx", None)

__all__ = [
    "cupy_avail",
    "ensure_np_array",
    "np_backend",
    "scipy_backend",
    "update_np",
    "update_scipy",
    "xcipy",
    "xp",
    "np",
    "scipy",
    "cupy",
    "cupyx",
]
