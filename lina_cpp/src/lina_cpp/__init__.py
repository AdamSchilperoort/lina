"""
Native lina C++/pybind implementation.

This package is intentionally importable without importing the original
Python `lina` package. That allows:

    import lina_cpp as lina

for testing native functions directly.
"""

from . import _core

# Re-export native symbols at package level.
for _name in dir(_core):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_core, _name)

__all__ = [name for name in dir(_core) if not name.startswith("_")]
