"""Compatibility shim for legacy experimental relative imports.

Older experimental modules imported `from .math_module ...` even though the
shared math backend lives at `lina.math_module`. Re-export those names here so
both import paths work.
"""

from ..math_module import *  # noqa: F401,F403
