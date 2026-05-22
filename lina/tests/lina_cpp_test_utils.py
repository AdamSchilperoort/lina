import os
import sys
import unittest
from pathlib import Path


def _prepend_once(path: Path) -> None:
    p = str(path)
    if p not in sys.path:
        sys.path.insert(0, p)


def bootstrap_lina_cpp_import() -> None:
    """Add repo-local lina_cpp package/build paths to sys.path.

    This makes test execution from a source checkout robust without requiring
    `pip install -e lina_cpp` beforehand.
    """
    repo_root = Path(__file__).resolve().parents[2]

    # Support raw CMake pybind builds where the extension lands directly in
    # build dirs (often named `lina_cpp*.so`).
    build_candidates = [
        repo_root / "cpp" / "build-native",
        repo_root / "cpp" / "build-cuda",
        repo_root / "cpp" / "build-release",
        repo_root / "cpp" / "build",
        repo_root / "cpp",
    ]
    for candidate in reversed(build_candidates):
        if candidate.is_dir():
            _prepend_once(candidate)

    # Fallback to the source-package layout (`lina_cpp/src/lina_cpp`) when
    # no standalone extension module is importable.
    src_dir = repo_root / "lina_cpp" / "src"
    if src_dir.is_dir():
        p = str(src_dir)
        if p not in sys.path:
            sys.path.append(p)


def import_lina_cpp_or_skip():
    """Import lina_cpp after bootstrap; skip test cleanly if unavailable."""
    bootstrap_lina_cpp_import()
    try:
        import lina_cpp
    except Exception as exc:  # pragma: no cover - environment dependent
        raise unittest.SkipTest(
            "lina_cpp unavailable. Build pybind with "
            "`cmake -S cpp -B cpp/build-native -DLINA_BUILD_PYBIND=ON` "
            "and expose build dir on PYTHONPATH."
        ) from exc
    return lina_cpp
