"""setup.py for the lina_cpp Python package.

This is a thin Python package that wraps the C++ implementation in
``../cpp``. The compiled extension is installed at
``lina_cpp/_core.cpython-...-...so`` and re-exported by the package's
``__init__.py``.

Build steps:
1. Resolve the ``../cpp`` source dir relative to this file.
2. Configure CMake with ``LINA_BUILD_PYBIND=ON`` and
   ``LINA_PYBIND_MODULE_NAME=_core`` so the .so is named ``_core``.
3. Build, then copy the .so into the staged package directory.

Pass ``-DLINA_USE_CUDA=ON`` via the env var ``LINA_CMAKE_ARGS`` for a
GPU build, e.g.::

    LINA_CMAKE_ARGS="-DLINA_USE_CUDA=ON" pip install -e .

"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


HERE = pathlib.Path(__file__).resolve().parent
# Location of the C++ source tree. We assume the standard repo layout
# (../cpp from this file), but allow override via env var so a user can
# point at a separate cpp checkout.
CPP_SOURCE_DIR = pathlib.Path(
    os.environ.get("LINA_CPP_SOURCE_DIR", HERE.parent / "cpp")
).resolve()


class CMakeExtension(Extension):
    """Marker Extension for our CMake-built _core.so."""

    def __init__(self, name: str, sourcedir: pathlib.Path):
        # No source files at the setuptools level -- CMake builds the .so.
        super().__init__(name, sources=[])
        self.sourcedir = sourcedir


class CMakeBuild(build_ext):
    """Drives a CMake build of the cpp/ tree and stages the resulting .so."""

    def run(self) -> None:
        # Sanity-check that cmake exists *before* setuptools commits to
        # the build_ext flow; pip will print a much clearer error.
        try:
            subprocess.check_output(["cmake", "--version"])
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(
                "CMake is required to build lina_cpp but was not found on PATH. "
                "Install cmake (e.g. `conda install cmake` or `apt install cmake`) "
                "and retry."
            ) from exc
        super().run()

    def build_extension(self, ext: CMakeExtension) -> None:
        if not isinstance(ext, CMakeExtension):
            return super().build_extension(ext)

        # Where setuptools wants the final .so to land in the staged tree.
        ext_fullpath = pathlib.Path(self.get_ext_fullpath(ext.name)).resolve()
        ext_dir = ext_fullpath.parent
        ext_dir.mkdir(parents=True, exist_ok=True)

        build_temp = pathlib.Path(self.build_temp).resolve()
        build_temp.mkdir(parents=True, exist_ok=True)

        cfg = "Debug" if self.debug else "Release"

        cmake_args = [
            f"-DCMAKE_BUILD_TYPE={cfg}",
            "-DLINA_BUILD_PYBIND=ON",
            # The pybind module's PYBIND11_MODULE() macro expands to this
            # token, AND the .so file gets named accordingly.
            "-DLINA_PYBIND_MODULE_NAME=_core",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            # Drop the .so directly into the staged package directory so
            # we don't have to copy it after the build.
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={ext_dir}",
            f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={ext_dir}",
        ]

        # Allow the caller to inject extra CMake flags, e.g.
        # LINA_CMAKE_ARGS="-DLINA_USE_CUDA=ON".
        if extra := os.environ.get("LINA_CMAKE_ARGS"):
            cmake_args.extend(extra.split())

        build_args = ["--config", cfg, "--target", "lina_py"]
        if jobs := os.environ.get("LINA_BUILD_JOBS"):
            build_args.extend(["--", f"-j{jobs}"])
        else:
            build_args.extend(["--", f"-j{os.cpu_count() or 4}"])

        if not CPP_SOURCE_DIR.exists():
            raise RuntimeError(
                f"C++ source dir not found at {CPP_SOURCE_DIR}. "
                f"Set LINA_CPP_SOURCE_DIR to override."
            )

        print(f"[lina_cpp] Configuring CMake: {CPP_SOURCE_DIR} -> {build_temp}")
        subprocess.check_call(
            ["cmake", str(CPP_SOURCE_DIR), *cmake_args],
            cwd=build_temp,
        )
        print(f"[lina_cpp] Building extension into {ext_dir}")
        subprocess.check_call(
            ["cmake", "--build", str(build_temp), *build_args],
            cwd=build_temp,
        )

        # CMake honours CMAKE_LIBRARY_OUTPUT_DIRECTORY for the .so, but
        # some generators put it under build/ regardless. Hunt for it.
        ext_suffix = (
            os.environ.get("EXT_SUFFIX")
            or sysconfig_ext_suffix()
        )
        candidates = [
            ext_dir / f"_core{ext_suffix}",
            build_temp / f"_core{ext_suffix}",
        ]
        for cand in candidates:
            if cand.exists():
                target = ext_fullpath  # this is what setuptools expects
                if cand.resolve() != target.resolve():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(cand, target)
                # Also drop a copy alongside __init__.py (when ext_dir
                # already equals the package dir, this is a no-op).
                package_dir = ext_dir / "lina_cpp"
                if package_dir.exists():
                    shutil.copy2(cand, package_dir / cand.name)
                return
        raise RuntimeError(
            f"Could not locate built extension; looked in: "
            + ", ".join(str(c) for c in candidates)
        )


def sysconfig_ext_suffix() -> str:
    """Get the platform extension suffix (e.g. .cpython-310-x86_64-linux-gnu.so)."""
    import sysconfig
    return sysconfig.get_config_var("EXT_SUFFIX") or ".so"


setup(
    ext_modules=[CMakeExtension("lina_cpp._core", sourcedir=CPP_SOURCE_DIR)],
    cmdclass={"build_ext": CMakeBuild},
)
