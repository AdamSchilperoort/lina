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

GPU support
-----------

By default the CMake configure step auto-detects whether a working
CUDA toolkit is available on the system and turns on the GPU backend
(cuFFT, cuBLAS, cuSOLVER, and custom kernels) when it is. If you want
to force or forbid the GPU build, use one of these knobs:

* ``LINA_USE_CUDA=1`` / ``LINA_USE_CUDA=0`` env var (simplest)::

    LINA_USE_CUDA=1 pip install -e ./lina_cpp/
    LINA_USE_CUDA=0 pip install -e ./lina_cpp/

* Or the more general ``LINA_CMAKE_ARGS`` escape hatch::

    LINA_CMAKE_ARGS="-DLINA_USE_CUDA=ON" pip install -e ./lina_cpp/

After install, verify with::

    python -c "import lina_cpp; print('GPU:', lina_cpp.gpu_available())"

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

        common_args = [
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

        # Simple env-var toggle for the most common case: GPU build yes/no.
        # Accepts 1/0/on/off/true/false/yes/no/auto (any other value falls
        # through to the auto-detect path in CMakeLists.txt).
        cuda_env = os.environ.get("LINA_USE_CUDA", "").strip().lower()
        forced_cuda_on = cuda_env in ("1", "on", "true", "yes")
        forced_cuda_off = cuda_env in ("0", "off", "false", "no")

        # Pre-flight nvcc sanity check when the user forces a GPU build.
        # Without this the build fails opaquely on the first cmake error.
        if forced_cuda_on:
            nvcc = shutil.which("nvcc")
            if not nvcc:
                raise RuntimeError(
                    "LINA_USE_CUDA=1 set but `nvcc` is not on PATH. This is "
                    "the most common failure mode in conda envs: cupy ships "
                    "its own CUDA runtime libs but does not pull in nvcc or "
                    "the dev headers. Install them with:\n\n"
                    "    conda install -c nvidia cuda-nvcc cuda-cudart-dev "
                    "cuda-libraries-dev\n\n"
                    "or unset LINA_USE_CUDA to build CPU-only."
                )
            print(f"[lina_cpp] LINA_USE_CUDA=1 set; forcing GPU build "
                  f"(nvcc={nvcc}).")
        elif forced_cuda_off:
            print("[lina_cpp] LINA_USE_CUDA=0 set; forcing CPU-only build.")
        else:
            if cuda_env and cuda_env != "auto":
                print(f"[lina_cpp] LINA_USE_CUDA={cuda_env!r} not "
                      "recognised; falling through to CMake auto-detect.")
            print("[lina_cpp] LINA_USE_CUDA not set; CMake will auto-detect "
                  "CUDA. Set LINA_USE_CUDA=1 to force GPU, =0 for CPU.")

        # Extra user-supplied CMake flags (e.g. CUDA_ARCHITECTURES).
        extra_args = (os.environ.get("LINA_CMAKE_ARGS") or "").split()

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

        # Build attempts. If the user forced a setting, we make exactly one
        # attempt. If they left it on auto, we first try whatever CMake
        # decides (typically GPU on a CUDA box), and on failure we wipe the
        # build dir and retry CPU-only, so partial CUDA installs degrade
        # gracefully instead of breaking `pip install`.
        if forced_cuda_on:
            attempts = [("forced GPU", ["-DLINA_USE_CUDA=ON", *extra_args])]
        elif forced_cuda_off:
            attempts = [("forced CPU", ["-DLINA_USE_CUDA=OFF", *extra_args])]
        else:
            attempts = [
                ("auto-detect", list(extra_args)),
                ("CPU fallback", ["-DLINA_USE_CUDA=OFF", *extra_args]),
            ]

        for i, (label, attempt_args) in enumerate(attempts):
            is_last = (i == len(attempts) - 1)
            attempt_cmake_args = common_args + attempt_args

            # Fresh build directory each attempt so a half-configured CUDA
            # cache from a failed run doesn't poison the CPU fallback.
            if i > 0 and build_temp.exists():
                print(f"[lina_cpp] Wiping {build_temp} before retry.")
                shutil.rmtree(build_temp)
                build_temp.mkdir(parents=True, exist_ok=True)

            print(f"[lina_cpp] === attempt {i+1}/{len(attempts)}: {label} ===")
            print(f"[lina_cpp] Configuring CMake: {CPP_SOURCE_DIR} "
                  f"-> {build_temp}")
            try:
                subprocess.check_call(
                    ["cmake", str(CPP_SOURCE_DIR), *attempt_cmake_args],
                    cwd=build_temp,
                )
                print(f"[lina_cpp] Building extension into {ext_dir}")
                subprocess.check_call(
                    ["cmake", "--build", str(build_temp), *build_args],
                    cwd=build_temp,
                )
                break  # Success.
            except subprocess.CalledProcessError as exc:
                if is_last:
                    print(f"[lina_cpp] {label} build failed with exit "
                          f"{exc.returncode}; no more fallbacks.")
                    raise
                print(f"[lina_cpp] {label} build failed with exit "
                      f"{exc.returncode}; retrying with CPU-only.")

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
