# Lina C++ Port — Files To Commit

This guide tells you exactly what to `git add` after the C++ port work
in this repo, what to leave out, and how to set up another machine
after pulling the changes.

The new tree breaks into four pieces:

| Directory                    | Purpose                                                |
| ---------------------------- | ------------------------------------------------------ |
| `cpp/`                       | The C++ library, headers, CUDA kernels, pybind bindings, and CLI runner |
| `lina_cpp/`                  | The Python package that wraps the C++ extension as a drop-in for `lina` |
| `lina/tests/`                | New per-method parity tests + helper runners            |
| `notebooks/`                 | Side-by-side comparison notebook                        |

## 1. Quick way to add everything new

After pulling, from the repo root:

```bash
git add .gitignore
git add cpp/
git add lina_cpp/
git add lina/tests/
git add notebooks/compare_lina_vs_lina_cpp.ipynb
git add CHECKIN.md
```

The `.gitignore` files I added (`./.gitignore`, `cpp/.gitignore`,
`lina_cpp/.gitignore`) will exclude every build artifact, so the above
commands will not pick up any compiled or generated files. You can
always sanity-check with:

```bash
git status --short                  # should only show new sources/docs
git diff --cached --stat            # confirm what is about to be committed
```

If you would rather see the explicit file lists (no shell globs), the
checked-in inventory is below.

## 2. What to commit

### Top-level

- `.gitignore` (updated)
- `CHECKIN.md` (this file)

### C++ library tree (`cpp/`)

- Build files
  - `cpp/CMakeLists.txt`
  - `cpp/cmake/linaConfig.cmake.in`
- Public headers (`cpp/include/`)
  - `cpp/include/lina.h`
  - `cpp/include/lina/{aefc,array,control_models,coro_utils,dm,efc,grid2d,
    iefc,lina,linalg,llowfsc,math_module,props,pwp,shmim_utils,stream,utils,
    wfe}.h`
- C++ implementations (`cpp/src/`)
  - `cpp/src/{aefc,array,control_models,coro_utils,dm,efc,fits_io,iefc,
    linalg,llowfsc,props,pwp,shmim_utils,utils,wfe}.cpp`
- CUDA implementations (`cpp/src/`)
  - `cpp/src/linalg_cuda.cu`
  - `cpp/src/props_cuda.cu`
- Pybind bindings
  - `cpp/pybind/lina_py.cpp`
- CLI runner used by some parity / benchmark tests
  - `cpp/tools/lina_runner.cpp`
- Documentation
  - `cpp/AUDIT.md`
  - `cpp/INTEGRATION.md`
  - `cpp/README.md`
- `cpp/.gitignore` (so build artifacts stay local)

### Python wrapper package (`lina_cpp/`)

- `lina_cpp/.gitignore`
- `lina_cpp/pyproject.toml`
- `lina_cpp/setup.py`  (custom `setuptools` step that drives CMake)
- `lina_cpp/README.md`
- Source layout (`lina_cpp/src/lina_cpp/`):
  - `__init__.py`
  - `math_module.py`, `utils.py`, `props.py`, `dm.py`, `coro_utils.py`,
    `llowfsc.py`, `rt_utils.py`, `wfe.py`, `efc.py`, `iefc.py`, `aefc.py`

### Tests (`lina/tests/`)

- `lina/tests/test_per_method_parity.py`  ← the canonical 46-test suite
- `lina/tests/test_pybind_bridge.py`
- `lina/tests/test_cpp_parity.py`         ← uses `lina_runner` CLI
- `lina/tests/test_benchmarks.py`
- `lina/tests/test_more_python.py`
- `lina/tests/run_native_tests.sh`
- `lina/tests/run_cuda_container_tests.sh`

### Notebook

- `notebooks/compare_lina_vs_lina_cpp.ipynb`

## 3. What NOT to commit

The `.gitignore` files cover all of these, so you should never see them
in `git status`. Leaving them out is critical because they're machine-,
toolchain-, and absolute-path-dependent.

- Any `cpp/build*/`, `cpp/install/`
- Any in-source CMake artifacts at the top of `cpp/`:
  `CMakeCache.txt`, `CMakeFiles/`, `cmake_install.cmake`, `Makefile`,
  `compile_commands.json`, `CMakeUserPresets.json`
- Compiled outputs anywhere: `*.o`, `*.a`, `*.so`, `*.cpython-*.so`,
  `lina_cpp/build/`, `lina_cpp/dist/`, `lina_cpp/src/lina_cpp.egg-info/`,
  `lina_cpp/src/lina_cpp/_core.cpython-*.so`
- Python bytecode: `__pycache__/`, `*.pyc`, `*.egg-info`
- IDE scratch: `.vscode/`, `.idea/`, `.cache/`, `.pytest_cache/`

> If `git status` ever shows `cpp/CMakeFiles/` or similar, that means
> someone ran `cmake .` directly inside `cpp/` instead of using an
> out-of-source build directory. Delete the in-source clutter before
> committing -- the canonical build is always under
> `cpp/build/` or `cpp/build-cuda/`.

## 4. Setting up a fresh machine after pulling

```bash
# 1) Repo
git clone <fork-url> AdamSchilperoort_lina
cd AdamSchilperoort_lina

# 2) System dependencies (Ubuntu 22.04 / RHEL 9 names; adapt as needed)
sudo apt-get install -y build-essential cmake \
    libfftw3-dev libfftw3-double-dev \
    libopenblas-dev liblapack-dev liblapacke-dev \
    libcfitsio-dev \
    pybind11-dev

# 3) Python deps + the original lina (the wrapper re-exports from it)
pip install -e .                       # installs `lina`
pip install scipy numpy poppy pybind11

# 4) Build + install lina_cpp.
#
#    GPU support is auto-detected: if a working CUDA toolkit is on PATH,
#    the GPU backend (cuFFT, cuBLAS, cuSOLVER, custom kernels) is
#    enabled automatically. Watch the build log for:
#
#        === lina_cpp configuration ===
#          GPU backend : ENABLED  (cuFFT, cuBLAS, cuSOLVER, custom kernels)
#
#    or, on a CPU-only host:
#
#          GPU backend : disabled (CPU-only build)
#
pip install -e ./lina_cpp/

# 5) Verify
python -c "import lina_cpp; print('GPU:', lina_cpp.gpu_available())"

# 6) Run the parity suite
python -m pytest lina/tests/test_per_method_parity.py -v
```

### Forcing GPU on or off

If you want to override the auto-detect (e.g. you have CUDA installed
but want a CPU-only build, or you have a non-standard toolkit layout
and want to force GPU and fail loudly if it can't link), set
`LINA_USE_CUDA` before `pip install`:

```bash
# Force GPU build (fails the build if CUDA isn't available):
LINA_USE_CUDA=1 pip install --force-reinstall --no-deps -e ./lina_cpp/

# Force CPU-only build (skip the auto-detect probe):
LINA_USE_CUDA=0 pip install --force-reinstall --no-deps -e ./lina_cpp/

# Pick a specific CUDA architecture:
LINA_USE_CUDA=1 LINA_CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=89" \
    pip install --force-reinstall --no-deps -e ./lina_cpp/
```

If you previously installed without CUDA and now want to enable it,
remember to add `--force-reinstall --no-deps` or pip will short-circuit
the rebuild and you'll still see `gpu_available() == False`.

> **Note on CUDA 12.4 + glibc 2.40 (Debian 13)**: `nvcc` from CUDA 12.4
> fails to compile against Debian 13's glibc 2.40. Use CUDA 12.6+, or
> downgrade to glibc 2.39, or build inside an Ubuntu 22.04 container.
> See the header note in `cpp/CMakeLists.txt`.

### Troubleshooting: stale `_core.so`

`pip install -e .` does **not** always recompile the C++ extension when
only the C++ sources change. Setuptools sees no new `.py` files and
skips `build_ext`. CMake also reuses `lina_cpp/build/` across runs, so a
half-built `.so` from a failed earlier attempt can stick around.

The canonical symptom of a stale binary on the parity suite is
`lina_cpp.llowfsc_reconstruct` returning a constant vector (every
coefficient equal to `coeff[0]`), which then triggers a cascade of
"constant DESIRED array" failures in
`lina/tests/test_per_method_parity.py::TestLlowfsc::*` and
`TestWfe::*`. The smoke test catches this explicitly:

```bash
python -m lina_cpp.smoke
# ...
# === llowfsc_reconstruct sanity (catches stale _core.so) ===
#   [FAIL] llowfsc_reconstruct produces non-constant vector: output is constant ...
```

If you ever see that, run the nuclear rebuild helper which wipes every
cache layer (the editable-install hash, the cmake build dir, the
in-tree `_core*.so`, the pip download cache) and re-installs:

```bash
bash scripts/clean_rebuild_lina_cpp.sh
# or, to force a specific backend:
LINA_USE_CUDA=1 bash scripts/clean_rebuild_lina_cpp.sh   # force GPU
LINA_USE_CUDA=0 bash scripts/clean_rebuild_lina_cpp.sh   # force CPU
```

After that script finishes the parity suite should report a clean
`46 passed, 1 xfailed` (or similar -- the actual number of tests grows
over time).

## 4b. Choosing CPU vs GPU at runtime

Every math hot path in `lina_cpp.props` has both a CPU and a GPU C++
implementation (FFTW / OpenBLAS for CPU; cuFFT / cuBLAS / cuSOLVER for
GPU). You select between them three ways:

```python
import lina_cpp
import numpy as np

# 1. Check what this build supports
lina_cpp.gpu_available()        # True iff compiled with -DLINA_USE_CUDA=ON
lina_cpp.get_device()           # 'cpu' or 'gpu'

# 2. Module-level default (sticky for the rest of the session)
lina_cpp.set_device("gpu")      # all subsequent calls go to GPU
lina_cpp.set_device("cpu")      # back to CPU

# 3. Per-call override
ft_gpu = lina_cpp.props.fft(arr, device="gpu")
ft_cpu = lina_cpp.props.fft(arr, device="cpu")
psf    = lina_cpp.props.mft_forward(wf, npix, npsf, du, device="gpu")
```

You can also set the default from the environment, which is useful for
benchmarking scripts and notebooks:

```bash
LINA_CPP_DEVICE=gpu jupyter lab
LINA_CPP_DEVICE=cpu python bench.py
```

Behavior on a CPU-only build (`LINA_USE_CUDA=OFF`):
- `lina_cpp.gpu_available()` returns `False`.
- `lina_cpp.set_device("gpu")` raises `RuntimeError` (so you can't
  silently end up on CPU when you wanted GPU).
- A per-call `device="gpu"` falls back to CPU and emits a one-shot
  `RuntimeWarning` so you know what happened.

Functions with `device=` support today:

| Function                                | CPU backend | GPU backend |
| --------------------------------------- | ----------- | ----------- |
| `lina_cpp.props.fft`                    | FFTW        | cuFFT       |
| `lina_cpp.props.ifft`                   | FFTW        | cuFFT       |
| `lina_cpp.props.ang_spec`               | FFTW        | cuFFT       |
| `lina_cpp.props.mft_forward`            | triple loop | cuBLAS zgemm|
| `lina_cpp.props.mft_reverse`            | triple loop | cuBLAS zgemm|
| `lina_cpp.props.make_vortex_phase_mask` | OpenMP      | CUDA kernel |
| `lina_cpp.props.get_fresnel_TF`         | OpenMP      | CUDA kernel |

Other modules (`utils`, `dm`, `efc`, `iefc`, `aefc`, `llowfsc`, `wfe`)
are CPU-only in the C++ extension today; their wrappers accept (and
ignore with a one-shot warning) a `device=` kwarg so that user code
written against the dispatch API doesn't break when those eventually
get GPU kernels.

## 5. After it's tested and you want to merge into the original `lina` repo

The `cpp/` and `lina_cpp/` trees are self-contained additions. To
graft them into `~/Steward/repos/lina`:

```bash
# from the original lina/ checkout
cp -a ../AdamSchilperoort_lina/cpp .
cp -a ../AdamSchilperoort_lina/lina_cpp .
cp ../AdamSchilperoort_lina/lina/tests/test_per_method_parity.py lina/tests/
cp ../AdamSchilperoort_lina/lina/tests/test_pybind_bridge.py     lina/tests/
cp ../AdamSchilperoort_lina/lina/tests/test_cpp_parity.py        lina/tests/
cp ../AdamSchilperoort_lina/lina/tests/test_benchmarks.py        lina/tests/
cp ../AdamSchilperoort_lina/lina/tests/test_more_python.py       lina/tests/
cp ../AdamSchilperoort_lina/lina/tests/run_native_tests.sh       lina/tests/
cp ../AdamSchilperoort_lina/lina/tests/run_cuda_container_tests.sh lina/tests/
cp ../AdamSchilperoort_lina/notebooks/compare_lina_vs_lina_cpp.ipynb notebooks/

# Add .gitignore updates (cpp/.gitignore, lina_cpp/.gitignore, root .gitignore)
git add cpp/ lina_cpp/ lina/tests/ notebooks/compare_lina_vs_lina_cpp.ipynb
git add cpp/.gitignore lina_cpp/.gitignore .gitignore
```

The two repos share the same `lina/` Python package, so no copying is
required for that side.

## 6. MagAOX migration roadmap (future work)

The current C++ kernels are deliberately written to be MagAOX-app-friendly:

- `cpp/src/llowfsc.cpp` exposes `loop_step()` -- one full closed-loop
  math iteration -- with an interface that takes plain pointers and
  Array2D buffers. A future `magaox-llowfsc-app` can call this directly
  inside its real-time loop after pulling pixels from ImageStreamIO.
- `cpp/src/efc.cpp`, `cpp/src/iefc.cpp`, `cpp/src/aefc.cpp` are similarly
  structured. The Python orchestration in `lina_cpp/src/lina_cpp/{efc,
  iefc,aefc}.py` is what the future MagAOX apps will replace; the math
  primitives stay.
- The 28 INDI hardware-control functions in `lina/coro_utils.py` are
  *not* ported to C++ here. They will be re-implemented inside each
  MagAOX app using the existing `libcommon`/INDI infrastructure that
  every MagAOX app already links to (`MagAOXApp.hpp`, `dev::stdMotionStage`,
  etc.). The math kernels in `coro_utils.cpp` (`normalize_coro_im`,
  `compute_contrast`) are reusable across all three future apps.
