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

# 4) Build + install lina_cpp (CPU only)
pip install -e ./lina_cpp/

# 5) Run the parity suite
python -m pytest lina/tests/test_per_method_parity.py -v

# 6) (Optional) Build with CUDA enabled
#    Requires CUDA Toolkit 12.4+ and a compatible host gcc.
#    See cpp/CMakeLists.txt header note about the CUDA 12.4 + glibc 2.40
#    incompatibility on Debian 13.
cmake -S cpp -B cpp/build-cuda \
    -DLINA_USE_CUDA=ON \
    -DLINA_BUILD_PYBIND=ON \
    -DCMAKE_CUDA_ARCHITECTURES="86" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build-cuda -j
# Re-install lina_cpp so it picks up the GPU-enabled extension:
pip install --force-reinstall --no-deps -e ./lina_cpp/
```

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
