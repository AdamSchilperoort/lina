# lina_cpp

C++/CUDA-backed implementation of the `lina` wavefront sensing & control
package, with a thin Python wrapper layer. It is a **self-contained,
standalone package**: it does not import or depend on the pure-Python
`lina` at runtime. The still-Python helpers it needs (poppy-based WFE
generation, plotting, a few utilities) are vendored privately under
`lina_cpp._pyref`.

`lina_cpp` and the pure-Python `lina` are now
maintained as two separate, independent repositories with equivalent
functionality. They can be installed side-by-side in one environment for
parity comparisons, but neither imports the other.

The Python API in `lina_cpp` surface is intended to mirror `lina` to make it easy to compare and run both:

```python
import lina_cpp as lina
lina.set_backend("gpu")                       # numpy<->cupy + C++ CPU<->CUDA
mask = lina.utils.create_annular_mask(256, 5, 30)
```

## Layout

```
lina_cpp/                 # repo root == package root
├── cpp/                  # C++/CUDA sources (back-end implementation)
├── src/lina_cpp/         # Python package (thin wrapper layer)
│   ├── _core*.so         # compiled pybind11 extension
│   └── _pyref/           # vendored pure-Python reference (un-ported helpers)
├── notebooks/            # similar to lina notebooks
├── pyproject.toml
└── setup.py              # drives a CMake build of cpp/ on pip intsall
```

## Installation

```bash
# From the repo root:
python -m pip install -e . --no-build-isolation

# Force the GPU/CUDA build explicitly (otherwise auto-detected):
LINA_USE_CUDA=1 python -m pip install -e . --no-build-isolation

# Extra CMake flags for explicitly finding fitsio:
LINA_CMAKE_ARGS="-DCFITSIO_LIBRARY=/opt/cfitsio/lib/libcfitsio.so" \
    python -m pip install -e . --no-build-isolation
```

The build invokes CMake against `./cpp` and stages the resulting
extension at `src/lina_cpp/_core.cpython-...-...so`.

System requirements:

- C++17 compiler, CMake >= 3.16
- FFTW3 (CPU FFT), OpenBLAS (gemm/gemv), LAPACKE (svd)
- cfitsio (FITS I/O), libLBFGS (optimization), ImageStreamIO (shmim interaction)
- Appropriately-versioned CUDA toolkit for the GPU

## Coverage

Math hot paths run in C++/CUDA. The optical `ControlModel.forward` and the
`iefc` calibration loop are fully native (GPU-resident on CUDA builds).
Remaining un-ported helpers (poppy WFE, plotting, `make_mft_*_matrices`,
`create_fourier_probes`, etc.) are served from the vendored
`lina_cpp._pyref` so the public API stays complete. Future work may replace
those with fully native C++ without changing the python wrapper layer, so 
notebooks should still work if the backend switches to pure c++.

## Parity comparison

To verify `lina` and `lina_cpp` are swappable, install both and run the
parity test suite shipped with `lina_cpp` in `tests/test_parity.py`. It runs
each math-backend function (and a small end-to-end optical model) on the
pure-Python `lina` and on the C++/CUDA `lina_cpp`, then asserts the results
agree to a sensible tolerance — pytest reports a per-function pass/fail.

The suite lives entirely in this repo and never modifies the `lina` package;
it imports `lina` only as an external reference for comparison (and is skipped
if `lina` is not installed).

Assuming `lina_cpp` is already installed, install `lina` alongside it and
`pytest`:

```bash
python -m pip install -e /path/to/lina   # provides `import lina`
python -m pip install pytest
```

Run the suite from the repo root:

```bash
# Everything (math backend + small end-to-end control-model / iEFC checks)
python -m pytest tests/test_parity.py -v

# Just the fast math-backend functions (skip the model/iEFC end-to-end tests)
python -m pytest tests/test_parity.py -v -m "not slow"

# Compare on the GPU backend instead of CPU (requires a CUDA build + cupy)
LINA_PARITY_DEVICE=gpu python -m pytest tests/test_parity.py -v
```

Expected output is a per-function report, e.g.:

```text
tests/test_parity.py::test_fft PASSED
tests/test_parity.py::test_mft_forward PASSED
tests/test_parity.py::test_lstsq PASSED
tests/test_parity.py::test_beta_reg PASSED
...
tests/test_parity.py::test_model_snap_vortex PASSED
tests/test_parity.py::test_iefc_calibrate PASSED
======================= 35 passed in 8.24s =======================
```

What it covers (small inputs / matrices, so it runs in seconds):

- `props`: `fft`, `ifft`, `mft_forward`, `mft_reverse`, `ang_spec`,
  `make_vortex_phase_mask`, `get_fresnel_TF`
- `utils`: `mean`, `rms`, `pad_or_crop`, `create_annular_mask`,
  `create_annular_focal_plane_mask`, `lstsq`, `tikhonov_inverse`, `beta_reg`
- `dm`: `create_mask`, `create_hadamard_modes`, `make_gaussian_inf_fun`,
  `make_fourier_command`, `make_f`
- `wfe`: `noll_index_to_mn`, `generate_freqs`, `roll_psd`
- `coro_utils`: `compute_contrast`
- `control_models` / `iefc` (marked `slow`): `MODEL.snap` (with and without
  the vortex), `dm_val_and_grad`, and `iefc.calibrate` on a small model — the
  same operations exercised by `sim_iefc_demo`.

The comparison runs on the CPU backend by default (deterministic, fast,
independent of GPU load); the math is identical to the CUDA path, which you
can exercise with `LINA_PARITY_DEVICE=gpu`.

For a quick one-off timing comparison without pytest:

```python
import lina, lina_cpp, numpy as np, time
x = np.random.RandomState(0).standard_normal((256, 256)).astype(np.complex128)
t0 = time.perf_counter(); _ = lina.props.fft(x);     t_py  = time.perf_counter() - t0
t0 = time.perf_counter(); _ = lina_cpp.props.fft(x); t_cpp = time.perf_counter() - t0
print(f"fft 256x256: lina={t_py*1e3:.1f} ms  lina_cpp={t_cpp*1e3:.1f} ms")
```

`lina_cpp.bench` and `lina_cpp.smoke` contain helpers that import the
external `lina` lazily for these comparisons; together with
`tests/test_parity.py` they are the only places that reference it, and
`lina_cpp` runs without `lina` installed.
