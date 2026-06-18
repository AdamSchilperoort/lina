# lina → lina_cpp: C++/CUDA Conversion & Wrapper Guide

This document describes the C++/CUDA conversion of the `lina` wavefront sensing
& control package into `lina_cpp`: what was ported, where it now lives, the
naming conventions, and how the thin Python wrapper layer works.

`lina_cpp` is a **self-contained, standalone package**. It does not import the
externally installed `lina` at runtime; the still-Python helpers it needs are
vendored privately under `lina_cpp._pyref`. `lina_cpp` and the pure-Python
`lina` are two independent repositories with equivalent
functionality that can be installed side-by-side for parity comparison.

---

## 1. Repository layout

```
lina_cpp/                      # repo root & package root
├── cpp/                       # C++/CUDA implementation (the backend)
│   ├── include/lina/*.h       # public headers
│   ├── src/*.cpp              # CPU implementations
│   ├── src/*.cu               # CUDA implementations
│   ├── pybind/lina_py.cpp     # pybind11 bindings -> _core extension
│   └── CMakeLists.txt
├── src/lina_cpp/              # Python wrapper package
│   ├── __init__.py            # extension load, backend toggle, thread cap
│   ├── _dispatch.py           # CPU/GPU device selection
│   ├── _core*.so              # compiled pybind11 extension (built & gitignored)
│   ├── _pyref/                # vendored pure-Python reference (un-ported helpers)
│   ├── math_module.py utils.py props.py dm.py wfe.py control_models.py
│   ├── iefc.py efc.py aefc.py coro_utils.py llowfsc.py rt_utils.py ...
├── notebooks/
│   ├── sim_iefc_demo_lina_cpp_native.ipynb   # lina_cpp iefc, comparable to sim_iefc_demo in lina
│   └── ...
├── pyproject.toml             # installable package metadata (src/ layout)
├── setup.py                   # CMake-driven build of cpp/ -> _core.so
├── README.md
└── CONVERSION.md              # conversion notes py->c++
```

### Layout convention changes (from -> to)

| Old | New |
| --- | --- |
| `lina/<module>.py` (pure Python) | `cpp/src/<module>.cpp` (+ `cpp/include/lina/<module>.h`) for math; thin wrapper at `src/lina_cpp/<module>.py` |
| nested package `lina_cpp/lina_cpp/src/lina_cpp/` | `src/lina_cpp/` (repo root is package root) |
| duplicate `lina/` copy inside the C++ repo | removed; vendored as `src/lina_cpp/_pyref/` |
| compiled module imported ad hoc | single pybind11 extension `lina_cpp._core` |

---

## 2. Code Changes

**C++/CUDA backend changes:**
- `cpp/src/control_models_cuda.cu` *(new)* — device-resident GPU optical forward.
- `cpp/src/control_models.cpp`, `cpp/include/lina/control_models.h` — GPU dispatch,
  external-mask setters, native `dm_val_and_grad` + `solve_flat_command`.
- `cpp/src/iefc.cpp`, `cpp/include/lina/iefc.h` — native `calibrate_control_model`
  with progress callback.
- `cpp/src/linalg.cpp`, `cpp/src/linalg_cuda.cu`, `cpp/include/lina/linalg.h` —
  thin SVD (`svd_thin`), native CUDA `beta_reg_gpu` (cuBLAS + cuSOLVER Cholesky).
- `cpp/src/props.cpp`, `cpp/src/props_cuda.cu`, `cpp/include/lina/props.h` — MFT
  `npix` is now `double` (fixes a sub-pixel scale error).
- `cpp/pybind/lina_py.cpp` — bindings for all of the above.
- `cpp/CMakeLists.txt` — adds `control_models_cuda.cu` to the CUDA target.

**Python wrapper changes:**
- `src/lina_cpp/__init__.py` — backend toggle (`set_backend`), BLAS thread cap.
- `src/lina_cpp/utils.py` — `beta_reg` routes to native GPU kernel; `lstsq`,
  `tikhonov_inverse`, masks, `mean`, etc. delegate to C++.
- `src/lina_cpp/control_models.py` — `MODEL_CPP` (C++-backed), `solve_flat_command`.
- `src/lina_cpp/iefc.py` — native calibrate dispatch + progress printing.

---

## 3. Python -> C++ function re-mapping

The compiled extension is `lina_cpp._core` (built from `cpp/pybind/lina_py.cpp`).
Each Python wrapper module delegates its math hot paths to `_core` and re-exports
the remaining (un-ported) helpers from `lina_cpp._pyref`.

### Optical model — `control_models`
| Python (`lina`) | C++/CUDA | File |
| --- | --- | --- |
| `MODEL` (forward/snap) | `ControlModelCpp` class; GPU-resident `forward()` | `control_models.cpp`, **`control_models_cuda.cu`** |
| `val_and_grad` (EFC) | `control_model_val_and_grad` | `control_models.cpp` |
| `dm_val_and_grad` (flat-DM fit) | `control_model_dm_val_and_grad` | `control_models.cpp` |
| flat-DM solve (was SciPy L-BFGS-B) | `control_model_solve_flat` (CG / libLBFGS) | `control_models.cpp` |

### iEFC — `iefc`
| Python | C++ | File |
| --- | --- | --- |
| `calibrate` (per-mode loop) | `calibrate_control_model` (full native loop + progress cb) | `iefc.cpp` |
| `compute_hadamard_scale_factors` | `compute_hadamard_scale_factors` | `iefc.cpp` |

### Propagation — `props`
| Python | C++ (CPU) | CUDA (GPU) | File |
| --- | --- | --- | --- |
| `fft` / `ifft` | `fft_cpu` / `ifft_cpu` | `fft_gpu` / `ifft_gpu` (cuFFT) | `props.cpp`, `props_cuda.cu` |
| `ang_spec` | `ang_spec` | `ang_spec_gpu` | same |
| `mft_forward` / `mft_reverse` | same | `mft_forward_gpu` / `mft_reverse_gpu` (cuBLAS) | same |
| `make_vortex_phase_mask` | same | `make_vortex_phase_mask_gpu` | same |
| `get_fresnel_TF` | same | `get_fresnel_TF_gpu` | same |

### Linear algebra — `utils` / linalg
| Python | C++ (CPU) | CUDA (GPU) | File |
| --- | --- | --- | --- |
| `xp.matmul` | `gemm` (OpenBLAS) | (cuBLAS in kernels) | `linalg.cpp` |
| `xp.linalg.lstsq` | `lstsq` (thin SVD) | — | `linalg.cpp` (`svd_thin`) |
| `tikhonov_inverse` | `tikhonov_inverse` | — | pybind + `linalg.cpp` |
| `beta_reg` | `beta_reg` (OpenBLAS+LAPACKE) | **`beta_reg_gpu`** (cuBLAS GEMM + cuSOLVER Cholesky) | `linalg.cpp`, **`linalg_cuda.cu`** |
| `xp.linalg.svd` | `svd` / `svd_thin` | `svd_float_gpu` (cuSOLVER) | `linalg.cpp`, `linalg_cuda.cu` |

### Utilities — `utils`
| Python | C++ | File |
| --- | --- | --- |
| `mean`, `rms`, `mean(arr, mask)` | `mean`, `rms`, `mean_masked` | `utils.cpp` |
| `pad_or_crop` | `pad_or_crop` (header template) | `utils.h` |
| `make_grid` | `make_grid` | `utils.cpp` |
| `create_annular_mask`, `create_annular_focal_plane_mask` | same | `utils.cpp` |
| `save_fits` / `load_fits` | same (cfitsio) | `fits_io.cpp` |

### DM modes — `dm`
| Python | C++ | File |
| --- | --- | --- |
| `create_mask`, `make_gaussian_inf_fun`, `create_hadamard_modes`, `create_fourier_modes`, `make_fourier_command`, `make_f`, `make_ring`, `make_cross_command` | same names | `dm.cpp` |

### WFE — `wfe`
| Python | C++ | File |
| --- | --- | --- |
| Noll/Fringe index conversions | `noll_index_to_mn`, `mn_to_noll_index`, `fringe_index_to_mn`, `mn_to_fringe_index` | `wfe.cpp` |
| `generate_freqs`, `roll_psd`, `generate_time_series`, `compute_cumulative_psd` | `wfe_generate_freqs`, `wfe_roll_psd`, `wfe_generate_time_series`, `wfe_compute_cumulative_psd` | `wfe.cpp` |

### Coronagraph / LLOWFSC
| Python | C++ | File |
| --- | --- | --- |
| `coro_utils.normalize_coro_im`, `compute_contrast` | same | `coro_utils.cpp` |
| `llowfsc` core (acquire_ref, reconstruct, compute_zpo, loop_step) | `llowfsc_*` | `llowfsc.cpp` |

---

## 4. Native CUDA additions

- **Device-resident optical `forward`** (`control_models_cuda.cu`): the whole
  coronagraph forward (DM phasor, vortex dual-resolution propagation, Lyot,
  focal MFT) runs on the GPU with all intermediates kept on-device; only the
  final focal-plane field is copied to host.
- **`beta_reg_gpu`** (`linalg_cuda.cu`): control-matrix inverse as cuBLAS GEMM +
  cuSOLVER Cholesky solve, entirely on GPU.
- **Native iEFC `calibrate_control_model`**: calibration loop runs in C++
  calling the device-resident forward; 1024 modes in ~22 s with a per-mode progress 
  callback surfaced in the sim_iefc_demo_lina_cpp_native.ipynb notebook.

### "Correctness" fixes made during conversion
- MFT `npix` is a fractional **sampling scale** (`dx = 1/npix`), not an array
  size — it is now `double` (was truncated to `size_t`, causing a ~1e-4 PSF
  scale error / "shifted" PSF).
- The C++ model still uses the **poppy** aperture / Lyot stop / vortex masks (pushed
  in from Python via `set_aperture`/`set_lyotstop`/`set_windowed_vortex_*`/
  `set_dm_model`) instead of hard-edged C++ circles, so the gray-pixel edges
  match the reference (parity to ~1e-15).
- `svd_thin` (economy SVD): avoids an O(m²) full-U allocation that made
  `lstsq` over a flattened image (m ~ 1e5) try to allocate ~500 GB.

---

## 5. The Python wrapper library

`import lina_cpp as lina` is a near drop-in for `import lina`.

### Backend / device control
```python
import lina_cpp
lina_cpp.set_backend("gpu")     # numpy<->cupy (vendored _pyref) + C++ CPU<->CUDA, one switch
lina_cpp.set_backend("cpu")
lina_cpp.gpu_available()        # True if compiled with CUDA
lina_cpp.get_device()           # "cpu" | "gpu"
```
- `set_backend` switches both the native C++/CUDA device dispatcher and the
  vendored pure-Python reference (`_pyref`, for poppy/plotting helpers) so the
  whole stack runs on one device. The external `lina` is never touched.
- Per-call overrides exist on the propagation primitives, e.g.
  `lina_cpp.props.fft(arr, device="cpu")`.

### How dispatch works
- `src/lina_cpp/_dispatch.py` holds the module-level device (`cpu`/`gpu`) and
  `resolve_device(name, device)` per call.
- Each `*_gpu` C++ symbol is the CUDA variant; the wrapper picks `*_gpu` vs the
  CPU symbol based on the active device.

### Vendored Python reference (`_pyref`)
Un-ported helpers (poppy-based WFE/aperture/Zernike generation, matplotlib
plotting, file/pickle helpers, INDI hardware I/O, high-level `efc`/`aefc` run
loops) are vendored under `lina_cpp._pyref` and re-exported by the matching
wrapper module. This keeps `lina_cpp` self-contained (no dependency on an
external `lina`).

### CPU thread cap
`__init__.py` caps OpenBLAS/LAPACKE to `min(16, ncpu)` threads (override with
`OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` / `LINA_CPP_NUM_THREADS`). On large
shared machines OpenBLAS otherwise defaults to *all* cores and oversubscribes
catastrophically for the matrix sizes here (e.g. a 1024×1024 solve could take
minutes instead of <1 s).

---

## 6. What is still Python (and why)

lina_cpp back-end code will become a library linked in MagAOX for performing iefc and other math ops,
however this python-wrapped version is as a way of bridging the gap between the testing and simulation 
progressing in the pure python lina, enabling a side-by-side comparison before feeling confident about
the pure C++ functionality. iefc is fully converted but efc, aefc, have superficial python still:

| Area | Where | Why |
| --- | --- | --- |
| poppy WFE / aperture / Zernike (`wfe.generate_wfe`, `create_zernike_modes`, `dm.create_fourier_probes`) | `_pyref` | depends on poppy; mask/WFE generation, not a hot path |
| plotting (`utils.imshow`, `plot_*`) | `_pyref` | matplotlib |
| hardware I/O (`coro_utils` INDI camera/DM/stage functions) | `_pyref` | instrument control |
| high-level loops `efc.run`, `aefc.run`, `efc/aefc.calibrate` | `_pyref` | orchestration; the math primitives they call are C++ |
| flat-DM `dm_val_and_grad` (default) | `control_models.py` (cupy) | matches the reference at the weakly-constrained aperture-rim actuators; a native C++ endpoint (`control_model_dm_val_and_grad`) and a fully native solver (`control_model_solve_flat`) are also available (opt in with `LINA_CPP_NATIVE_DM_VG=1`) but `dm_val_and_grad` C++ is still suffering from errors |
| thin glue in `MODEL_CPP` (e.g. `xp.asarray` of C++ results, optional focal rotation, as rotation in cupy is different for some reason (e.g. 90 == 270 in C++)) | `control_models.py` | marshalling around the C++ forward |

---

## 7. Build & install

```bash
# From the repo root (CUDA auto-detected; force with LINA_USE_CUDA=1):
python -m pip install -e . --no-build-isolation

# Optional pure-Python baseline for parity comparison (kept separate):
python -m pip install -e /path/to/lina   # provides `import lina`
```
Requires: C++17, CMake ≥ 3.16, FFTW3, OpenBLAS, LAPACKE, cfitsio, libLBFGS, and
(optionally) the CUDA toolkit for the GPU paths.

After install:
```python
import lina_cpp
print("GPU:", lina_cpp.gpu_available())
```

---

## 8. Parity status

The `sim_iefc_demo_lina_cpp_native.ipynb` (lina_cpp) reproduces
`sim_iefc_demo.ipynb` (pure-Python `lina`) end-to-end:
- Reference PSF & coronagraph image: match to ~1e-15.
- Flat-DM command (incl. rim actuators): match to ~1e-6.
- iEFC calibration response matrix / cube: match to ~1e-14.
- `beta_reg` control matrix: match to ~1e-9…1e-12.

Residual ~1e-6 differences trace to one step (Zernike removal in
`generate_wfe`) where the GPU SVD differs from cupy's at the floating-point
level — not a logic difference.
