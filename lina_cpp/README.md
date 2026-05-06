# lina_cpp

C++-backed mirror of the [lina](../lina) wavefront sensing & control
package. The Python API surface is identical to `lina`, so swapping
backends is a one-line change:

```python
# Before:
import lina
mask = lina.utils.create_annular_mask(256, 5, 30)

# After (numerically identical, faster on the math hot paths):
import lina_cpp as lina
mask = lina.utils.create_annular_mask(256, 5, 30)
```

## Coverage

| Module                | C++-backed today                           | Re-exported from `lina` |
| --------------------- | ------------------------------------------ | ----------------------- |
| `utils`               | mean, rms, pad_or_crop, lstsq, tikhonov_inverse, beta_reg, create_annular_mask, create_annular_focal_plane_mask, make_grid, save_fits, load_fits | imshow, create_zernike_modes, rotate_arr, interp_arr, plot_radial_contrast, fs helpers |
| `props`               | fft, ifft, ang_spec, mft_forward, mft_reverse, make_vortex_phase_mask, get_fresnel_TF | get_scaled_coords, make_mft_*_matrices |
| `dm`                  | create_mask, make_gaussian_inf_fun, create_hadamard_modes, create_fourier_modes, make_fourier_command, make_f, make_ring, make_cross_command | create_all_poke_modes, create_fourier_probes |
| `coro_utils`          | normalize_coro_im, compute_contrast        | all 28 INDI-driven hardware functions  |
| `iefc`                | compute_hadamard_scale_factors             | calibrate, run, init_data, ...         |
| `efc`, `aefc`         | (math primitives indirectly)               | calibrate, run, init_data, ...         |
| `llowfsc`             | -                                          | the entire module (loop + helpers)     |
| `rt_utils`            | -                                          | the entire module                      |
| `wfe`                 | -                                          | the entire module                      |
| `math_module`         | -                                          | the entire module (xp / xcipy shim)    |

The re-exports use `lina` as a hard runtime dependency so the public
API surface stays complete. Future work will replace each row of the
right-hand column with a native C++ implementation; doing so does not
require any change to user code.

## Installation

```bash
# From the repo root:
pip install -e ./lina_cpp/
```

The build invokes CMake against `../cpp` and stages the resulting
extension at `lina_cpp/_core.cpython-...-...so`. Build options can be
passed via the `LINA_CMAKE_ARGS` env var, e.g.:

```bash
# CUDA build:
LINA_CMAKE_ARGS="-DLINA_USE_CUDA=ON" pip install -e ./lina_cpp/

# Pin a specific cfitsio:
LINA_CMAKE_ARGS="-DCFITSIO_LIBRARY=/opt/cfitsio/lib/libcfitsio.so" pip install -e ./lina_cpp/
```

System requirements (these are the same as for the standalone CMake
build of the C++ tree):

- C++17 compiler, CMake >= 3.16
- FFTW3 (math), OpenBLAS (gemm/gemv), LAPACKE (svd)
- ImageStreamIO (shmim), libLBFGS (optimization)
- cfitsio (FITS I/O; required for `lina_cpp.utils.save_fits`)
- (optional) CUDA toolkit for GPU code paths

## Performance comparison

A side-by-side parity / timing notebook lives at
`notebooks/compare_lina_vs_lina_cpp.ipynb`. The pattern is::

```python
import lina, lina_cpp
import numpy as np
import time

x = np.random.RandomState(0).standard_normal((256, 256)).astype(np.complex128)

t0 = time.perf_counter(); _ = lina.props.fft(x);     t_py  = time.perf_counter() - t0
t0 = time.perf_counter(); _ = lina_cpp.props.fft(x); t_cpp = time.perf_counter() - t0
print(f"fft 256x256: lina={t_py*1e3:.1f} ms  lina_cpp={t_cpp*1e3:.1f} ms")
```

Numerical equivalence is verified by `lina/tests/test_per_method_parity.py`
(36 passing tests covering every wrapped function).
