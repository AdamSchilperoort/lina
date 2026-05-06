# Lina C++ Port (WIP)

This directory contains a C++ translation of core numerical routines from the
Python `lina` package, intended for benchmarking and reducing runtime
dependencies on Conda-only libraries.

## Build

From the repository root:

```
cmake -S cpp -B cpp/build
cmake --build cpp/build
cmake --install cpp/build --prefix /opt/MagAOX/local
```

Dependencies (system):
- `liblapacke-dev`, `libopenblas-dev`, `gfortran`

Dependencies (MagAOX vendor):
- FFTW3 under `/opt/MagAOX/vendor/fftw-3.3.8`
- Eigen under `/opt/MagAOX/vendor/eigen-3.3.4` (fallback SVD)

Environment notes:
- ImageStreamIO headers/libs are expected under `/opt/MagAOX/source/milk/src/ImageStreamIO`.

If FFTW3 is installed, the build will automatically link it when
`LINA_USE_FFTW=ON` (default). The current implementation falls back to a naive
FFT implementation if FFTW3 is not detected.

Optional optimized backends:
- OpenBLAS (CBLAS) for GEMM/GEMV when `LINA_USE_OPENBLAS=ON`.
- LAPACKE for CPU SVD when `LINA_USE_LAPACKE=ON`.
- CUDA/cuSOLVER for GPU SVD when `LINA_USE_CUDA=ON`.
- libLBFGS for L-BFGS optimization when `LINA_USE_LBFGS=ON`.

ImageStreamIO is a required dependency for shmim support. Provide
`IMAGESTREAMIO_ROOT` pointing at the MagAOX `ImageStreamIO` source/build root
(default: `/opt/MagAOX/source/milk/src/ImageStreamIO`).

## Implemented Modules

- `lina/utils`: `mean`, `rms`, `make_grid`, `pad_or_crop`
- `lina/props`: `fft`, `ifft`, `mft_forward`, `mft_reverse`
- `lina/coro_utils`: `normalize_coro_im`, `compute_contrast`
- `lina/linalg`: `gemm`, `gemv`, `svd`
- `lina/shmim_utils`: `ShmimStream` wrapper for ImageStreamIO
- `lina/dm`: DM masks, Gaussian influence function, Fourier/Hadamard modes
- `lina/efc`, `lina/iefc`, `lina/aefc`: typed algorithm ports (plotting removed)
- `lina/control_models`: optical propagation model (Poppy-free)
- `lina/pwp`: pairwise probing solver (uses LAPACKE SVD)

Notes:
- `lina::PwpEstimator` and `lina::Optimizer` are abstract interfaces; provide
  concrete implementations in your application for PWP estimation and L-BFGS
  (or another optimizer) if needed.

Include in C++:
```
#include <lina.h>
```

Find from other CMake projects:
```
set(CMAKE_PREFIX_PATH "/opt/MagAOX/local")
find_package(lina REQUIRED)
target_link_libraries(your_app PRIVATE lina::lina_cpp)
```

## Running

The build produces `lina_runner`, a small CLI used by tests:

```
./cpp/build/lina_runner fft
./cpp/build/lina_runner svd
./cpp/build/lina_runner efc
./cpp/build/lina_runner iefc
./cpp/build/lina_runner shmim
```

Benchmarks:
```
./cpp/build/lina_runner bench_fft
./cpp/build/lina_runner bench_svd
./cpp/build/lina_runner bench_efc
./cpp/build/lina_runner bench_iefc
```

## Unit Tests (Python parity)

Activate conda base (for numpy/scipy):
```
source /home/adamschilperoort/conda/etc/profile.d/conda.sh
conda activate base
```

Run parity tests:
```
python -m unittest lina.tests.test_cpp_parity lina.tests.test_more_python
```

Run timing benchmarks (optional):
```
LINA_RUN_BENCHMARKS=1 python -m unittest lina.tests.test_benchmarks
```

## Matrix and shmim comparisons

The parity tests compare FFT/SVD outputs and the EFC/iEFC command matrices.

For shmim, compare the shared memory image files directly with Linux tools.
Both C++ and Python write to `/milk/shm/<name>.im.shm`, so you can verify
the stream contents using MagAOX/MILK tools (e.g., `shmimstats` or
`shmimview`) rather than dumping matrices to disk.

## Next Steps

- Replace naive FFT with FFTW3 or another optimized backend.
- Add bindings or a CLI harness for direct benchmarking.
- Port higher-level EFC/iEFC algorithms and model propagation.
