# Using `lina` from a Larger C++ Program

This document describes how to consume the `lina` C++ library from a
downstream project, both via CMake's `find_package` and through direct
linkage. It assumes the library has already been built and installed (see
`cpp/README.md`).

## 1. Quick recap of what you get

After running

```bash
cmake -S cpp -B cpp/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/opt/MagAOX/local
cmake --build cpp/build -j$(nproc)
cmake --install cpp/build
```

you end up with the following layout under the install prefix:

```
<prefix>/include/lina.h
<prefix>/include/lina/*.h            # public headers (utils, props, ...)
<prefix>/lib/liblina_cpp.a           # static archive
<prefix>/lib/liblina.so              # shared library (if LINA_BUILD_SHARED=ON)
<prefix>/lib/cmake/lina/linaConfig.cmake          # CMake package config
<prefix>/lib/cmake/lina/linaTargets*.cmake        # exported targets
<prefix>/bin/lina_runner                          # CLI used by parity tests
```

The CMake package exports two targets:

| Target              | Library        | Notes                          |
| ------------------- | -------------- | ------------------------------ |
| `lina::lina_cpp`    | `liblina_cpp.a`| Static linkage                 |
| `lina::lina_cpp_shared` | `liblina.so` | Shared linkage (recommended)  |

Both bring along their PUBLIC dependencies (FFTW3, OpenBLAS/CBLAS, LAPACKE,
optionally CUDA, optionally libLBFGS, ImageStreamIO) via their interface
properties — so consumers do **not** need to add those dependencies again
manually.

## 2. The recommended way: CMake `find_package`

In your downstream project's `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.16)
project(my_app LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Tell CMake where to find the lina package config.
list(APPEND CMAKE_PREFIX_PATH "/opt/MagAOX/local")

find_package(lina REQUIRED)

add_executable(my_app src/main.cpp)
target_link_libraries(my_app PRIVATE lina::lina_cpp_shared)
```

Configure and build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

If the install prefix is non-standard, also set the runtime loader path so
`libina.so` is discoverable:

```cmake
set_target_properties(my_app PROPERTIES
    INSTALL_RPATH "/opt/MagAOX/local/lib"
    BUILD_RPATH   "/opt/MagAOX/local/lib")
```

## 3. The unmediated way: direct linkage

If your project does not use CMake (or you don't want `find_package`):

```bash
g++ -std=c++17 -O3 \
    -I/opt/MagAOX/local/include \
    src/main.cpp \
    -L/opt/MagAOX/local/lib -llina \
    -Wl,-rpath,/opt/MagAOX/local/lib \
    -lfftw3 -lfftw3_threads -lopenblas -llapacke -lImageStreamIO -lpthread \
    -o my_app
```

You may also need:

- `-llbfgs` if you built lina with `LINA_USE_LBFGS=ON` and use `LbfgsOptimizer`.
- `-lcudart -lcusolver -lcufft` if you built lina with `LINA_USE_CUDA=ON`
  and call any GPU code paths.

The static archive `liblina_cpp.a` works the same way — just substitute
`-llina_cpp` for `-llina` and you'll need to provide all transitive deps
yourself (which is why CMake `find_package` is recommended).

## 4. Minimal end-to-end example

Below is a self-contained example that constructs a small DM mask and
applies an iEFC-style probe-difference with the C++ MFT. It illustrates
how to use the public API in `lina.h`.

`src/main.cpp`:

```cpp
#include <lina.h>

#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

int main() {
    using namespace lina;

    constexpr std::size_t Nact   = 34;       // 34x34 DM
    constexpr std::size_t npix   = 256;      // pupil sampling
    constexpr std::size_t Ndef   = npix + 2; // defocus oversample
    constexpr std::size_t ncamsci = 96;
    constexpr double psf_pxscl_lamD = 0.354;

    // 1. DM mask (active actuators)
    auto dm_mask = create_mask(Nact);
    std::size_t nacts = 0;
    for (std::size_t i = 0; i < dm_mask.size(); ++i) {
        if (dm_mask.data()[i]) ++nacts;
    }
    std::cout << "Active actuators: " << nacts << "\n";

    // Note on `create_annular_mask` / `create_annular_focal_plane_mask`:
    //   Default `edge` is `kNoEdgeFilter` (no half-plane cut, equivalent to
    //   Python's `edge=None`). Pass any explicit numeric value (including
    //   0.0) to apply the cut `xr > edge` (Python's numeric `edge`
    //   semantics). See `cpp/AUDIT.md`.

    // 2. Build a small Fourier "probe" command
    auto probe = make_fourier_command(/*x_cpa=*/8, /*y_cpa=*/0,
                                      /*nact=*/Nact, /*phase=*/0.0);

    // 3. Inject the probe through a (placeholder) flat pupil and propagate
    //    via MFT to the focal plane. In a real application you'd use
    //    ControlModel::forward instead.
    Array2D<std::complex<double>> pupil(Ndef, Ndef, {0.0, 0.0});
    const auto aperture = create_annular_mask(
        Ndef, /*pixelscale=*/1.0,
        /*irad=*/0.0, /*orad=*/static_cast<double>(npix) / 2.0);
    for (std::size_t r = 0; r < Ndef; ++r) {
        for (std::size_t c = 0; c < Ndef; ++c) {
            pupil(r, c) = aperture(r, c) ? std::complex<double>(1.0, 0.0)
                                         : std::complex<double>(0.0, 0.0);
        }
    }

    auto e_fp = mft_forward(pupil, npix, ncamsci, psf_pxscl_lamD,
                            /*convention=*/'-',
                            /*pp_centering=*/"odd",
                            /*fp_centering=*/"odd");

    // 4. Compute mean intensity over an annular dark-hole mask
    Array2D<double> intensity(ncamsci, ncamsci, 0.0);
    for (std::size_t i = 0; i < intensity.size(); ++i) {
        intensity.data()[i] = std::norm(e_fp.data()[i]);
    }
    auto dh_mask = create_annular_focal_plane_mask(
        ncamsci, psf_pxscl_lamD,
        /*irad=*/2.5, /*orad=*/13.0);

    std::vector<std::uint8_t> mask_flat(dh_mask.data(),
                                        dh_mask.data() + dh_mask.size());
    const double mean_dh = mean(intensity, &mask_flat);
    std::cout << "Mean DH intensity = " << mean_dh << "\n";
    return 0;
}
```

Build & run with the CMake setup from §2:

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build && ./build/my_app
```

## 5. Pulling in the `ControlModel` (optical model)

`lina::ControlModel` is the C++ port of the Python `MODEL` class. Once
constructed, `forward(actuators, wavelength, use_vortex)` returns the
focal-plane field — a single self-contained call.

```cpp
#include <lina.h>
#include <vector>

int main() {
    lina::ControlModel model(
        /*wavelength_c=*/630e-9,
        /*wavelength=*/{},          // std::optional, defaults to wavelength_c
        /*npix=*/256,
        /*ndef=*/258,
        /*n_vortex_lres=*/2048,
        /*vortex_win_diam=*/30.0,
        /*vortex_hres_sampling=*/0.025,
        /*vortex_dot_mask_diam_lamDc=*/0.5,
        /*dm_beam_diam=*/9.3e-3,
        /*lyot_pupil_diam=*/9.1e-3,
        /*lyot_stop_diam=*/8.6e-3,
        /*exit_pupil_prop_dist=*/{},
        /*camsci_pxscl_lamDc=*/0.2,
        /*ncamsci=*/256,
        /*nact=*/34,
        /*act_spacing=*/300e-6,
        /*act_coupling=*/0.15);

    std::vector<double> actuators(model.nacts(), 0.0);
    auto e_fp = model.forward(actuators, /*wavelength=*/630e-9,
                              /*use_vortex=*/true);
    return 0;
}
```

## 6. Threading / runtime knobs

The lina runtime reads a few environment variables when `liblina` is loaded:

| Variable                  | Effect                                                    |
| ------------------------- | --------------------------------------------------------- |
| `LINA_FFTW_THREADS`       | `fftw_plan_with_nthreads(N)` (defaults to 1)             |
| `LINA_FFTW_PLAN`          | `MEASURE` to use `FFTW_MEASURE` instead of `FFTW_ESTIMATE` |
| `LINA_FFTW_WISDOM_DIR`    | Directory for cached FFTW wisdom files                   |
| `OPENBLAS_NUM_THREADS`    | Number of BLAS threads used for `gemm`/`gemv`/SVD        |

You typically set these in the same shell that runs your binary.

## 7. Mixing with Python

If you also want to drive lina from Python, the `cpp/pybind/lina_py.cpp`
module exposes the math primitives as a Python extension named `lina_cpp`.
Build it with `-DLINA_BUILD_PYBIND=ON`:

```
cmake -S cpp -B cpp/build \
      -DLINA_BUILD_PYBIND=ON \
      -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir)
cmake --build cpp/build
PYTHONPATH=$PWD/cpp/build python3 -c "import lina_cpp; print(dir(lina_cpp))"
```

This module is what `lina/tests/test_per_method_parity.py` depends on for
its direct, in-process parity tests.
