cmake_minimum_required(VERSION 3.16)
project(lina_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(LINA_USE_FFTW "Use FFTW3 for FFTs if available" ON)
option(LINA_USE_OPENBLAS "Use OpenBLAS/CBLAS for GEMM/GEMV" ON)
option(LINA_USE_LAPACKE "Use LAPACKE for SVD" ON)
# LINA_USE_CUDA: tri-state. If the user passes -DLINA_USE_CUDA=ON/OFF
# explicitly we honour it. Otherwise (the unset case), the auto-detect
# block below probes for a working CUDA toolkit and flips it ON when
# available. To force-disable auto-detect, pass -DLINA_USE_CUDA=OFF.
option(LINA_USE_CUDA_AUTO "Auto-detect CUDA toolkit when LINA_USE_CUDA is unset" ON)
# Track whether the user explicitly set LINA_USE_CUDA via -D. We have
# to check BEFORE option() populates a default, because option() makes
# the variable look "defined" even when the user never touched it.
set(_lina_cuda_user_explicit FALSE)
if(DEFINED CACHE{LINA_USE_CUDA})
    set(_lina_cuda_user_explicit TRUE)
endif()
option(LINA_USE_CUDA "Enable CUDA/cuSOLVER implementations" OFF)

option(LINA_USE_LBFGS "Use libLBFGS for optimization" ON)
option(LINA_USE_EIGEN_SVD "Use Eigen for SVD fallback" OFF)
option(LINA_FORCE_LAPACKE "Force LAPACKE SVD over Eigen" ON)
option(LINA_CUDA_ALLOW_UNSUPPORTED "Allow unsupported host compiler for nvcc" ON)
option(LINA_BUILD_PYBIND "Build pybind11 Python module" OFF)
option(LINA_USE_CFITSIO "Enable native FITS I/O via cfitsio" ON)

set(MAGAOX_VENDOR_ROOT "/opt/MagAOX/vendor" CACHE PATH "MagAOX vendor dependency root")
option(LINA_BUILD_SHARED "Build shared library" ON)
set(IMAGESTREAMIO_ROOT "/opt/MagAOX/source/milk/src/ImageStreamIO" CACHE PATH "Root of ImageStreamIO source/build")

# --- CUDA auto-detection -----------------------------------------------------
# If the user did not pass -DLINA_USE_CUDA=ON/OFF explicitly, probe for
# a working CUDA toolkit. This makes `pip install ./lina_cpp/` "just work"
# with GPU support on any machine that has CUDA installed, without
# requiring LINA_CMAKE_ARGS gymnastics. Users on CPU-only boxes get a
# clean CPU build; users with CUDA get the GPU backend automatically.
if(NOT _lina_cuda_user_explicit AND LINA_USE_CUDA_AUTO AND NOT LINA_USE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        find_package(CUDAToolkit QUIET)
        if(CUDAToolkit_FOUND)
            message(STATUS "lina: CUDA toolkit auto-detected "
                "(nvcc=${CMAKE_CUDA_COMPILER}, "
                "version=${CUDAToolkit_VERSION}); enabling GPU backend. "
                "Pass -DLINA_USE_CUDA=OFF to force a CPU-only build.")
            set(LINA_USE_CUDA ON CACHE BOOL
                "Enable CUDA/cuSOLVER implementations" FORCE)
        else()
            message(STATUS "lina: CUDA compiler found but CUDAToolkit "
                "package missing; building CPU-only. "
                "Set CUDAToolkit_ROOT or pass -DLINA_USE_CUDA=ON to force.")
        endif()
    else()
        message(STATUS "lina: no CUDA compiler found on PATH; "
            "building CPU-only. Install CUDA and re-run cmake, or pass "
            "-DLINA_USE_CUDA=ON to force a (likely failing) GPU build.")
    endif()
endif()

if(LINA_USE_CUDA)
    # NOTE on supported toolchains:
    #   * CUDA 12.4 + glibc <= 2.39 + gcc <= 13: builds cleanly out of the box.
    #     This is the standard Ubuntu 22.04 / RHEL 9 environment, and the
    #     environment that MagAOX targets.
    #   * CUDA 12.6 + glibc 2.40 (Debian 13) + gcc 14: also fine.
    #   * CUDA 12.4 + glibc 2.40 (Debian 13): nvcc fails on cospi/sinpi
    #     noexcept mismatches in the system math headers. Use CUDA 12.6+
    #     or downgrade glibc; see NVIDIA bug NVCC-1543. We leave the GPU
    #     code in place but expect the build to fail in this combination.
    if(LINA_CUDA_ALLOW_UNSUPPORTED)
        set(CMAKE_CUDA_FLAGS_INIT "${CMAKE_CUDA_FLAGS_INIT} -allow-unsupported-compiler" CACHE STRING "" FORCE)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
    endif()
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        # Defaulting to compute_86 = Ampere (A40, A100, RTX 30xx). Override
        # with -DCMAKE_CUDA_ARCHITECTURES="89;86" etc. for other GPUs.
        set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures" FORCE)
    endif()
    enable_language(CUDA)
endif()

set(LINA_SOURCES
    src/array.cpp
    src/utils.cpp
    src/fits_io.cpp
    src/props.cpp
    src/coro_utils.cpp
    src/linalg.cpp
    src/dm.cpp
    src/efc.cpp
    src/iefc.cpp
    src/aefc.cpp
    src/shmim_utils.cpp
    src/control_models.cpp
    src/pwp.cpp
    src/wfe.cpp
    src/llowfsc.cpp
)

add_library(lina_cpp STATIC ${LINA_SOURCES})
set_target_properties(lina_cpp PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(LINA_BUILD_SHARED)
    add_library(lina_cpp_shared SHARED ${LINA_SOURCES})
    set_target_properties(lina_cpp_shared PROPERTIES OUTPUT_NAME "lina")
endif()

set(LINA_TARGETS lina_cpp)
if(LINA_BUILD_SHARED)
    list(APPEND LINA_TARGETS lina_cpp_shared)
endif()

foreach(target IN LISTS LINA_TARGETS)
    target_include_directories(${target} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
endforeach()

add_executable(lina_runner tools/lina_runner.cpp)
target_link_libraries(lina_runner PRIVATE lina_cpp)

if(LINA_USE_FFTW)
    find_path(FFTW3_INCLUDE_DIR fftw3.h
              HINTS
              ${MAGAOX_VENDOR_ROOT}/fftw-3.3.8
              ${MAGAOX_VENDOR_ROOT}/fftw-3.3.8/include)
    find_library(FFTW3_LIBRARY fftw3
                 HINTS
                 ${MAGAOX_VENDOR_ROOT}/fftw-3.3.8
                 ${MAGAOX_VENDOR_ROOT}/fftw-3.3.8/lib)
    find_library(FFTW3_THREADS_LIBRARY fftw3_threads
                 HINTS
                 ${MAGAOX_VENDOR_ROOT}/fftw-3.3.8
                 ${MAGAOX_VENDOR_ROOT}/fftw-3.3.8/lib)
    if(FFTW3_INCLUDE_DIR AND FFTW3_LIBRARY)
        foreach(target IN LISTS LINA_TARGETS)
            target_compile_definitions(${target} PUBLIC LINA_USE_FFTW)
            target_include_directories(${target} PUBLIC ${FFTW3_INCLUDE_DIR})
            target_link_libraries(${target} PUBLIC ${FFTW3_LIBRARY})
            if(FFTW3_THREADS_LIBRARY)
                target_compile_definitions(${target} PUBLIC LINA_USE_FFTW_THREADS)
                target_link_libraries(${target} PUBLIC ${FFTW3_THREADS_LIBRARY})
            endif()
        endforeach()
    else()
        message(FATAL_ERROR "FFTW3 not found. Set MAGAOX_VENDOR_ROOT or install FFTW3.")
    endif()
endif()

if(LINA_USE_OPENBLAS)
    find_path(OPENBLAS_INCLUDE_DIR cblas.h
              HINTS
              ${MAGAOX_VENDOR_ROOT}/OpenBLAS
              ${MAGAOX_VENDOR_ROOT}/OpenBLAS/include)
    find_library(OPENBLAS_LIBRARY openblas
                 HINTS
                 ${MAGAOX_VENDOR_ROOT}/OpenBLAS
                 ${MAGAOX_VENDOR_ROOT}/OpenBLAS/lib)
    if(OPENBLAS_INCLUDE_DIR AND OPENBLAS_LIBRARY)
        foreach(target IN LISTS LINA_TARGETS)
            target_compile_definitions(${target} PUBLIC LINA_USE_OPENBLAS)
            target_include_directories(${target} PUBLIC ${OPENBLAS_INCLUDE_DIR})
            target_link_libraries(${target} PUBLIC ${OPENBLAS_LIBRARY})
        endforeach()
    else()
        message(FATAL_ERROR "OpenBLAS not found. Set MAGAOX_VENDOR_ROOT or install OpenBLAS.")
    endif()
endif()

if(LINA_USE_LAPACKE)
    find_path(LAPACKE_INCLUDE_DIR lapacke.h
              HINTS
              ${MAGAOX_VENDOR_ROOT}/OpenBLAS
              ${MAGAOX_VENDOR_ROOT}/OpenBLAS/include
              ${MAGAOX_VENDOR_ROOT}/OpenBLAS-0.3.24/lapack-netlib/LAPACKE/include
              ${MAGAOX_VENDOR_ROOT}/lapacke
              ${MAGAOX_VENDOR_ROOT}/lapacke/include)
    find_library(LAPACKE_LIBRARY lapacke
                 HINTS
                 ${MAGAOX_VENDOR_ROOT}/OpenBLAS
                 ${MAGAOX_VENDOR_ROOT}/OpenBLAS/lib
                 ${MAGAOX_VENDOR_ROOT}/lapacke
                 ${MAGAOX_VENDOR_ROOT}/lapacke/lib)
    if(NOT LAPACKE_INCLUDE_DIR)
        find_path(LAPACKE_INCLUDE_DIR lapacke.h)
    endif()
    if(NOT LAPACKE_LIBRARY)
        find_library(LAPACKE_LIBRARY lapacke)
    endif()
    if(LAPACKE_INCLUDE_DIR AND LAPACKE_LIBRARY)
        foreach(target IN LISTS LINA_TARGETS)
            target_compile_definitions(${target} PUBLIC LINA_USE_LAPACKE)
            target_include_directories(${target} PUBLIC ${LAPACKE_INCLUDE_DIR})
            target_link_libraries(${target} PUBLIC ${LAPACKE_LIBRARY})
        endforeach()
        if(LINA_FORCE_LAPACKE)
            foreach(target IN LISTS LINA_TARGETS)
                target_compile_definitions(${target} PUBLIC LINA_FORCE_LAPACKE)
            endforeach()
        endif()
    endif()
endif()

if(LINA_USE_EIGEN_SVD)
    find_path(EIGEN3_INCLUDE_DIR Eigen/Core
              HINTS
              ${MAGAOX_VENDOR_ROOT}/eigen-3.3.4
              ${MAGAOX_VENDOR_ROOT}/eigen-3.3.4/include)
    if(EIGEN3_INCLUDE_DIR)
        foreach(target IN LISTS LINA_TARGETS)
            target_compile_definitions(${target} PUBLIC LINA_USE_EIGEN_SVD)
            target_include_directories(${target} PUBLIC ${EIGEN3_INCLUDE_DIR})
        endforeach()
    endif()
endif()

if(LINA_USE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_library(lina_cuda src/linalg_cuda.cu src/props_cuda.cu)
    target_include_directories(lina_cuda PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        ${CUDAToolkit_INCLUDE_DIRS})
    target_compile_definitions(lina_cuda PUBLIC LINA_USE_CUDA)
    set_target_properties(lina_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(lina_cuda PUBLIC CUDA::cudart CUDA::cusolver CUDA::cufft CUDA::cublas)
    foreach(target IN LISTS LINA_TARGETS)
        target_link_libraries(${target} PUBLIC lina_cuda)
    endforeach()
endif()

if(LINA_USE_LBFGS)
    find_library(LBFGS_LIBRARY lbfgs)
    if(LBFGS_LIBRARY)
        foreach(target IN LISTS LINA_TARGETS)
            target_compile_definitions(${target} PUBLIC LINA_USE_LBFGS)
            target_link_libraries(${target} PUBLIC ${LBFGS_LIBRARY})
        endforeach()
    endif()
endif()

if(LINA_USE_CFITSIO)
    find_path(CFITSIO_INCLUDE_DIR fitsio.h
              HINTS
              ${MAGAOX_VENDOR_ROOT}/cfitsio-3.47
              ${MAGAOX_VENDOR_ROOT}/cfitsio-3.47/include
              /usr/include
              /usr/local/include)
    find_library(CFITSIO_LIBRARY cfitsio
                 HINTS
                 ${MAGAOX_VENDOR_ROOT}/cfitsio-3.47
                 ${MAGAOX_VENDOR_ROOT}/cfitsio-3.47/lib
                 /usr/lib
                 /usr/lib/x86_64-linux-gnu
                 /usr/local/lib)
    if(CFITSIO_INCLUDE_DIR AND CFITSIO_LIBRARY)
        message(STATUS "lina: cfitsio FOUND at ${CFITSIO_LIBRARY}")
        foreach(target IN LISTS LINA_TARGETS)
            target_compile_definitions(${target} PUBLIC LINA_USE_CFITSIO)
            target_include_directories(${target} PUBLIC ${CFITSIO_INCLUDE_DIR})
            target_link_libraries(${target} PUBLIC ${CFITSIO_LIBRARY})
        endforeach()
    else()
        message(WARNING "lina: cfitsio NOT found; FITS I/O will throw at runtime. "
                        "Pass -DLINA_USE_CFITSIO=OFF to silence, or install cfitsio.")
    endif()
endif()

find_path(IMAGESTREAMIO_INCLUDE_DIR ImageStreamIO.h
          HINTS ${IMAGESTREAMIO_ROOT} ${IMAGESTREAMIO_ROOT}/include)
find_library(IMAGESTREAMIO_LIBRARY ImageStreamIO
             HINTS
             ${IMAGESTREAMIO_ROOT}/build
             ${IMAGESTREAMIO_ROOT}/build/temp.linux-x86_64-cpython-310
             ${IMAGESTREAMIO_ROOT}/build/lib.linux-x86_64-cpython-310
             ${IMAGESTREAMIO_ROOT}/lib)

if(NOT IMAGESTREAMIO_INCLUDE_DIR OR NOT IMAGESTREAMIO_LIBRARY)
    message(FATAL_ERROR "ImageStreamIO not found. Set IMAGESTREAMIO_ROOT to the ImageStreamIO source/build root.")
endif()

foreach(target IN LISTS LINA_TARGETS)
    target_compile_definitions(${target} PUBLIC LINA_USE_IMAGESTREAMIO)
    target_include_directories(${target} PUBLIC ${IMAGESTREAMIO_INCLUDE_DIR})
    target_link_libraries(${target} PUBLIC ${IMAGESTREAMIO_LIBRARY})
endforeach()

install(TARGETS lina_cpp
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)

if(LINA_BUILD_SHARED)
    install(TARGETS lina_cpp_shared
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)
endif()

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION include)

install(TARGETS lina_runner RUNTIME DESTINATION bin)

if(LINA_USE_CUDA)
    install(TARGETS lina_cuda
            EXPORT linaTargets
            ARCHIVE DESTINATION lib
            LIBRARY DESTINATION lib
            RUNTIME DESTINATION bin)
endif()

install(TARGETS lina_cpp
        EXPORT linaTargets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)

if(LINA_BUILD_SHARED)
    install(TARGETS lina_cpp_shared
            EXPORT linaTargets
            ARCHIVE DESTINATION lib
            LIBRARY DESTINATION lib
            RUNTIME DESTINATION bin)
endif()

install(EXPORT linaTargets
        FILE linaTargets.cmake
        NAMESPACE lina::
        DESTINATION lib/cmake/lina)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/linaConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/linaConfig.cmake
    INSTALL_DESTINATION lib/cmake/lina)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/linaConfigVersion.cmake
    VERSION 0.1.0
    COMPATIBILITY SameMajorVersion)

install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/linaConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/linaConfigVersion.cmake
        DESTINATION lib/cmake/lina)

# --- Final configuration banner ---------------------------------------------
# Print a clear one-liner so users know what they're getting. The auto-detect
# block above resolves LINA_USE_CUDA before we land here.
message(STATUS "")
message(STATUS "=== lina_cpp configuration ===")
if(LINA_USE_CUDA)
    message(STATUS "  GPU backend : ENABLED  (cuFFT, cuBLAS, cuSOLVER, custom kernels)")
    message(STATUS "  CUDA arch   : ${CMAKE_CUDA_ARCHITECTURES}")
    message(STATUS "  nvcc        : ${CMAKE_CUDA_COMPILER}")
else()
    message(STATUS "  GPU backend : disabled (CPU-only build)")
    message(STATUS "                pass -DLINA_USE_CUDA=ON or set LINA_USE_CUDA=1")
    message(STATUS "                in the environment before pip install to enable.")
endif()
if(LINA_USE_FFTW AND FFTW3_LIBRARY)
    message(STATUS "  CPU FFT     : FFTW3 (${FFTW3_LIBRARY})")
else()
    message(STATUS "  CPU FFT     : naive")
endif()
message(STATUS "  CPU LinAlg  : OpenBLAS / LAPACKE")
if(CFITSIO_LIBRARY)
    message(STATUS "  FITS I/O    : cfitsio (${CFITSIO_LIBRARY})")
else()
    message(STATUS "  FITS I/O    : disabled (no cfitsio)")
endif()
message(STATUS "==============================")
message(STATUS "")

if(LINA_BUILD_PYBIND)
    # The Python extension's OUTPUT_NAME is configurable so the same source
    # can be built either standalone (parity tests do `import lina_cpp` against
    # this .so) or as part of the lina_cpp Python package (where it lives at
    # `lina_cpp/_core.so` and the package __init__.py re-exports from it).
    set(LINA_PYBIND_MODULE_NAME "lina_cpp" CACHE STRING
        "Python extension module name (passed to PYBIND11_MODULE)")
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(lina_py MODULE pybind/lina_py.cpp)
    target_link_libraries(lina_py PRIVATE lina_cpp)
    if(LINA_USE_CUDA)
        target_link_libraries(lina_py PRIVATE lina_cuda)
    endif()
    target_include_directories(lina_py PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_definitions(lina_py PRIVATE
        LINA_PYBIND_MODULE_NAME=${LINA_PYBIND_MODULE_NAME})
    set_target_properties(lina_py PROPERTIES OUTPUT_NAME "${LINA_PYBIND_MODULE_NAME}")
endif()
