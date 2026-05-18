cmake_minimum_required(VERSION 3.16)
project(lina_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(LINA_USE_FFTW "Use FFTW3 for FFTs if available" ON)
option(LINA_USE_OPENBLAS "Use OpenBLAS/CBLAS for GEMM/GEMV" ON)
option(LINA_USE_LAPACKE "Use LAPACKE for SVD" ON)
option(LINA_USE_CUDA "Enable CUDA/cuSOLVER implementations" OFF)
option(LINA_USE_LBFGS "Use libLBFGS for optimization" ON)
option(LINA_USE_EIGEN_SVD "Use Eigen for SVD fallback" OFF)
option(LINA_FORCE_LAPACKE "Force LAPACKE SVD over Eigen" ON)
option(LINA_CUDA_ALLOW_UNSUPPORTED "Allow unsupported host compiler for nvcc" ON)
option(LINA_BUILD_PYBIND "Build pybind11 Python module" OFF)
option(LINA_USE_CFITSIO "Enable native FITS I/O via cfitsio" ON)

set(MAGAOX_VENDOR_ROOT "/opt/MagAOX/vendor" CACHE PATH "MagAOX vendor dependency root")
set(IMAGESTREAMIO_ROOT "/opt/MagAOX/source/milk/src/ImageStreamIO" CACHE PATH "Root of ImageStreamIO source/build/install tree")

option(LINA_BUILD_SHARED "Build shared library" ON)

if(LINA_USE_CUDA)
    if(LINA_CUDA_ALLOW_UNSUPPORTED)
        set(CMAKE_CUDA_FLAGS_INIT "${CMAKE_CUDA_FLAGS_INIT} -allow-unsupported-compiler" CACHE STRING "" FORCE)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
    endif()

    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
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
        $<INSTALL_INTERFACE:include>
    )
endforeach()

add_executable(lina_runner tools/lina_runner.cpp)
target_link_libraries(lina_runner PRIVATE lina_cpp)

if(LINA_USE_FFTW)
    find_path(FFTW3_INCLUDE_DIR fftw3.h
        HINTS
            ${MAGAOX_VENDOR_ROOT}/fftw-3.3.8
            ${MAGAOX_VENDOR_ROOT}/fftw-3.3.8/include
    )

    find_library(FFTW3_LIBRARY fftw3
        HINTS
            ${MAGAOX_VENDOR_ROOT}/fftw-3.3.8
            ${MAGAOX_VENDOR_ROOT}/fftw-3.3.8/lib
    )

    find_library(FFTW3_THREADS_LIBRARY fftw3_threads
        HINTS
            ${MAGAOX_VENDOR_ROOT}/fftw-3.3.8
            ${MAGAOX_VENDOR_ROOT}/fftw-3.3.8/lib
    )

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
            ${MAGAOX_VENDOR_ROOT}/OpenBLAS/include
            /usr/include
            /usr/local/include
    )

    find_library(OPENBLAS_LIBRARY openblas
        HINTS
            ${MAGAOX_VENDOR_ROOT}/OpenBLAS
            ${MAGAOX_VENDOR_ROOT}/OpenBLAS/lib
            /usr/lib
            /usr/lib64
            /usr/lib/x86_64-linux-gnu
            /usr/local/lib
            /usr/local/lib64
    )

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
            ${MAGAOX_VENDOR_ROOT}/lapacke/include
            /usr/include
            /usr/local/include
    )

    find_library(LAPACKE_LIBRARY lapacke
        HINTS
            ${MAGAOX_VENDOR_ROOT}/OpenBLAS
            ${MAGAOX_VENDOR_ROOT}/OpenBLAS/lib
            ${MAGAOX_VENDOR_ROOT}/lapacke
            ${MAGAOX_VENDOR_ROOT}/lapacke/lib
            /usr/lib
            /usr/lib64
            /usr/lib/x86_64-linux-gnu
            /usr/local/lib
            /usr/local/lib64
    )

    if(LAPACKE_INCLUDE_DIR AND LAPACKE_LIBRARY)
        foreach(target IN LISTS LINA_TARGETS)
            target_compile_definitions(${target} PUBLIC LINA_USE_LAPACKE)
            target_include_directories(${target} PUBLIC ${LAPACKE_INCLUDE_DIR})
            target_link_libraries(${target} PUBLIC ${LAPACKE_LIBRARY})

            if(LINA_FORCE_LAPACKE)
                target_compile_definitions(${target} PUBLIC LINA_FORCE_LAPACKE)
            endif()
        endforeach()
    else()
        message(WARNING "LAPACKE not found. SVD support may require Eigen fallback or system LAPACKE.")
    endif()
endif()

if(LINA_USE_EIGEN_SVD)
    find_path(EIGEN3_INCLUDE_DIR Eigen/Core
        HINTS
            ${MAGAOX_VENDOR_ROOT}/eigen-3.3.4
            ${MAGAOX_VENDOR_ROOT}/eigen-3.3.4/include
            /usr/include/eigen3
            /usr/local/include/eigen3
    )

    if(EIGEN3_INCLUDE_DIR)
        foreach(target IN LISTS LINA_TARGETS)
            target_compile_definitions(${target} PUBLIC LINA_USE_EIGEN_SVD)
            target_include_directories(${target} PUBLIC ${EIGEN3_INCLUDE_DIR})
        endforeach()
    endif()
endif()

if(LINA_USE_CUDA)
    find_package(CUDAToolkit REQUIRED)

    add_library(lina_cuda
        src/linalg_cuda.cu
        src/props_cuda.cu
    )

    target_include_directories(lina_cuda PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        ${CUDAToolkit_INCLUDE_DIRS}
    )

    target_compile_definitions(lina_cuda PUBLIC LINA_USE_CUDA)
    set_target_properties(lina_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(lina_cuda PUBLIC CUDA::cudart CUDA::cusolver CUDA::cufft CUDA::cublas)

    foreach(target IN LISTS LINA_TARGETS)
        target_compile_definitions(${target} PUBLIC LINA_USE_CUDA)
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
            /usr/local/include
    )

    find_library(CFITSIO_LIBRARY cfitsio
        HINTS
            ${MAGAOX_VENDOR_ROOT}/cfitsio-3.47
            ${MAGAOX_VENDOR_ROOT}/cfitsio-3.47/lib
            /usr/lib
            /usr/lib64
            /usr/lib/x86_64-linux-gnu
            /usr/local/lib
            /usr/local/lib64
    )

    if(CFITSIO_INCLUDE_DIR AND CFITSIO_LIBRARY)
        message(STATUS "lina: cfitsio FOUND at ${CFITSIO_LIBRARY}")

        foreach(target IN LISTS LINA_TARGETS)
            target_compile_definitions(${target} PUBLIC LINA_USE_CFITSIO)
            target_include_directories(${target} PUBLIC ${CFITSIO_INCLUDE_DIR})
            target_link_libraries(${target} PUBLIC ${CFITSIO_LIBRARY})
        endforeach()
    else()
        message(WARNING "lina: cfitsio NOT found; FITS I/O will throw at runtime. Pass -DLINA_USE_CFITSIO=OFF to silence, or install cfitsio.")
    endif()
endif()

find_path(IMAGESTREAMIO_INCLUDE_DIR ImageStreamIO.h
    HINTS
        ${IMAGESTREAMIO_ROOT}
        ${IMAGESTREAMIO_ROOT}/include
        ${IMAGESTREAMIO_ROOT}/include/ImageStreamIO
        /usr/local/milk-1.03.00/include
        /usr/local/milk-1.03.00/include/ImageStreamIO
)

find_path(IMAGESTRUCT_INCLUDE_DIR ImageStruct.h
    HINTS
        ${IMAGESTREAMIO_ROOT}
        ${IMAGESTREAMIO_ROOT}/..
        ${IMAGESTREAMIO_ROOT}/../ImageStruct
        ${IMAGESTREAMIO_ROOT}/include
        ${IMAGESTREAMIO_ROOT}/include/ImageStreamIO
        /opt/MagAOX/source/milk/src
        /opt/MagAOX/source/milk/src/ImageStreamIO
        /usr/local/milk-1.03.00/include
        /usr/local/milk-1.03.00/include/ImageStreamIO
)

find_library(IMAGESTREAMIO_LIBRARY ImageStreamIO
    HINTS
        ${IMAGESTREAMIO_ROOT}
        ${IMAGESTREAMIO_ROOT}/build
        ${IMAGESTREAMIO_ROOT}/build/temp.linux-x86_64-cpython-310
        ${IMAGESTREAMIO_ROOT}/build/lib.linux-x86_64-cpython-310
        ${IMAGESTREAMIO_ROOT}/lib
        ${IMAGESTREAMIO_ROOT}/lib64
        /usr/local/milk-1.03.00/lib
        /usr/local/milk-1.03.00/lib64
)

message(STATUS "ImageStreamIO root: ${IMAGESTREAMIO_ROOT}")
message(STATUS "ImageStreamIO include dir: ${IMAGESTREAMIO_INCLUDE_DIR}")
message(STATUS "ImageStruct include dir: ${IMAGESTRUCT_INCLUDE_DIR}")
message(STATUS "ImageStreamIO library: ${IMAGESTREAMIO_LIBRARY}")

if(NOT IMAGESTREAMIO_INCLUDE_DIR OR NOT IMAGESTREAMIO_LIBRARY)
    message(FATAL_ERROR "ImageStreamIO not found. Set IMAGESTREAMIO_ROOT to the ImageStreamIO source/build/install root.")
endif()

if(NOT IMAGESTRUCT_INCLUDE_DIR)
    message(FATAL_ERROR "ImageStruct.h not found. Set IMAGESTREAMIO_ROOT to the ImageStreamIO source/build/install root.")
endif()

foreach(target IN LISTS LINA_TARGETS)
    target_compile_definitions(${target} PUBLIC LINA_USE_IMAGESTREAMIO)
    target_include_directories(${target} PUBLIC
        ${IMAGESTREAMIO_INCLUDE_DIR}
        ${IMAGESTRUCT_INCLUDE_DIR}
    )
    target_link_libraries(${target} PUBLIC ${IMAGESTREAMIO_LIBRARY})
endforeach()

install(TARGETS lina_cpp
    EXPORT linaTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

if(LINA_BUILD_SHARED)
    install(TARGETS lina_cpp_shared
        EXPORT linaTargets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
    )
endif()

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION include)
install(TARGETS lina_runner RUNTIME DESTINATION bin)

if(LINA_USE_CUDA)
    install(TARGETS lina_cuda
        EXPORT linaTargets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
    )
endif()

install(EXPORT linaTargets
    FILE linaTargets.cmake
    NAMESPACE lina::
    DESTINATION lib/cmake/lina
)

include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/linaConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/linaConfig.cmake
    INSTALL_DESTINATION lib/cmake/lina
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/linaConfigVersion.cmake
    VERSION 0.1.0
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/linaConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/linaConfigVersion.cmake
    DESTINATION lib/cmake/lina
)

if(LINA_BUILD_PYBIND)
    set(LINA_PYBIND_MODULE_NAME "lina_cpp" CACHE STRING "Python extension module name passed to PYBIND11_MODULE")

    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(lina_py MODULE pybind/lina_py.cpp)
    target_link_libraries(lina_py PRIVATE lina_cpp)

    if(LINA_USE_CUDA)
        target_link_libraries(lina_py PRIVATE lina_cuda)
    endif()

    target_include_directories(lina_py PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${IMAGESTREAMIO_INCLUDE_DIR}
        ${IMAGESTRUCT_INCLUDE_DIR}
    )

    target_compile_definitions(lina_py PRIVATE
        LINA_PYBIND_MODULE_NAME=${LINA_PYBIND_MODULE_NAME}
    )

    set_target_properties(lina_py PROPERTIES OUTPUT_NAME "${LINA_PYBIND_MODULE_NAME}")
endif()
