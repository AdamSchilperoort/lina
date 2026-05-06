#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

BUILD_DIR="${REPO_ROOT}/cpp/build-native"
IMAGESTREAMIO_ROOT="${IMAGESTREAMIO_ROOT:-/opt/MagAOX/source/milk/src/ImageStreamIO}"
CUDA_COMPILER="${CUDA_COMPILER:-/usr/local/cuda/bin/nvcc}"
CUDA_ARCH="${CUDA_ARCH:-86}"

PYBIND11_DIR=$(python3 -m pybind11 --cmakedir || true)
cmake -S "${REPO_ROOT}/cpp" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLINA_USE_CUDA=ON \
  -DLINA_FORCE_LAPACKE=ON \
  -DLINA_USE_EIGEN_SVD=OFF \
  -DLINA_BUILD_PYBIND=ON \
  ${PYBIND11_DIR:+-Dpybind11_DIR=${PYBIND11_DIR}} \
  -DCMAKE_CUDA_COMPILER="${CUDA_COMPILER}" \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
  -DIMAGESTREAMIO_ROOT="${IMAGESTREAMIO_ROOT}"

cmake --build "${BUILD_DIR}"
echo "LAPACK/BLAS linkage:"
ldd "${BUILD_DIR}/lina_runner" | grep -i -E "openblas|blas|lapack|lapacke" || true

export LINA_CPP_RUNNER="${BUILD_DIR}/lina_runner"
export LINA_RUN_BENCHMARKS=1
export LINA_RUN_LARGE_SVD=1
export LINA_BENCH_SVD_M=5000
export LINA_BENCH_SVD_N=2000
export LINA_BENCH_REPORT="${REPO_ROOT}/lina/tests/bench_report.md"
export PYTHONPATH="${BUILD_DIR}:${PYTHONPATH:-}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(nproc)}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$(nproc)}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$(nproc)}"
export LINA_FFTW_THREADS="${LINA_FFTW_THREADS:-$(nproc)}"
export LINA_FFTW_WISDOM_PATH="${LINA_FFTW_WISDOM_PATH:-${REPO_ROOT}/lina/tests/fftw_wisdom.dat}"
export LINA_FFTW_PLAN="${LINA_FFTW_PLAN:-MEASURE}"

python3 -m unittest lina.tests.test_cpp_parity lina.tests.test_more_python lina.tests.test_benchmarks lina.tests.test_pybind_bridge
