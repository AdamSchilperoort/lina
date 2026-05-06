#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

CUDA_IMAGE="nvidia/cuda:12.4.1-devel-ubuntu22.04"
IMAGESTREAMIO_HOST="/opt/MagAOX/source/milk/src/ImageStreamIO"

docker run --gpus all -it --rm \
  -v "${REPO_ROOT}:/work" \
  -v "${IMAGESTREAMIO_HOST}:${IMAGESTREAMIO_HOST}" \
  "${CUDA_IMAGE}" \
  bash -lc '
    set -euo pipefail
    apt-get update
    apt-get install -y cmake g++ gcc make python3 python3-pip \
                       libfftw3-dev libopenblas-dev liblapacke-dev
    python3 -m pip install --no-cache-dir numpy scipy astropy poppy ipython scikit-image pybind11 cupy-cuda12x
    PYBIND11_DIR=$(python3 -m pybind11 --cmakedir)
    cmake -S /work/cpp -B /work/cpp/build-cuda \
      -DCMAKE_BUILD_TYPE=Release \
      -DLINA_USE_CUDA=ON \
      -DLINA_FORCE_LAPACKE=ON \
      -DLINA_USE_EIGEN_SVD=OFF \
      -DLINA_BUILD_PYBIND=ON \
      -Dpybind11_DIR="${PYBIND11_DIR}" \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
      -DCMAKE_CUDA_ARCHITECTURES=86 \
      -DIMAGESTREAMIO_ROOT='"${IMAGESTREAMIO_HOST}"'
    cmake --build /work/cpp/build-cuda
    echo "LAPACK/BLAS linkage:"
    ldd /work/cpp/build-cuda/lina_runner | grep -i -E "openblas|blas|lapack|lapacke" || true
    cd /work
    export LINA_CPP_RUNNER=/work/cpp/build-cuda/lina_runner
    export LINA_RUN_BENCHMARKS=1
    export LINA_RUN_LARGE_SVD=1
    export LINA_BENCH_SVD_M=5000
    export LINA_BENCH_SVD_N=2000
    export LINA_BENCH_REPORT=/work/lina/tests/bench_report.md
    export PYTHONPATH=/work/cpp/build-cuda:${PYTHONPATH:-}
    export OMP_NUM_THREADS=$(nproc)
    export OPENBLAS_NUM_THREADS=$(nproc)
    export MKL_NUM_THREADS=$(nproc)
    python3 -m unittest lina.tests.test_cpp_parity lina.tests.test_more_python lina.tests.test_benchmarks lina.tests.test_pybind_bridge
  '
