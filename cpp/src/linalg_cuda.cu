#include "lina/linalg.h"

#include <stdexcept>

#ifdef LINA_USE_CUDA

#include <cusolverDn.h>
#include <cuda_runtime.h>

namespace lina {

namespace {

void check_cuda(cudaError_t status, const char* msg) {
    if (status != cudaSuccess) {
        throw std::runtime_error(msg);
    }
}

void check_cusolver(cusolverStatus_t status, const char* msg) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        throw std::runtime_error(msg);
    }
}

} // namespace

SvdResultF svd_float_cuda(const Array2D<float>& a) {
    const std::size_t m = a.rows();
    const std::size_t n = a.cols();
    const std::size_t min_mn = std::min(m, n);

    SvdResultF result;
    result.u = Array2D<float>(m, m, 0.0f);
    result.vt = Array2D<float>(n, n, 0.0f);
    result.s = std::vector<float>(min_mn, 0.0f);

    float* d_a = nullptr;
    float* d_s = nullptr;
    float* d_u = nullptr;
    float* d_v = nullptr;
    int* d_info = nullptr;

    cusolverDnHandle_t handle = nullptr;
    check_cusolver(cusolverDnCreate(&handle), "cusolverDnCreate failed");

    const int m_i = static_cast<int>(m);
    const int n_i = static_cast<int>(n);
    const int lda = m_i;
    const int ldu = m_i;
    const int ldv = n_i;

    // Column-major copy
    std::vector<float> a_col(m * n, 0.0f);
    for (std::size_t r = 0; r < m; ++r) {
        for (std::size_t c = 0; c < n; ++c) {
            a_col[c * m + r] = a(r, c);
        }
    }

    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_a), sizeof(float) * a_col.size()),
               "cudaMalloc d_a failed");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_s), sizeof(float) * min_mn),
               "cudaMalloc d_s failed");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_u), sizeof(float) * m * m),
               "cudaMalloc d_u failed");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_v), sizeof(float) * n * n),
               "cudaMalloc d_v failed");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_info), sizeof(int)),
               "cudaMalloc d_info failed");

    check_cuda(cudaMemcpy(d_a, a_col.data(), sizeof(float) * a_col.size(), cudaMemcpyHostToDevice),
               "cudaMemcpy d_a failed");

    gesvdjInfo_t params = nullptr;
    check_cusolver(cusolverDnCreateGesvdjInfo(&params), "cusolverDnCreateGesvdjInfo failed");
    check_cusolver(cusolverDnXgesvdjSetTolerance(params, 1.0e-7f),
                   "cusolverDnXgesvdjSetTolerance failed");
    check_cusolver(cusolverDnXgesvdjSetMaxSweeps(params, 100),
                   "cusolverDnXgesvdjSetMaxSweeps failed");
    check_cusolver(cusolverDnXgesvdjSetSortEig(params, 1),
                   "cusolverDnXgesvdjSetSortEig failed");

    auto run_gesvdj = [&](int max_sweeps, float tol) {
        check_cusolver(cusolverDnXgesvdjSetTolerance(params, tol),
                       "cusolverDnXgesvdjSetTolerance failed");
        check_cusolver(cusolverDnXgesvdjSetMaxSweeps(params, max_sweeps),
                       "cusolverDnXgesvdjSetMaxSweeps failed");

        int lwork = 0;
        check_cusolver(cusolverDnSgesvdj_bufferSize(
                           handle, CUSOLVER_EIG_MODE_VECTOR, 0, m_i, n_i,
                           d_a, lda, d_s, d_u, ldu, d_v, ldv, &lwork, params),
                       "cusolverDnSgesvdj_bufferSize failed");

        float* d_work = nullptr;
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_work), sizeof(float) * lwork),
                   "cudaMalloc d_work failed");

        check_cusolver(cusolverDnSgesvdj(
                           handle, CUSOLVER_EIG_MODE_VECTOR, 0, m_i, n_i,
                           d_a, lda, d_s, d_u, ldu, d_v, ldv, d_work, lwork, d_info, params),
                       "cusolverDnSgesvdj failed");

        int info = 0;
        check_cuda(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost),
                   "cudaMemcpy info failed");
        cudaFree(d_work);
        return info;
    };

    int info = run_gesvdj(200, 1.0e-4f);
    if (info != 0) {
        check_cuda(cudaMemcpy(d_a, a_col.data(), sizeof(float) * a_col.size(), cudaMemcpyHostToDevice),
                   "cudaMemcpy d_a failed");
        info = run_gesvdj(5000, 1.0e-5f);
        if (info != 0) {
            throw std::runtime_error("cuSOLVER SVD (Jacobi) failed with info=" + std::to_string(info));
        }
    }

    std::vector<float> u_col(m * m, 0.0f);
    std::vector<float> v_col(n * n, 0.0f);
    check_cuda(cudaMemcpy(result.s.data(), d_s, sizeof(float) * min_mn,
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy s failed");
    check_cuda(cudaMemcpy(u_col.data(), d_u, sizeof(float) * m * m,
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy u failed");
    check_cuda(cudaMemcpy(v_col.data(), d_v, sizeof(float) * n * n,
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy v failed");

    for (std::size_t r = 0; r < m; ++r) {
        for (std::size_t c = 0; c < m; ++c) {
            result.u(r, c) = u_col[c * m + r];
        }
    }
    for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t c = 0; c < n; ++c) {
            result.vt(r, c) = v_col[r * n + c];
        }
    }

    cudaFree(d_a);
    cudaFree(d_s);
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_info);
    cusolverDnDestroyGesvdjInfo(params);
    cusolverDnDestroy(handle);

    return result;
}

} // namespace lina

#else

// CUDA not enabled; keep translation unit for build systems.
namespace lina {
SvdResultF svd_float_cuda(const Array2D<float>&) {
    throw std::runtime_error("CUDA SVD unavailable: build with LINA_USE_CUDA=ON");
}
} // namespace lina

#endif
