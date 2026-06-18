#include "lina/linalg.h"

#include <stdexcept>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <cstdlib>
#include <string>
#include <fstream>

#ifdef LINA_USE_CUDA

#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <algorithm>

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

void check_cublas(cublasStatus_t status, const char* msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(msg);
    }
}

cublasHandle_t blas_handle() {
    static cublasHandle_t h = [] {
        cublasHandle_t tmp = nullptr;
        check_cublas(cublasCreate(&tmp), "cublasCreate failed");
        return tmp;
    }();
    return h;
}

// Gather the diagonal of a column-major (n x n) matrix into a contiguous buffer.
__global__ void gather_diag_kernel(const double* A, double* d, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = A[static_cast<long long>(i) * (n + 1)];
}

// Add a scalar to the diagonal of a column-major (n x n) matrix in place.
__global__ void add_diag_kernel(double* A, int n, double val) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) A[static_cast<long long>(i) * (n + 1)] += val;
}

struct SvdKey {
    std::size_t m;
    std::size_t n;
    bool operator==(const SvdKey& o) const noexcept { return m == o.m && n == o.n; }
};

struct SvdKeyHash {
    std::size_t operator()(const SvdKey& k) const noexcept {
        return std::hash<std::size_t>{}(k.m) ^ (std::hash<std::size_t>{}(k.n) << 1);
    }
};

cusolverDnHandle_t svd_handle() {
    static cusolverDnHandle_t h = [] {
        cusolverDnHandle_t tmp = nullptr;
        check_cusolver(cusolverDnCreate(&tmp), "cusolverDnCreate failed");
        return tmp;
    }();
    return h;
}

struct SvdWorkspace {
    int m_i = 0;
    int n_i = 0;
    int lda = 0;
    int ldu = 0;
    int ldv = 0;
    std::size_t min_mn = 0;

    float* d_a_rm = nullptr;
    float* d_a = nullptr;
    float* d_work_gesvd = nullptr;
    int lwork_gesvd = 0;
    float* d_s = nullptr;
    float* d_u = nullptr;
    float* d_v = nullptr;
    int* d_info = nullptr;
    float* d_work = nullptr;
    int lwork = 0;
    gesvdjInfo_t params = nullptr;
    std::mutex run_mu;

    SvdWorkspace(std::size_t m, std::size_t n)
        : m_i(static_cast<int>(m)),
          n_i(static_cast<int>(n)),
          lda(m_i),
          ldu(m_i),
          ldv(n_i),
          min_mn(std::min(m, n)) {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_a_rm), sizeof(float) * m * n),
                   "cudaMalloc d_a_rm failed");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_a), sizeof(float) * m * n),
                   "cudaMalloc d_a failed");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_s), sizeof(float) * min_mn),
                   "cudaMalloc d_s failed");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_u), sizeof(float) * m * m),
                   "cudaMalloc d_u failed");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_v), sizeof(float) * n * n),
                   "cudaMalloc d_v failed");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_info), sizeof(int)),
                   "cudaMalloc d_info failed");

        check_cusolver(cusolverDnCreateGesvdjInfo(&params), "cusolverDnCreateGesvdjInfo failed");
        check_cusolver(cusolverDnXgesvdjSetTolerance(params, 1.0e-7f),
                       "cusolverDnXgesvdjSetTolerance failed");
        check_cusolver(cusolverDnXgesvdjSetMaxSweeps(params, 100),
                       "cusolverDnXgesvdjSetMaxSweeps failed");
        check_cusolver(cusolverDnXgesvdjSetSortEig(params, 1),
                       "cusolverDnXgesvdjSetSortEig failed");

        check_cusolver(
            cusolverDnSgesvdj_bufferSize(
                svd_handle(), CUSOLVER_EIG_MODE_VECTOR, 0, m_i, n_i,
                d_a, lda, d_s, d_u, ldu, d_v, ldv, &lwork, params),
            "cusolverDnSgesvdj_bufferSize failed");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_work), sizeof(float) * lwork),
                   "cudaMalloc d_work failed");

        check_cusolver(cusolverDnSgesvd_bufferSize(svd_handle(), m_i, n_i, &lwork_gesvd),
                       "cusolverDnSgesvd_bufferSize failed");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_work_gesvd), sizeof(float) * lwork_gesvd),
                   "cudaMalloc d_work_gesvd failed");
    }

    ~SvdWorkspace() {
        if (d_work_gesvd) cudaFree(d_work_gesvd);
        if (d_work) cudaFree(d_work);
        if (d_info) cudaFree(d_info);
        if (d_v) cudaFree(d_v);
        if (d_u) cudaFree(d_u);
        if (d_s) cudaFree(d_s);
        if (d_a) cudaFree(d_a);
        if (d_a_rm) cudaFree(d_a_rm);
        if (params) cusolverDnDestroyGesvdjInfo(params);
    }
};

class SvdWorkspaceCache {
public:
    std::shared_ptr<SvdWorkspace> get(std::size_t m, std::size_t n) {
        std::lock_guard<std::mutex> lock(mu_);
        const SvdKey key{m, n};
        auto it = pool_.find(key);
        if (it != pool_.end()) return it->second;
        if (pool_.size() >= kCap) pool_.erase(pool_.begin());
        auto ws = std::make_shared<SvdWorkspace>(m, n);
        pool_.emplace(key, ws);
        return ws;
    }

private:
    static constexpr std::size_t kCap = 8;
    std::unordered_map<SvdKey, std::shared_ptr<SvdWorkspace>, SvdKeyHash> pool_;
    std::mutex mu_;
};

SvdWorkspaceCache& svd_workspace_cache() {
    static SvdWorkspaceCache c;
    return c;
}

__global__ void transpose_rm_to_cm_kernel(const float* in_rm, float* out_cm, int rows, int cols) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r < rows && c < cols) {
        out_cm[c * rows + r] = in_rm[r * cols + c];
    }
}

int active_sm() {
    int dev = 0;
    cudaDeviceProp prop{};
    if (cudaGetDevice(&dev) == cudaSuccess &&
        cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
        return prop.major * 10 + prop.minor;
    }
    return 0;
}

std::size_t arch_default_threshold_elems(int sm) {
    if (sm >= 90) return static_cast<std::size_t>(6'000'000);
    if (sm >= 87) return static_cast<std::size_t>(20'000'000);
    if (sm >= 80) return static_cast<std::size_t>(12'000'000);
    if (sm > 0) return static_cast<std::size_t>(8'000'000);
    return static_cast<std::size_t>(12'000'000);
}

std::string autocal_cache_path(int sm) {
    return "/tmp/lina_cuda_svd_threshold_sm" + std::to_string(sm) + ".txt";
}

bool load_cached_threshold(int sm, std::size_t* out) {
    if (!out || sm <= 0) return false;
    std::ifstream in(autocal_cache_path(sm));
    if (!in.good()) return false;
    std::size_t v = 0;
    in >> v;
    if (!in.good() || v == 0) return false;
    *out = v;
    return true;
}

void save_cached_threshold(int sm, std::size_t threshold) {
    if (sm <= 0 || threshold == 0) return;
    std::ofstream out(autocal_cache_path(sm), std::ios::trunc);
    if (!out.good()) return;
    out << threshold << "\n";
}

SvdCalibrationResult run_svd_autocalibration();

std::size_t gesvd_threshold_elems() {
    static std::size_t v = [] {
        const char* e = std::getenv("LINA_CUDA_SVD_GESVD_THRESHOLD");
        if (e && *e) {
            return static_cast<std::size_t>(std::strtoull(e, nullptr, 10));
        }
        const char* autocal = std::getenv("LINA_CUDA_SVD_AUTOCALIBRATE");
        if (autocal && std::string(autocal) == "1") {
            const int sm = active_sm();
            std::size_t cached = 0;
            if (load_cached_threshold(sm, &cached)) {
                return cached;
            }
            const auto cal = run_svd_autocalibration();
            save_cached_threshold(sm, cal.threshold_elems);
            return cal.threshold_elems;
        }
        return arch_default_threshold_elems(active_sm());
    }();
    return v;
}

void prepare_col_major_input(SvdWorkspace& ws, const float* host_row_major, std::size_t count) {
    check_cuda(cudaMemcpy(ws.d_a_rm, host_row_major, sizeof(float) * count, cudaMemcpyHostToDevice),
               "cudaMemcpy d_a_rm failed");
    const dim3 block(16, 16);
    const dim3 grid((static_cast<unsigned>(ws.n_i) + block.x - 1) / block.x,
                    (static_cast<unsigned>(ws.m_i) + block.y - 1) / block.y);
    transpose_rm_to_cm_kernel<<<grid, block>>>(ws.d_a_rm, ws.d_a, ws.m_i, ws.n_i);
    check_cuda(cudaGetLastError(), "transpose_rm_to_cm_kernel failed");
}

Array2D<float> transpose_host(const Array2D<float>& in) {
    Array2D<float> out(in.cols(), in.rows(), 0.0f);
    for (std::size_t r = 0; r < in.rows(); ++r) {
        for (std::size_t c = 0; c < in.cols(); ++c) {
            out(c, r) = in(r, c);
        }
    }
    return out;
}

double benchmark_solver_kernel_ms(SvdWorkspace& ws,
                                  const Array2D<float>& a,
                                  bool use_gesvd,
                                  int iters) {
    if (iters < 1) iters = 1;
    std::lock_guard<std::mutex> lock(ws.run_mu);
    prepare_col_major_input(ws, a.data(), a.size());
    float* d_a0 = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_a0), sizeof(float) * a.size()), "cudaMalloc d_a0 failed");
    check_cuda(cudaMemcpy(d_a0, ws.d_a, sizeof(float) * a.size(), cudaMemcpyDeviceToDevice), "cudaMemcpy d_a0 failed");

    auto run_once = [&]() {
        check_cuda(cudaMemcpy(ws.d_a, d_a0, sizeof(float) * a.size(), cudaMemcpyDeviceToDevice),
                   "cudaMemcpy d_a reset failed");
        if (use_gesvd) {
            check_cusolver(cusolverDnSgesvd(
                svd_handle(), 'A', 'A', ws.m_i, ws.n_i, ws.d_a, ws.lda, ws.d_s,
                ws.d_u, ws.ldu, ws.d_v, ws.ldv, ws.d_work_gesvd, ws.lwork_gesvd,
                nullptr, ws.d_info), "cusolverDnSgesvd failed");
        } else {
            check_cusolver(cusolverDnSgesvdj(
                svd_handle(), CUSOLVER_EIG_MODE_VECTOR, 0, ws.m_i, ws.n_i,
                ws.d_a, ws.lda, ws.d_s, ws.d_u, ws.ldu, ws.d_v, ws.ldv,
                ws.d_work, ws.lwork, ws.d_info, ws.params), "cusolverDnSgesvdj failed");
        }
    };

    // One warmup to reduce one-time initialization noise.
    run_once();
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize warmup failed");

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0); cudaEventCreate(&ev1);
    cudaEventRecord(ev0);
    for (int i = 0; i < iters; ++i) {
        run_once();
    }
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, ev0, ev1);
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    cudaFree(d_a0);
    return static_cast<double>(ms) / static_cast<double>(iters);
}

SvdCalibrationResult run_svd_autocalibration() {
    SvdCalibrationResult out{};
    out.sm = active_sm();
    const std::size_t arch_default = arch_default_threshold_elems(out.sm);
    const std::vector<std::pair<std::size_t, std::size_t>> candidates = {
        {800, 400},
        {1200, 600},
        {2000, 1000},
        {3000, 1500},
        {5000, 2000},
    };

    out.points.reserve(candidates.size());
    for (const auto& [m, n] : candidates) {
        Array2D<float> a(m, n, 0.0f);
        for (std::size_t i = 0; i < a.size(); ++i) {
            a.data()[i] = static_cast<float>((i % 1024) + 1) * (1.0f / 1024.0f);
        }
        auto ws = svd_workspace_cache().get(m, n);
        const double gesvd_ms = benchmark_solver_kernel_ms(*ws, a, /*use_gesvd=*/true, 1);
        const double gesvdj_ms = benchmark_solver_kernel_ms(*ws, a, /*use_gesvd=*/false, 1);
        out.points.push_back({m, n, gesvd_ms, gesvdj_ms});
    }

    constexpr double kWinFactor = 0.97;  // require ~3% win to switch.
    std::size_t threshold = arch_default;
    bool found = false;
    for (const auto& p : out.points) {
        if (p.gesvd_ms <= p.gesvdj_ms * kWinFactor) {
            threshold = p.m * p.n;
            found = true;
            break;
        }
    }
    if (!found && !out.points.empty()) {
        const auto& last = out.points.back();
        threshold = std::max(arch_default, (last.m * last.n) + 1);
    }
    out.threshold_elems = threshold;
    return out;
}

} // namespace

SvdResultF svd_float_cuda(const Array2D<float>& a) {
    const std::size_t m = a.rows();
    const std::size_t n = a.cols();
    const std::size_t min_mn = std::min(m, n);

    // cuSOLVER gesvd is most reliable in the m>=n orientation. For wide
    // matrices (m<n), solve SVD(A^T) and map back:
    //   A^T = U_t S V_t^T  =>  A = V_t S U_t^T
    // so U(A)=V_t and VT(A)=U_t^T.
    if (m < n) {
        const auto at = transpose_host(a);  // (n, m), now n>=m
        const auto rt = svd_float_cuda(at);
        SvdResultF result;
        result.u = Array2D<float>(m, m, 0.0f);
        result.vt = Array2D<float>(n, n, 0.0f);
        result.s = rt.s;
        for (std::size_t r = 0; r < m; ++r) {
            for (std::size_t c = 0; c < m; ++c) {
                result.u(r, c) = rt.vt(c, r);
            }
        }
        for (std::size_t r = 0; r < n; ++r) {
            for (std::size_t c = 0; c < n; ++c) {
                result.vt(r, c) = rt.u(c, r);
            }
        }
        return result;
    }

    SvdResultF result;
    result.u = Array2D<float>(m, m, 0.0f);
    result.vt = Array2D<float>(n, n, 0.0f);
    result.s = std::vector<float>(min_mn, 0.0f);

    auto ws = svd_workspace_cache().get(m, n);
    std::lock_guard<std::mutex> lock(ws->run_mu);

    prepare_col_major_input(*ws, a.data(), a.size());

    auto run_gesvdj = [&](int max_sweeps, float tol) {
        check_cusolver(cusolverDnXgesvdjSetTolerance(ws->params, tol),
                       "cusolverDnXgesvdjSetTolerance failed");
        check_cusolver(cusolverDnXgesvdjSetMaxSweeps(ws->params, max_sweeps),
                       "cusolverDnXgesvdjSetMaxSweeps failed");

        check_cusolver(cusolverDnSgesvdj(
                           svd_handle(), CUSOLVER_EIG_MODE_VECTOR, 0, ws->m_i, ws->n_i,
                           ws->d_a, ws->lda, ws->d_s, ws->d_u, ws->ldu, ws->d_v, ws->ldv,
                           ws->d_work, ws->lwork, ws->d_info, ws->params),
                       "cusolverDnSgesvdj failed");

        int info = 0;
        check_cuda(cudaMemcpy(&info, ws->d_info, sizeof(int), cudaMemcpyDeviceToHost),
                   "cudaMemcpy info failed");
        return info;
    };

    const bool use_gesvd = (m * n) >= gesvd_threshold_elems();
    int info = 0;
    bool returns_vt_direct = false;
    if (use_gesvd) {
        returns_vt_direct = true;
        check_cusolver(cusolverDnSgesvd(
                           svd_handle(), 'A', 'A',
                           ws->m_i, ws->n_i,
                           ws->d_a, ws->lda,
                           ws->d_s,
                           ws->d_u, ws->ldu,
                           ws->d_v, ws->ldv,
                           ws->d_work_gesvd, ws->lwork_gesvd,
                           nullptr, ws->d_info),
                       "cusolverDnSgesvd failed");
        check_cuda(cudaMemcpy(&info, ws->d_info, sizeof(int), cudaMemcpyDeviceToHost),
                   "cudaMemcpy info failed");
        if (info != 0) {
            // Fallback path: if gesvd fails to converge/execute on this
            // shape+GPU combo, retry with Jacobi instead of hard-failing.
            prepare_col_major_input(*ws, a.data(), a.size());
            info = run_gesvdj(5000, 1.0e-5f);
            returns_vt_direct = false;
            if (info != 0) {
                throw std::runtime_error("cuSOLVER SVD failed (GESVD and GESVDJ) with info=" + std::to_string(info));
            }
        }
    } else {
        info = run_gesvdj(200, 1.0e-4f);
        if (info != 0) {
            prepare_col_major_input(*ws, a.data(), a.size());
            info = run_gesvdj(5000, 1.0e-5f);
            if (info != 0) {
                throw std::runtime_error("cuSOLVER SVD (Jacobi) failed with info=" + std::to_string(info));
            }
        }
    }

    std::vector<float> u_col(m * m, 0.0f);
    std::vector<float> v_col(n * n, 0.0f);
    check_cuda(cudaMemcpy(result.s.data(), ws->d_s, sizeof(float) * min_mn,
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy s failed");
    check_cuda(cudaMemcpy(u_col.data(), ws->d_u, sizeof(float) * m * m,
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy u failed");
    check_cuda(cudaMemcpy(v_col.data(), ws->d_v, sizeof(float) * n * n,
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy v failed");

    for (std::size_t r = 0; r < m; ++r) {
        for (std::size_t c = 0; c < m; ++c) {
            result.u(r, c) = u_col[c * m + r];
        }
    }
    if (returns_vt_direct) {
        // cuSOLVER gesvd returns VT directly in column-major layout.
        for (std::size_t r = 0; r < n; ++r) {
            for (std::size_t c = 0; c < n; ++c) {
                result.vt(r, c) = v_col[c * n + r];
            }
        }
    } else {
        // gesvdj returns V in column-major; reinterpret as row-major gives V^T.
        for (std::size_t r = 0; r < n; ++r) {
            for (std::size_t c = 0; c < n; ++c) {
                result.vt(r, c) = v_col[r * n + c];
            }
        }
    }

    return result;
}

double benchmark_svd_float_gpu_kernel_ms(const Array2D<float>& a, int iters) {
    if (iters < 1) iters = 1;
    if (a.rows() < a.cols()) {
        // Benchmark in stable orientation; element count is unchanged.
        return benchmark_svd_float_gpu_kernel_ms(transpose_host(a), iters);
    }
    const std::size_t m = a.rows(), n = a.cols();
    auto ws = svd_workspace_cache().get(m, n);
    std::lock_guard<std::mutex> lock(ws->run_mu);
    prepare_col_major_input(*ws, a.data(), a.size());
    float* d_a0 = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_a0), sizeof(float) * a.size()), "cudaMalloc d_a0 failed");
    check_cuda(cudaMemcpy(d_a0, ws->d_a, sizeof(float) * a.size(), cudaMemcpyDeviceToDevice), "cudaMemcpy d_a0 failed");

    const bool use_gesvd = (m * n) >= gesvd_threshold_elems();
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0); cudaEventCreate(&ev1);
    cudaEventRecord(ev0);
    for (int i = 0; i < iters; ++i) {
        check_cuda(cudaMemcpy(ws->d_a, d_a0, sizeof(float) * a.size(), cudaMemcpyDeviceToDevice),
                   "cudaMemcpy d_a reset failed");
        if (use_gesvd) {
            check_cusolver(cusolverDnSgesvd(
                svd_handle(), 'A', 'A', ws->m_i, ws->n_i, ws->d_a, ws->lda, ws->d_s,
                ws->d_u, ws->ldu, ws->d_v, ws->ldv, ws->d_work_gesvd, ws->lwork_gesvd,
                nullptr, ws->d_info), "cusolverDnSgesvd failed");
        } else {
            check_cusolver(cusolverDnSgesvdj(
                svd_handle(), CUSOLVER_EIG_MODE_VECTOR, 0, ws->m_i, ws->n_i,
                ws->d_a, ws->lda, ws->d_s, ws->d_u, ws->ldu, ws->d_v, ws->ldv,
                ws->d_work, ws->lwork, ws->d_info, ws->params), "cusolverDnSgesvdj failed");
        }
    }
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, ev0, ev1);
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    cudaFree(d_a0);
    return static_cast<double>(ms) / static_cast<double>(iters);
}

double benchmark_svd_float_gpu_xfer_ms(const Array2D<float>& a, int iters) {
    if (iters < 1) iters = 1;
    if (a.rows() < a.cols()) {
        return benchmark_svd_float_gpu_xfer_ms(transpose_host(a), iters);
    }
    const std::size_t m = a.rows(), n = a.cols();
    auto ws = svd_workspace_cache().get(m, n);
    std::lock_guard<std::mutex> lock(ws->run_mu);
    std::vector<float> u(m * m), vt(n * n), s(std::min(m, n));

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0); cudaEventCreate(&ev1);
    cudaEventRecord(ev0);
    for (int i = 0; i < iters; ++i) {
        check_cuda(cudaMemcpy(ws->d_a_rm, a.data(), sizeof(float) * a.size(), cudaMemcpyHostToDevice), "H2D");
        check_cuda(cudaMemcpy(u.data(), ws->d_u, sizeof(float) * u.size(), cudaMemcpyDeviceToHost), "D2H U");
        check_cuda(cudaMemcpy(vt.data(), ws->d_v, sizeof(float) * vt.size(), cudaMemcpyDeviceToHost), "D2H VT");
        check_cuda(cudaMemcpy(s.data(), ws->d_s, sizeof(float) * s.size(), cudaMemcpyDeviceToHost), "D2H S");
    }
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, ev0, ev1);
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    return static_cast<double>(ms) / static_cast<double>(iters);
}

double benchmark_svd_float_gpu_e2e_ms(const Array2D<float>& a, int iters) {
    if (iters < 1) iters = 1;
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0); cudaEventCreate(&ev1);
    cudaEventRecord(ev0);
    for (int i = 0; i < iters; ++i) {
        (void)svd_float_cuda(a);
    }
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, ev0, ev1);
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    return static_cast<double>(ms) / static_cast<double>(iters);
}

std::size_t svd_float_gpu_threshold_elems() {
    return gesvd_threshold_elems();
}

SvdCalibrationResult calibrate_svd_float_gpu_threshold() {
    const auto cal = run_svd_autocalibration();
    save_cached_threshold(cal.sm, cal.threshold_elems);
    return cal;
}

Array2D<double> beta_reg_gpu(const Array2D<double>& S, double beta) {
    const std::size_t m = S.rows();
    const std::size_t n = S.cols();
    const int ni = static_cast<int>(n);
    const int mi = static_cast<int>(m);

    // Upload S (m x n row-major). Interpreted by cuBLAS as a column-major
    // (n x m) matrix, this device buffer IS S^T (col-major). Call it dST.
    double* dST = nullptr;   // (n x m) col-major == S^T
    double* dA = nullptr;    // (n x n) col-major == S^T S (symmetric)
    double* dB = nullptr;    // (n x m) col-major RHS / solution
    double* d_diag = nullptr;
    double* d_work = nullptr;
    int* d_info = nullptr;
    check_cuda(cudaMalloc(&dST, sizeof(double) * m * n), "cudaMalloc dST");
    check_cuda(cudaMalloc(&dA, sizeof(double) * n * n), "cudaMalloc dA");
    check_cuda(cudaMalloc(&dB, sizeof(double) * m * n), "cudaMalloc dB");
    check_cuda(cudaMalloc(&d_diag, sizeof(double) * n), "cudaMalloc d_diag");
    check_cuda(cudaMalloc(&d_info, sizeof(int)), "cudaMalloc d_info");

    auto cleanup = [&]() {
        if (dST) cudaFree(dST);
        if (dA) cudaFree(dA);
        if (dB) cudaFree(dB);
        if (d_diag) cudaFree(d_diag);
        if (d_work) cudaFree(d_work);
        if (d_info) cudaFree(d_info);
    };

    try {
        check_cuda(cudaMemcpy(dST, S.data(), sizeof(double) * m * n, cudaMemcpyHostToDevice),
                   "H2D S");
        // RHS B = S^T (copy of dST).
        check_cuda(cudaMemcpy(dB, dST, sizeof(double) * m * n, cudaMemcpyDeviceToDevice),
                   "D2D B=S^T");

        // A = S^T S = dST (n x m) * dST^T (m x n)  -> (n x n), col-major.
        const double one = 1.0, zero = 0.0;
        check_cublas(cublasDgemm(blas_handle(), CUBLAS_OP_N, CUBLAS_OP_T,
                                 ni, ni, mi, &one, dST, ni, dST, ni, &zero, dA, ni),
                     "cublasDgemm S^T S");

        // alpha2 = max diagonal of A.
        const int block = 256;
        const int grid = (ni + block - 1) / block;
        gather_diag_kernel<<<grid, block>>>(dA, d_diag, ni);
        check_cuda(cudaGetLastError(), "gather_diag_kernel");
        std::vector<double> diag(n, 0.0);
        check_cuda(cudaMemcpy(diag.data(), d_diag, sizeof(double) * n, cudaMemcpyDeviceToHost),
                   "D2H diag");
        double alpha2 = 0.0;
        for (double v : diag) alpha2 = std::max(alpha2, v);
        const double reg = alpha2 * std::pow(10.0, beta);

        // A += reg * I.
        add_diag_kernel<<<grid, block>>>(dA, ni, reg);
        check_cuda(cudaGetLastError(), "add_diag_kernel");

        // Cholesky factor + solve  A X = B  (A SPD).  X overwrites dB.
        const cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
        int lwork = 0;
        check_cusolver(cusolverDnDpotrf_bufferSize(svd_handle(), uplo, ni, dA, ni, &lwork),
                       "Dpotrf_bufferSize");
        check_cuda(cudaMalloc(&d_work, sizeof(double) * std::max(lwork, 1)), "cudaMalloc work");
        check_cusolver(cusolverDnDpotrf(svd_handle(), uplo, ni, dA, ni, d_work, lwork, d_info),
                       "Dpotrf");
        int info = 0;
        check_cuda(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost), "D2H info potrf");
        if (info != 0) {
            throw std::runtime_error("beta_reg_gpu: Cholesky failed (info=" + std::to_string(info) + ")");
        }
        check_cusolver(cusolverDnDpotrs(svd_handle(), uplo, ni, mi, dA, ni, dB, ni, d_info),
                       "Dpotrs");
        check_cuda(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost), "D2H info potrs");
        if (info != 0) {
            throw std::runtime_error("beta_reg_gpu: solve failed (info=" + std::to_string(info) + ")");
        }

        // dB is now control = inv(A) @ S^T, shape (n x m) col-major.
        // Download and transpose into row-major (n x m).
        std::vector<double> host(m * n, 0.0);
        check_cuda(cudaMemcpy(host.data(), dB, sizeof(double) * m * n, cudaMemcpyDeviceToHost),
                   "D2H control");
        Array2D<double> control(n, m, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < m; ++j) {
                control(i, j) = host[j * n + i];  // col-major (i,j) -> row-major
            }
        }
        cleanup();
        return control;
    } catch (...) {
        cleanup();
        throw;
    }
}

} // namespace lina

#else

// CUDA not enabled; keep translation unit for build systems.
namespace lina {
SvdResultF svd_float_cuda(const Array2D<float>&) {
    throw std::runtime_error("CUDA SVD unavailable: build with LINA_USE_CUDA=ON");
}
double benchmark_svd_float_gpu_kernel_ms(const Array2D<float>&, int) {
    throw std::runtime_error("CUDA SVD unavailable: build with LINA_USE_CUDA=ON");
}
double benchmark_svd_float_gpu_xfer_ms(const Array2D<float>&, int) {
    throw std::runtime_error("CUDA SVD unavailable: build with LINA_USE_CUDA=ON");
}
double benchmark_svd_float_gpu_e2e_ms(const Array2D<float>&, int) {
    throw std::runtime_error("CUDA SVD unavailable: build with LINA_USE_CUDA=ON");
}
std::size_t svd_float_gpu_threshold_elems() {
    throw std::runtime_error("CUDA SVD unavailable: build with LINA_USE_CUDA=ON");
}
SvdCalibrationResult calibrate_svd_float_gpu_threshold() {
    throw std::runtime_error("CUDA SVD unavailable: build with LINA_USE_CUDA=ON");
}
Array2D<double> beta_reg_gpu(const Array2D<double>&, double) {
    throw std::runtime_error("beta_reg_gpu unavailable: build with LINA_USE_CUDA=ON");
}
} // namespace lina

#endif
