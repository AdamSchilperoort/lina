#include "lina/props.h"

#include <stdexcept>

#ifdef LINA_USE_CUDA

#include <cufft.h>
#include <cublas_v2.h>
#include <cuComplex.h>
#include <cuda_runtime.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace lina {
namespace {

void check_cuda(cudaError_t status, const char* msg) {
    if (status != cudaSuccess) {
        throw std::runtime_error(msg);
    }
}

void check_cufft(cufftResult status, const char* msg) {
    if (status != CUFFT_SUCCESS) {
        throw std::runtime_error(msg);
    }
}

Array2D<std::complex<double>> shift2d(const Array2D<std::complex<double>>& in,
                                      std::size_t shift_r,
                                      std::size_t shift_c) {
    const std::size_t rows = in.rows();
    const std::size_t cols = in.cols();
    Array2D<std::complex<double>> out(rows, cols, {0.0, 0.0});
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            const std::size_t rr = (r + shift_r) % rows;
            const std::size_t cc = (c + shift_c) % cols;
            out(rr, cc) = in(r, c);
        }
    }
    return out;
}

Array2D<std::complex<double>> fftshift_local(const Array2D<std::complex<double>>& in) {
    return shift2d(in, in.rows() / 2, in.cols() / 2);
}

Array2D<std::complex<double>> ifftshift_local(const Array2D<std::complex<double>>& in) {
    return shift2d(in, (in.rows() + 1) / 2, (in.cols() + 1) / 2);
}

struct CufftPlanKey {
    std::size_t rows;
    std::size_t cols;
    bool inverse;

    bool operator==(const CufftPlanKey& other) const {
        return rows == other.rows && cols == other.cols && inverse == other.inverse;
    }
};

struct CufftPlanKeyHash {
    std::size_t operator()(const CufftPlanKey& key) const noexcept {
        return std::hash<std::size_t>()(key.rows) ^
               (std::hash<std::size_t>()(key.cols) << 1) ^
               (std::hash<bool>()(key.inverse) << 2);
    }
};

class CufftPlanCache {
public:
    cufftHandle get_plan(std::size_t rows, std::size_t cols, bool inverse) {
        std::lock_guard<std::mutex> lock(mutex_);
        const CufftPlanKey key{rows, cols, inverse};
        const auto it = plans_.find(key);
        if (it != plans_.end()) {
            return it->second;
        }

        cufftHandle plan;
        check_cufft(cufftPlan2d(&plan, static_cast<int>(rows), static_cast<int>(cols), CUFFT_Z2Z),
                    "cufftPlan2d failed");
        plans_.emplace(key, plan);
        return plan;
    }

    ~CufftPlanCache() {
        for (const auto& item : plans_) {
            cufftDestroy(item.second);
        }
    }

private:
    std::unordered_map<CufftPlanKey, cufftHandle, CufftPlanKeyHash> plans_;
    std::mutex mutex_;
};

CufftPlanCache& cufft_cache() {
    static CufftPlanCache cache;
    return cache;
}

Array2D<std::complex<double>> fft_cufft(const Array2D<std::complex<double>>& arr, bool inverse) {
    const std::size_t rows = arr.rows();
    const std::size_t cols = arr.cols();
    const auto shifted = inverse ? ifftshift_local(arr) : fftshift_local(arr);

    Array2D<std::complex<double>> out(rows, cols, {0.0, 0.0});

    const std::size_t count = rows * cols;
    cufftDoubleComplex* d_data = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_data), sizeof(cufftDoubleComplex) * count),
               "cudaMalloc cufft data failed");

    check_cuda(cudaMemcpy(d_data, shifted.data(),
                          sizeof(cufftDoubleComplex) * count,
                          cudaMemcpyHostToDevice),
               "cudaMemcpy cufft input failed");

    const int direction = inverse ? CUFFT_INVERSE : CUFFT_FORWARD;
    cufftHandle plan = cufft_cache().get_plan(rows, cols, inverse);
    check_cufft(cufftExecZ2Z(plan, d_data, d_data, direction), "cufftExecZ2Z failed");

    check_cuda(cudaMemcpy(out.data(), d_data,
                          sizeof(cufftDoubleComplex) * count,
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy cufft output failed");

    if (inverse) {
        const double norm = 1.0 / static_cast<double>(rows * cols);
        for (std::size_t i = 0; i < out.size(); ++i) {
            out.data()[i] *= norm;
        }
        out = fftshift_local(out);
    } else {
        out = ifftshift_local(out);
    }

    cudaFree(d_data);
    return out;
}

} // namespace

Array2D<std::complex<double>> fft_gpu(const Array2D<std::complex<double>>& arr) {
    return fft_cufft(arr, false);
}

Array2D<std::complex<double>> ifft_gpu(const Array2D<std::complex<double>>& arr) {
    return fft_cufft(arr, true);
}

// ===========================================================================
// CUDA kernels for elementwise phase-mask construction.
// ===========================================================================
namespace {

constexpr double kPi    = 3.141592653589793238462643383279502884;
constexpr double kTwoPi = 6.283185307179586476925286766559005768;

// Vortex phase mask: phasor[r,c] = exp(j * charge * atan2(y, x)), where
// (x, y) are coordinates relative to the array centre. `offset` is 0 for
// "odd" centering (sample at zero) and 0.5 for "even" centering (no
// sample at zero).
__global__ void vortex_kernel(cuDoubleComplex* out,
                              std::size_t npix,
                              double charge,
                              double offset) {
    const std::size_t c = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= npix || c >= npix) return;
    const double half = static_cast<double>(npix) * 0.5;
    const double x = static_cast<double>(c) - half + offset;
    const double y = static_cast<double>(r) - half + offset;
    const double th = atan2(y, x);
    const double phase = charge * th;
    out[r * npix + c] = make_cuDoubleComplex(cos(phase), sin(phase));
}

// Fresnel TF: TF[r,c] = exp(-j * pi * dz * wavelength * r^2) where
// r = sqrt( ((i-(N-1)/2)*df)^2 + ((j-(N-1)/2)*df)^2 ) and df = 1/(N*wavelength*fnum).
// We pre-compute df and rho_scale on the host.
__global__ void fresnel_TF_kernel(cuDoubleComplex* out,
                                  std::size_t n,
                                  double df,
                                  double phase_coeff /* = -pi*dz*wavelength */) {
    const std::size_t c = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= n || c >= n) return;
    const double half = static_cast<double>(n) * 0.5 - 0.5; // (N-1)/2
    const double x = (static_cast<double>(c) - half) * df;
    const double y = (static_cast<double>(r) - half) * df;
    const double rho2 = x * x + y * y;
    const double phase = phase_coeff * rho2;
    out[r * n + c] = make_cuDoubleComplex(cos(phase), sin(phase));
}

// Pointwise multiply: out[i] = a[i] * b[i] (used by ang_spec).
__global__ void cmul_kernel(cuDoubleComplex* a,
                            const cuDoubleComplex* b,
                            std::size_t count) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    a[i] = cuCmul(a[i], b[i]);
}

// Build the angular-spectrum transfer function in-place on the device:
// tf[r,c] = exp(j * kz * distance) with kz = sqrt(k^2 - kx^2 - ky^2).
// kxy[i] = (i - n/2 + 0.5) * delkx.
__global__ void ang_spec_tf_kernel(cuDoubleComplex* tf,
                                   std::size_t n,
                                   double delkx,
                                   double k,
                                   double distance) {
    const std::size_t c = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= n || c >= n) return;
    const double half = static_cast<double>(n) * 0.5 - 0.5;
    const double kx = (static_cast<double>(c) - half) * delkx;
    const double ky = (static_cast<double>(r) - half) * delkx;
    const double rad2 = k * k - kx * kx - ky * ky;
    // kz might be imaginary for evanescent waves; handle both branches.
    if (rad2 >= 0.0) {
        const double kz = sqrt(rad2);
        const double phase = kz * distance;
        tf[r * n + c] = make_cuDoubleComplex(cos(phase), sin(phase));
    } else {
        // exp(j * (i*sqrt(-rad2)) * distance) = exp(-sqrt(-rad2)*distance)
        const double kz_im = sqrt(-rad2);
        const double mag = exp(-kz_im * distance);
        tf[r * n + c] = make_cuDoubleComplex(mag, 0.0);
    }
}

void check_cublas(cublasStatus_t s, const char* msg) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(msg);
    }
}

// Lazy cuBLAS handle.
cublasHandle_t cublas_handle() {
    static cublasHandle_t h = nullptr;
    static std::mutex m;
    std::lock_guard<std::mutex> lock(m);
    if (h == nullptr) {
        check_cublas(cublasCreate(&h), "cublasCreate failed");
    }
    return h;
}

// 1D coordinate vector matching lina::props::build_coordinates() for the
// "odd"/"even" centering used in mft_forward / mft_reverse.
//   odd:  x_i = (i - n/2) * pixelscale
//   even: x_i = (i - n/2 + 0.5) * pixelscale
std::vector<double> mft_coords(std::size_t n, double pixelscale, const std::string& centering) {
    std::vector<double> out(n);
    const double half = static_cast<double>(n) / 2.0;
    const double off = (centering == "even") ? 0.5 : 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = (static_cast<double>(i) - half + off) * pixelscale;
    }
    return out;
}

// Build the row-major MFT matrix M[ a, b ] = exp(j * sign * 2pi * U[a] * X[b]).
// `rows` is len(U), `cols` is len(X).
std::vector<std::complex<double>>
build_mft_matrix(const std::vector<double>& U,
                 const std::vector<double>& X,
                 double sign) {
    std::vector<std::complex<double>> M(U.size() * X.size());
    const std::complex<double> j(0.0, 1.0);
    for (std::size_t a = 0; a < U.size(); ++a) {
        for (std::size_t b = 0; b < X.size(); ++b) {
            const double phase = sign * kTwoPi * U[a] * X[b];
            M[a * X.size() + b] = std::exp(j * phase);
        }
    }
    return M;
}

// cuBLAS row-major Z-gemm: C(m, n) = alpha * A(m, k) * B(k, n) + beta * C
// using the standard "column-major view of row-major data" trick. All
// pointers are device pointers.
void zgemm_rm(cublasHandle_t handle,
              std::size_t m, std::size_t n, std::size_t k,
              const cuDoubleComplex* d_A,  // (m x k) row-major
              const cuDoubleComplex* d_B,  // (k x n) row-major
              cuDoubleComplex*       d_C,  // (m x n) row-major
              cuDoubleComplex alpha,
              cuDoubleComplex beta) {
    // Treat row-major (m x n) as column-major (n x m). Then
    // A_rm (m x k) = A_cm^T (k x m); to compute C_rm = A_rm @ B_rm we
    // ask cuBLAS to compute  C_cm (n x m) = B_cm (n x k) @ A_cm (k x m)
    // = B_rm^T @ A_rm^T  =  (A_rm @ B_rm)^T.
    // Storing the (n x m) column-major result into d_C lays out the
    // bytes exactly as the (m x n) row-major C_rm we wanted. Easy.
    check_cublas(
        cublasZgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    static_cast<int>(n), static_cast<int>(m), static_cast<int>(k),
                    &alpha,
                    d_B, static_cast<int>(n),
                    d_A, static_cast<int>(k),
                    &beta,
                    d_C, static_cast<int>(n)),
        "cublasZgemm failed");
}

// Helper: launch a 2D kernel covering an n x n grid.
inline dim3 grid2d(std::size_t n, dim3 block) {
    return dim3((static_cast<unsigned>(n) + block.x - 1) / block.x,
                (static_cast<unsigned>(n) + block.y - 1) / block.y);
}

} // namespace

// ===========================================================================
// New GPU implementations.
// ===========================================================================

Array2D<std::complex<double>> make_vortex_phase_mask_gpu(std::size_t npix,
                                                         int charge,
                                                         const char* grid) {
    const std::string g = grid ? grid : "odd";
    const double offset = (g == "even") ? 0.5 : 0.0;
    Array2D<std::complex<double>> out(npix, npix, {0.0, 0.0});

    cuDoubleComplex* d_out = nullptr;
    const std::size_t bytes = sizeof(cuDoubleComplex) * npix * npix;
    check_cuda(cudaMalloc(&d_out, bytes), "cudaMalloc vortex out");

    const dim3 block(16, 16);
    const dim3 gridDim = grid2d(npix, block);
    vortex_kernel<<<gridDim, block>>>(d_out, npix,
                                       static_cast<double>(charge), offset);
    check_cuda(cudaGetLastError(), "vortex_kernel launch");
    check_cuda(cudaMemcpy(out.data(), d_out, bytes, cudaMemcpyDeviceToHost),
               "cudaMemcpy vortex out");
    cudaFree(d_out);
    return out;
}

Array2D<std::complex<double>> get_fresnel_TF_gpu(double dz,
                                                 std::size_t n,
                                                 double wavelength,
                                                 double fnum) {
    const double df = 1.0 / (static_cast<double>(n) * wavelength * fnum);
    const double phase_coeff = -kPi * dz * wavelength;

    Array2D<std::complex<double>> out(n, n, {0.0, 0.0});
    cuDoubleComplex* d_out = nullptr;
    const std::size_t bytes = sizeof(cuDoubleComplex) * n * n;
    check_cuda(cudaMalloc(&d_out, bytes), "cudaMalloc fresnel out");

    const dim3 block(16, 16);
    const dim3 gridDim = grid2d(n, block);
    fresnel_TF_kernel<<<gridDim, block>>>(d_out, n, df, phase_coeff);
    check_cuda(cudaGetLastError(), "fresnel_TF_kernel launch");
    check_cuda(cudaMemcpy(out.data(), d_out, bytes, cudaMemcpyDeviceToHost),
               "cudaMemcpy fresnel out");
    cudaFree(d_out);
    return out;
}

Array2D<std::complex<double>> ang_spec_gpu(const Array2D<std::complex<double>>& wavefront,
                                           double wavelength,
                                           double distance,
                                           double pixelscale) {
    const std::size_t n = wavefront.rows();
    if (n != wavefront.cols()) {
        throw std::invalid_argument("ang_spec_gpu expects square wavefront");
    }

    // 1) FFT (cuFFT path, host-side fftshift wrappers as for fft_gpu)
    Array2D<std::complex<double>> wf_as = fft_cufft(wavefront, false);

    // 2) Multiply by transfer function on the GPU.
    const double delkx = kTwoPi / (static_cast<double>(n) * pixelscale);
    const double k = kTwoPi / wavelength;

    cuDoubleComplex* d_wf = nullptr;
    cuDoubleComplex* d_tf = nullptr;
    const std::size_t bytes = sizeof(cuDoubleComplex) * n * n;
    check_cuda(cudaMalloc(&d_wf, bytes), "cudaMalloc ang_spec wf");
    check_cuda(cudaMalloc(&d_tf, bytes), "cudaMalloc ang_spec tf");
    check_cuda(cudaMemcpy(d_wf, wf_as.data(), bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy ang_spec wf");

    const dim3 block(16, 16);
    const dim3 gridDim = grid2d(n, block);
    ang_spec_tf_kernel<<<gridDim, block>>>(d_tf, n, delkx, k, distance);
    check_cuda(cudaGetLastError(), "ang_spec_tf_kernel launch");

    const std::size_t total = n * n;
    const dim3 b1(256);
    const dim3 g1((static_cast<unsigned>(total) + b1.x - 1) / b1.x);
    cmul_kernel<<<g1, b1>>>(d_wf, d_tf, total);
    check_cuda(cudaGetLastError(), "cmul_kernel launch");

    Array2D<std::complex<double>> filtered(n, n, {0.0, 0.0});
    check_cuda(cudaMemcpy(filtered.data(), d_wf, bytes, cudaMemcpyDeviceToHost),
               "cudaMemcpy ang_spec filtered");
    cudaFree(d_wf);
    cudaFree(d_tf);

    // 3) IFFT
    return fft_cufft(filtered, true);
}

// ---------------------------------------------------------------------------
// MFT on GPU. Implemented as two zgemm calls on the device.
//
//   mft_forward:   out = (Mx @ wavefront @ My) * (psf_pixelscale_lamD / npix)
//   mft_reverse:   out = (Mx_rev @ fpwf @ My_rev) * (psf_pixelscale_lamD / npix)
//
// The matrices Mx / My are built on the host (double-complex, row-major)
// and copied to the device. For typical sizes (npix in [128, 2048],
// npsf in [32, 1024]) this is dominated by the gemm cost so the H2D
// matrix transfer is negligible.
// ---------------------------------------------------------------------------

Array2D<std::complex<double>> mft_forward_gpu(
    const Array2D<std::complex<double>>& wavefront,
    std::size_t npix,
    std::size_t npsf,
    double psf_pixelscale_lamD,
    char convention,
    const char* pp_centering,
    const char* fp_centering) {

    const std::size_t N = wavefront.rows();
    if (N != wavefront.cols()) {
        throw std::invalid_argument("mft_forward_gpu expects square wavefront");
    }
    const double dx = 1.0 / static_cast<double>(npix);
    const double du = psf_pixelscale_lamD;

    const auto Xs = mft_coords(N, dx,    pp_centering ? pp_centering : "odd");
    const auto Us = mft_coords(npsf, du, fp_centering ? fp_centering : "odd");

    const double sign = (convention == '-') ? -1.0 : 1.0;

    // Mx (npsf x N) and My (N x npsf), both row-major.
    auto h_Mx = build_mft_matrix(Us, Xs, sign);  // (npsf, N)
    auto h_My = build_mft_matrix(Xs, Us, sign);  // (N, npsf)  -- note flipped order

    // Allocate device buffers.
    cuDoubleComplex *d_W = nullptr, *d_Mx = nullptr, *d_My = nullptr,
                    *d_T = nullptr, *d_O = nullptr;
    const std::size_t W_bytes  = sizeof(cuDoubleComplex) * N * N;
    const std::size_t Mx_bytes = sizeof(cuDoubleComplex) * npsf * N;
    const std::size_t My_bytes = sizeof(cuDoubleComplex) * N * npsf;
    const std::size_t T_bytes  = sizeof(cuDoubleComplex) * npsf * N;
    const std::size_t O_bytes  = sizeof(cuDoubleComplex) * npsf * npsf;
    check_cuda(cudaMalloc(&d_W,  W_bytes),  "cudaMalloc mft_fwd W");
    check_cuda(cudaMalloc(&d_Mx, Mx_bytes), "cudaMalloc mft_fwd Mx");
    check_cuda(cudaMalloc(&d_My, My_bytes), "cudaMalloc mft_fwd My");
    check_cuda(cudaMalloc(&d_T,  T_bytes),  "cudaMalloc mft_fwd T");
    check_cuda(cudaMalloc(&d_O,  O_bytes),  "cudaMalloc mft_fwd O");

    check_cuda(cudaMemcpy(d_W,  wavefront.data(), W_bytes,  cudaMemcpyHostToDevice), "H2D W");
    check_cuda(cudaMemcpy(d_Mx, h_Mx.data(),      Mx_bytes, cudaMemcpyHostToDevice), "H2D Mx");
    check_cuda(cudaMemcpy(d_My, h_My.data(),      My_bytes, cudaMemcpyHostToDevice), "H2D My");

    cublasHandle_t handle = cublas_handle();
    const cuDoubleComplex one  = make_cuDoubleComplex(1.0, 0.0);
    const cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);

    // T = Mx @ W   (npsf x N) = (npsf x N)(N x N)
    zgemm_rm(handle, npsf, N, N, d_Mx, d_W, d_T, one, zero);
    // O = T @ My   (npsf x npsf) = (npsf x N)(N x npsf)
    zgemm_rm(handle, npsf, npsf, N, d_T, d_My, d_O, one, zero);

    // Scale by psf_pixelscale_lamD / npix in-place on the device.
    const double scale = psf_pixelscale_lamD / static_cast<double>(npix);
    const cuDoubleComplex scale_cplx = make_cuDoubleComplex(scale, 0.0);
    check_cublas(cublasZscal(handle, static_cast<int>(npsf * npsf),
                              &scale_cplx, d_O, 1),
                 "cublasZscal mft_fwd scale");

    Array2D<std::complex<double>> out(npsf, npsf, {0.0, 0.0});
    check_cuda(cudaMemcpy(out.data(), d_O, O_bytes, cudaMemcpyDeviceToHost), "D2H mft_fwd");

    cudaFree(d_W); cudaFree(d_Mx); cudaFree(d_My); cudaFree(d_T); cudaFree(d_O);
    return out;
}

Array2D<std::complex<double>> mft_reverse_gpu(
    const Array2D<std::complex<double>>& fpwf,
    double psf_pixelscale_lamD,
    std::size_t npix,
    std::size_t N,
    char convention,
    const char* pp_centering,
    const char* fp_centering) {

    const std::size_t npsf = fpwf.rows();
    if (npsf != fpwf.cols()) {
        throw std::invalid_argument("mft_reverse_gpu expects square focal plane");
    }
    const double du = psf_pixelscale_lamD;
    const double dx = 1.0 / static_cast<double>(npix);

    const auto Us = mft_coords(npsf, du, fp_centering ? fp_centering : "odd");
    const auto Xs = mft_coords(N,    dx, pp_centering ? pp_centering : "odd");

    const double sign = (convention == '+') ? 1.0 : -1.0;

    // Mx (N x npsf): exp(j*sign*2pi * Xs[x] * Us[u])
    // My (npsf x N): exp(j*sign*2pi * Us[v] * Xs[y])
    auto h_Mx = build_mft_matrix(Xs, Us, sign);  // (N, npsf)
    auto h_My = build_mft_matrix(Us, Xs, sign);  // (npsf, N)

    cuDoubleComplex *d_F = nullptr, *d_Mx = nullptr, *d_My = nullptr,
                    *d_T = nullptr, *d_O = nullptr;
    const std::size_t F_bytes  = sizeof(cuDoubleComplex) * npsf * npsf;
    const std::size_t Mx_bytes = sizeof(cuDoubleComplex) * N * npsf;
    const std::size_t My_bytes = sizeof(cuDoubleComplex) * npsf * N;
    const std::size_t T_bytes  = sizeof(cuDoubleComplex) * N * npsf;
    const std::size_t O_bytes  = sizeof(cuDoubleComplex) * N * N;
    check_cuda(cudaMalloc(&d_F,  F_bytes),  "cudaMalloc mft_rev F");
    check_cuda(cudaMalloc(&d_Mx, Mx_bytes), "cudaMalloc mft_rev Mx");
    check_cuda(cudaMalloc(&d_My, My_bytes), "cudaMalloc mft_rev My");
    check_cuda(cudaMalloc(&d_T,  T_bytes),  "cudaMalloc mft_rev T");
    check_cuda(cudaMalloc(&d_O,  O_bytes),  "cudaMalloc mft_rev O");

    check_cuda(cudaMemcpy(d_F,  fpwf.data(),  F_bytes,  cudaMemcpyHostToDevice), "H2D F");
    check_cuda(cudaMemcpy(d_Mx, h_Mx.data(),  Mx_bytes, cudaMemcpyHostToDevice), "H2D Mx");
    check_cuda(cudaMemcpy(d_My, h_My.data(),  My_bytes, cudaMemcpyHostToDevice), "H2D My");

    cublasHandle_t handle = cublas_handle();
    const cuDoubleComplex one  = make_cuDoubleComplex(1.0, 0.0);
    const cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);

    // T = Mx @ F  (N x npsf) = (N x npsf)(npsf x npsf)
    zgemm_rm(handle, N, npsf, npsf, d_Mx, d_F, d_T, one, zero);
    // O = T @ My  (N x N) = (N x npsf)(npsf x N)
    zgemm_rm(handle, N, N, npsf, d_T, d_My, d_O, one, zero);

    const double scale = psf_pixelscale_lamD / static_cast<double>(npix);
    const cuDoubleComplex scale_cplx = make_cuDoubleComplex(scale, 0.0);
    check_cublas(cublasZscal(handle, static_cast<int>(N * N),
                              &scale_cplx, d_O, 1),
                 "cublasZscal mft_rev scale");

    Array2D<std::complex<double>> out(N, N, {0.0, 0.0});
    check_cuda(cudaMemcpy(out.data(), d_O, O_bytes, cudaMemcpyDeviceToHost), "D2H mft_rev");

    cudaFree(d_F); cudaFree(d_Mx); cudaFree(d_My); cudaFree(d_T); cudaFree(d_O);
    return out;
}

} // namespace lina

#else

namespace lina {
Array2D<std::complex<double>> fft_gpu(const Array2D<std::complex<double>>& ) {
    throw std::runtime_error("CUDA FFT unavailable: build with LINA_USE_CUDA=ON");
}
Array2D<std::complex<double>> ifft_gpu(const Array2D<std::complex<double>>& ) {
    throw std::runtime_error("CUDA FFT unavailable: build with LINA_USE_CUDA=ON");
}
Array2D<std::complex<double>> ang_spec_gpu(const Array2D<std::complex<double>>&,
                                           double, double, double) {
    throw std::runtime_error("ang_spec_gpu unavailable: build with LINA_USE_CUDA=ON");
}
Array2D<std::complex<double>> make_vortex_phase_mask_gpu(std::size_t, int, const char*) {
    throw std::runtime_error("make_vortex_phase_mask_gpu unavailable: build with LINA_USE_CUDA=ON");
}
Array2D<std::complex<double>> mft_forward_gpu(const Array2D<std::complex<double>>&,
                                              std::size_t, std::size_t, double, char,
                                              const char*, const char*) {
    throw std::runtime_error("mft_forward_gpu unavailable: build with LINA_USE_CUDA=ON");
}
Array2D<std::complex<double>> mft_reverse_gpu(const Array2D<std::complex<double>>&,
                                              double, std::size_t, std::size_t, char,
                                              const char*, const char*) {
    throw std::runtime_error("mft_reverse_gpu unavailable: build with LINA_USE_CUDA=ON");
}
Array2D<std::complex<double>> get_fresnel_TF_gpu(double, std::size_t, double, double) {
    throw std::runtime_error("get_fresnel_TF_gpu unavailable: build with LINA_USE_CUDA=ON");
}
} // namespace lina

#endif
