#include "lina/props.h"

#include <complex>
#include <cstddef>
#include <stdexcept>

#ifdef LINA_USE_CUDA

#include <cufft.h>
#include <cublas_v2.h>
#include <cuComplex.h>
#include <cuda_runtime.h>

#include <cmath>
#include <memory>
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

// ---------------------------------------------------------------------------
// GPU shift kernels (replace host-side fftshift / ifftshift).
//
// shift2d_kernel reads from `in[r, c]` and writes to `out[(r+shift_r)%rows,
// (c+shift_c)%cols]`. This is the same convention as the host-side
// shift2d above, with two specialisations:
//   * fftshift  : shift_r = rows/2,        shift_c = cols/2         (floor)
//   * ifftshift : shift_r = (rows+1)/2,    shift_c = (cols+1)/2     (ceil)
// For even sizes these coincide; for odd sizes they differ by one row /
// column, matching numpy / cupy.
//
// The `_scaled` variant multiplies by a real scalar in the same pass,
// which lets the inverse-FFT normalisation (1/N) happen "for free"
// while we are already touching every output element.
// ---------------------------------------------------------------------------

__global__ void shift2d_kernel(const cufftDoubleComplex* __restrict__ in,
                               cufftDoubleComplex*       __restrict__ out,
                               std::size_t rows, std::size_t cols,
                               std::size_t shift_r, std::size_t shift_c) {
    const std::size_t c = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= rows || c >= cols) return;
    const std::size_t rr = (r + shift_r) % rows;
    const std::size_t cc = (c + shift_c) % cols;
    out[rr * cols + cc] = in[r * cols + c];
}

__global__ void shift2d_scaled_kernel(const cufftDoubleComplex* __restrict__ in,
                                      cufftDoubleComplex*       __restrict__ out,
                                      std::size_t rows, std::size_t cols,
                                      std::size_t shift_r, std::size_t shift_c,
                                      double scale) {
    const std::size_t c = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= rows || c >= cols) return;
    const std::size_t rr = (r + shift_r) % rows;
    const std::size_t cc = (c + shift_c) % cols;
    const cufftDoubleComplex v = in[r * cols + c];
    out[rr * cols + cc] = make_cuDoubleComplex(v.x * scale, v.y * scale);
}

struct CufftPlanKey {
    std::size_t rows;
    std::size_t cols;

    bool operator==(const CufftPlanKey& other) const {
        return rows == other.rows && cols == other.cols;
    }
};

struct CufftPlanKeyHash {
    std::size_t operator()(const CufftPlanKey& key) const noexcept {
        return std::hash<std::size_t>()(key.rows) ^
               (std::hash<std::size_t>()(key.cols) << 1);
    }
};

// A single cuFFT plan handles both forward and inverse (direction is
// passed to cufftExecZ2Z). So we cache one plan per (rows, cols).
class CufftPlanCache {
public:
    cufftHandle get_plan(std::size_t rows, std::size_t cols) {
        std::lock_guard<std::mutex> lock(mutex_);
        const CufftPlanKey key{rows, cols};
        const auto it = plans_.find(key);
        if (it != plans_.end()) {
            return it->second;
        }
        cufftHandle plan;
        check_cufft(cufftPlan2d(&plan,
                                static_cast<int>(rows),
                                static_cast<int>(cols),
                                CUFFT_Z2Z),
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

// ---------------------------------------------------------------------------
// Device scratch pool.
//
// Each (rows, cols) FFT call needs two device-side complex buffers (one
// for the staged input + shifted-output, one for the FFT in/out). We
// keep them per-size in a small LRU cache so that a steady-state loop
// never calls cudaMalloc / cudaFree.
// ---------------------------------------------------------------------------

struct DeviceScratch {
    cufftDoubleComplex* buf_a = nullptr;
    cufftDoubleComplex* buf_b = nullptr;
    std::size_t count = 0;

    DeviceScratch(std::size_t n) : count(n) {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&buf_a),
                              sizeof(cufftDoubleComplex) * n),
                   "cudaMalloc fft scratch A failed");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&buf_b),
                              sizeof(cufftDoubleComplex) * n),
                   "cudaMalloc fft scratch B failed");
    }
    ~DeviceScratch() {
        if (buf_a) cudaFree(buf_a);
        if (buf_b) cudaFree(buf_b);
    }
    DeviceScratch(const DeviceScratch&) = delete;
    DeviceScratch& operator=(const DeviceScratch&) = delete;
};

class ScratchCache {
public:
    std::shared_ptr<DeviceScratch> get(std::size_t rows, std::size_t cols) {
        std::lock_guard<std::mutex> lock(mutex_);
        const CufftPlanKey key{rows, cols};
        auto it = pool_.find(key);
        if (it != pool_.end()) return it->second;
        if (pool_.size() >= kCap) pool_.erase(pool_.begin());
        auto s = std::make_shared<DeviceScratch>(rows * cols);
        pool_.emplace(key, s);
        return s;
    }

private:
    static constexpr std::size_t kCap = 16;
    std::unordered_map<CufftPlanKey,
                       std::shared_ptr<DeviceScratch>,
                       CufftPlanKeyHash> pool_;
    std::mutex mutex_;
};

ScratchCache& scratch_cache() {
    static ScratchCache c;
    return c;
}

// Dedicated CUDA stream for FFTs so multiple library entry points can
// pipeline H2D / kernels / D2H without serialising on the default
// stream against unrelated work.
cudaStream_t fft_stream() {
    static cudaStream_t s = []() {
        cudaStream_t tmp;
        check_cuda(cudaStreamCreate(&tmp), "cudaStreamCreate failed");
        return tmp;
    }();
    return s;
}

Array2D<std::complex<double>> fft_cufft(const Array2D<std::complex<double>>& arr,
                                        bool inverse) {
    const std::size_t rows  = arr.rows();
    const std::size_t cols  = arr.cols();
    const std::size_t count = rows * cols;

    auto scratch     = scratch_cache().get(rows, cols);
    cufftHandle plan = cufft_cache().get_plan(rows, cols);
    cudaStream_t stm = fft_stream();
    check_cufft(cufftSetStream(plan, stm), "cufftSetStream failed");

    // Pre-shift parameters (numpy convention used by lina):
    //   forward : input is fftshifted before fft2, ifftshifted after.
    //   inverse : input is ifftshifted before ifft2, fftshifted after.
    const std::size_t pre_r  = inverse ? (rows + 1) / 2 : rows / 2;
    const std::size_t pre_c  = inverse ? (cols + 1) / 2 : cols / 2;
    const std::size_t post_r = inverse ? rows / 2 : (rows + 1) / 2;
    const std::size_t post_c = inverse ? cols / 2 : (cols + 1) / 2;

    // 1) Async H2D into buf_a.
    check_cuda(cudaMemcpyAsync(scratch->buf_a, arr.data(),
                               sizeof(cufftDoubleComplex) * count,
                               cudaMemcpyHostToDevice, stm),
               "cudaMemcpyAsync H2D failed");

    // 2) Pre-shift on GPU: buf_a -> buf_b.
    {
        const dim3 block(16, 16);
        const dim3 grid((cols + block.x - 1) / block.x,
                        (rows + block.y - 1) / block.y);
        shift2d_kernel<<<grid, block, 0, stm>>>(scratch->buf_a, scratch->buf_b,
                                                rows, cols, pre_r, pre_c);
    }

    // 3) cuFFT in-place on buf_b.
    const int direction = inverse ? CUFFT_INVERSE : CUFFT_FORWARD;
    check_cufft(cufftExecZ2Z(plan, scratch->buf_b, scratch->buf_b, direction),
                "cufftExecZ2Z failed");

    // 4) Post-shift (and inverse scale, fused) on GPU: buf_b -> buf_a.
    {
        const dim3 block(16, 16);
        const dim3 grid((cols + block.x - 1) / block.x,
                        (rows + block.y - 1) / block.y);
        if (inverse) {
            const double norm = 1.0 / static_cast<double>(count);
            shift2d_scaled_kernel<<<grid, block, 0, stm>>>(
                scratch->buf_b, scratch->buf_a, rows, cols, post_r, post_c, norm);
        } else {
            shift2d_kernel<<<grid, block, 0, stm>>>(
                scratch->buf_b, scratch->buf_a, rows, cols, post_r, post_c);
        }
    }

    // 5) Async D2H into the output buffer.
    Array2D<std::complex<double>> out(rows, cols, {0.0, 0.0});
    check_cuda(cudaMemcpyAsync(out.data(), scratch->buf_a,
                               sizeof(cufftDoubleComplex) * count,
                               cudaMemcpyDeviceToHost, stm),
               "cudaMemcpyAsync D2H failed");

    // 6) Synchronise -- caller observes a fully-materialised host array.
    check_cuda(cudaStreamSynchronize(stm), "cudaStreamSynchronize failed");

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

// MFT matrix builder on device:
//   M[a, b] = exp(j * sign * 2pi * U[a] * X[b])
// `U` is a device buffer of length `rows`; `X` is a device buffer of
// length `cols`. The grid maps (a, b) -> (blockIdx.y * blockDim.y +
// threadIdx.y, blockIdx.x * blockDim.x + threadIdx.x).
__global__ void mft_matrix_kernel(cuDoubleComplex* __restrict__ M,
                                  const double*    __restrict__ U,
                                  const double*    __restrict__ X,
                                  std::size_t rows, std::size_t cols,
                                  double sign_two_pi /* = sign * 2*pi */) {
    const std::size_t b = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t a = blockIdx.y * blockDim.y + threadIdx.y;
    if (a >= rows || b >= cols) return;
    const double phase = sign_two_pi * U[a] * X[b];
    M[a * cols + b] = make_cuDoubleComplex(cos(phase), sin(phase));
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

// ---------------------------------------------------------------------------
// Device-side MFT matrix cache.
//
// The CPU MFT learned the same trick: in a wavefront-control loop the
// geometry (npix, npsf, du, sign, centering) is invariant, so we should
// build Mx / My exactly once per geometry. On the GPU we additionally
// avoid the host build + H2D entirely by building the matrices on the
// device using a small kernel. A single shared_ptr<DeviceMftMatrix>
// holds the device pointer + size so eviction is reference-counted and
// the matrix cannot be freed while any caller is mid-gemm.
// ---------------------------------------------------------------------------

struct DeviceMftMatrix {
    cuDoubleComplex* d_M = nullptr;
    std::size_t rows = 0;
    std::size_t cols = 0;

    DeviceMftMatrix(std::size_t r, std::size_t c) : rows(r), cols(c) {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_M),
                              sizeof(cuDoubleComplex) * r * c),
                   "cudaMalloc mft matrix failed");
    }
    ~DeviceMftMatrix() { if (d_M) cudaFree(d_M); }
    DeviceMftMatrix(const DeviceMftMatrix&) = delete;
    DeviceMftMatrix& operator=(const DeviceMftMatrix&) = delete;
};

struct MftDevKey {
    std::size_t rows;
    std::size_t cols;
    double sign_two_pi;
    // Hash of the U and X coordinate vectors (which fully determine the
    // matrix). We hash bit-for-bit so values that compare equal as
    // doubles hash equal; the coordinate vectors are deterministic
    // functions of (n, pixelscale, centering), so this matches the
    // intent of "same geometry -> same matrix".
    std::size_t u_hash;
    std::size_t x_hash;
    bool operator==(const MftDevKey& o) const noexcept {
        return rows == o.rows && cols == o.cols
            && sign_two_pi == o.sign_two_pi
            && u_hash == o.u_hash && x_hash == o.x_hash;
    }
};

struct MftDevKeyHash {
    std::size_t operator()(const MftDevKey& k) const noexcept {
        std::size_t h = std::hash<std::size_t>{}(k.rows);
        h ^= std::hash<std::size_t>{}(k.cols)  + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
        h ^= std::hash<double>{}(k.sign_two_pi)+ 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
        h ^= k.u_hash                          + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
        h ^= k.x_hash                          + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
        return h;
    }
};

std::size_t hash_dvec(const std::vector<double>& v) {
    std::size_t h = std::hash<std::size_t>{}(v.size());
    for (double x : v) {
        h ^= std::hash<double>{}(x) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    }
    return h;
}

class MftDevCache {
public:
    std::shared_ptr<DeviceMftMatrix> get_or_build(
        const std::vector<double>& U,
        const std::vector<double>& X,
        double sign) {
        const double s2pi = sign * kTwoPi;
        const MftDevKey key{U.size(), X.size(), s2pi,
                            hash_dvec(U), hash_dvec(X)};

        std::lock_guard<std::mutex> lk(mu_);
        auto it = pool_.find(key);
        if (it != pool_.end()) return it->second;
        if (pool_.size() >= kCap) pool_.erase(pool_.begin());

        // Stage U and X to device, fire the matrix kernel, free U/X.
        double* d_U = nullptr;
        double* d_X = nullptr;
        check_cuda(cudaMalloc(&d_U, sizeof(double) * U.size()),
                   "cudaMalloc mft U failed");
        check_cuda(cudaMalloc(&d_X, sizeof(double) * X.size()),
                   "cudaMalloc mft X failed");
        check_cuda(cudaMemcpy(d_U, U.data(), sizeof(double) * U.size(),
                              cudaMemcpyHostToDevice),
                   "H2D mft U failed");
        check_cuda(cudaMemcpy(d_X, X.data(), sizeof(double) * X.size(),
                              cudaMemcpyHostToDevice),
                   "H2D mft X failed");

        auto M = std::make_shared<DeviceMftMatrix>(U.size(), X.size());
        const dim3 block(16, 16);
        const dim3 grid((static_cast<unsigned>(X.size()) + block.x - 1) / block.x,
                        (static_cast<unsigned>(U.size()) + block.y - 1) / block.y);
        mft_matrix_kernel<<<grid, block>>>(M->d_M, d_U, d_X,
                                           U.size(), X.size(), s2pi);
        check_cuda(cudaGetLastError(),  "mft_matrix_kernel launch failed");
        check_cuda(cudaDeviceSynchronize(),
                   "cudaDeviceSynchronize mft matrix failed");

        cudaFree(d_U);
        cudaFree(d_X);

        pool_.emplace(key, M);
        return M;
    }

private:
    static constexpr std::size_t kCap = 32;
    std::unordered_map<MftDevKey, std::shared_ptr<DeviceMftMatrix>,
                       MftDevKeyHash> pool_;
    std::mutex mu_;
};

MftDevCache& mft_dev_cache() {
    static MftDevCache c;
    return c;
}

// ---------------------------------------------------------------------------
// Device-side complex scratch pool (generic, by element count).
//
// Used for the wavefront, T, and O buffers of mft_forward_gpu /
// mft_reverse_gpu. Each call grabs three independent scratches; the
// cache holds up to kCap distinct sizes so a steady-state pipeline with
// a few sizes is allocation-free.
// ---------------------------------------------------------------------------

struct DeviceComplexScratch {
    cuDoubleComplex* d = nullptr;
    std::size_t count = 0;
    explicit DeviceComplexScratch(std::size_t n) : count(n) {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d),
                              sizeof(cuDoubleComplex) * n),
                   "cudaMalloc scratch failed");
    }
    ~DeviceComplexScratch() { if (d) cudaFree(d); }
    DeviceComplexScratch(const DeviceComplexScratch&) = delete;
    DeviceComplexScratch& operator=(const DeviceComplexScratch&) = delete;
};

class ComplexScratchPool {
public:
    // Returns a buffer of at least `n` complex elements. Same buffer may
    // be returned to multiple callers; callers should treat the
    // contents as undefined. The shared_ptr lifetime guards eviction.
    std::shared_ptr<DeviceComplexScratch> get(std::size_t n, int slot) {
        std::lock_guard<std::mutex> lk(mu_);
        const auto key = std::make_pair(n, slot);
        auto it = pool_.find(key);
        if (it != pool_.end()) return it->second;
        if (pool_.size() >= kCap) pool_.erase(pool_.begin());
        auto s = std::make_shared<DeviceComplexScratch>(n);
        pool_.emplace(key, s);
        return s;
    }
private:
    struct PairHash {
        std::size_t operator()(const std::pair<std::size_t, int>& p) const noexcept {
            return std::hash<std::size_t>{}(p.first) ^ (std::hash<int>{}(p.second) << 1);
        }
    };
    static constexpr std::size_t kCap = 48;
    std::unordered_map<std::pair<std::size_t, int>,
                       std::shared_ptr<DeviceComplexScratch>,
                       PairHash> pool_;
    std::mutex mu_;
};

ComplexScratchPool& complex_scratch_pool() {
    static ComplexScratchPool p;
    return p;
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

    // Mx (npsf x N) and My (N x npsf) cached on the device. First call
    // for a given (npix, npsf, du, sign, centering) pays a small kernel
    // build cost; subsequent calls are free.
    auto Mx = mft_dev_cache().get_or_build(Us, Xs, sign); // (npsf, N)
    auto My = mft_dev_cache().get_or_build(Xs, Us, sign); // (N, npsf)

    // Scratch device buffers (W, T, O) drawn from the size-keyed pool.
    auto W_buf = complex_scratch_pool().get(N * N,        /*slot=*/0);
    auto T_buf = complex_scratch_pool().get(npsf * N,     /*slot=*/1);
    auto O_buf = complex_scratch_pool().get(npsf * npsf,  /*slot=*/2);

    const std::size_t W_bytes = sizeof(cuDoubleComplex) * N * N;
    const std::size_t O_bytes = sizeof(cuDoubleComplex) * npsf * npsf;

    check_cuda(cudaMemcpy(W_buf->d, wavefront.data(), W_bytes,
                          cudaMemcpyHostToDevice), "H2D W");

    cublasHandle_t handle = cublas_handle();
    const cuDoubleComplex one  = make_cuDoubleComplex(1.0, 0.0);
    const cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);

    // T = Mx @ W   (npsf x N) = (npsf x N)(N x N)
    zgemm_rm(handle, npsf, N, N, Mx->d_M, W_buf->d, T_buf->d, one, zero);
    // O = T @ My   (npsf x npsf) = (npsf x N)(N x npsf)
    zgemm_rm(handle, npsf, npsf, N, T_buf->d, My->d_M, O_buf->d, one, zero);

    // Scale by psf_pixelscale_lamD / npix in-place on the device.
    const double scale = psf_pixelscale_lamD / static_cast<double>(npix);
    const cuDoubleComplex scale_cplx = make_cuDoubleComplex(scale, 0.0);
    check_cublas(cublasZscal(handle, static_cast<int>(npsf * npsf),
                              &scale_cplx, O_buf->d, 1),
                 "cublasZscal mft_fwd scale");

    Array2D<std::complex<double>> out(npsf, npsf, {0.0, 0.0});
    check_cuda(cudaMemcpy(out.data(), O_buf->d, O_bytes,
                          cudaMemcpyDeviceToHost), "D2H mft_fwd");
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
    auto Mx = mft_dev_cache().get_or_build(Xs, Us, sign);  // (N, npsf)
    auto My = mft_dev_cache().get_or_build(Us, Xs, sign);  // (npsf, N)

    // Use disjoint scratch slots from the forward path so a mixed
    // forward/reverse loop does not stomp each other's buffers.
    auto F_buf = complex_scratch_pool().get(npsf * npsf, /*slot=*/3);
    auto T_buf = complex_scratch_pool().get(N * npsf,    /*slot=*/4);
    auto O_buf = complex_scratch_pool().get(N * N,       /*slot=*/5);

    const std::size_t F_bytes = sizeof(cuDoubleComplex) * npsf * npsf;
    const std::size_t O_bytes = sizeof(cuDoubleComplex) * N * N;

    check_cuda(cudaMemcpy(F_buf->d, fpwf.data(), F_bytes,
                          cudaMemcpyHostToDevice), "H2D F");

    cublasHandle_t handle = cublas_handle();
    const cuDoubleComplex one  = make_cuDoubleComplex(1.0, 0.0);
    const cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);

    // T = Mx @ F  (N x npsf) = (N x npsf)(npsf x npsf)
    zgemm_rm(handle, N, npsf, npsf, Mx->d_M, F_buf->d, T_buf->d, one, zero);
    // O = T @ My  (N x N) = (N x npsf)(npsf x N)
    zgemm_rm(handle, N, N, npsf, T_buf->d, My->d_M, O_buf->d, one, zero);

    const double scale = psf_pixelscale_lamD / static_cast<double>(npix);
    const cuDoubleComplex scale_cplx = make_cuDoubleComplex(scale, 0.0);
    check_cublas(cublasZscal(handle, static_cast<int>(N * N),
                              &scale_cplx, O_buf->d, 1),
                 "cublasZscal mft_rev scale");

    Array2D<std::complex<double>> out(N, N, {0.0, 0.0});
    check_cuda(cudaMemcpy(out.data(), O_buf->d, O_bytes,
                          cudaMemcpyDeviceToHost), "D2H mft_rev");
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
