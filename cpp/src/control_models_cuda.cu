// Device-resident GPU forward for lina::ControlModel.
//
// The generic GPU primitives in props_cuda.cu each copy host->device,
// compute, then device->host. Chaining ~10 of them per optical "forward"
// (as ControlModel::forward_internal does) pays that transfer cost --
// plus several single-threaded host-side elementwise loops over 2048^2
// and 1200^2 arrays -- on every call. In an iEFC calibration that is
// thousands of forwards, which dominates runtime.
//
// This file implements ControlModel forward entirely on the device:
//   * model constants (DM MFT matrices, influence-function FFT, aperture,
//     Lyot stop, windowed vortex masks) are uploaded once and cached;
//   * every intermediate stays in device memory;
//   * only the final focal-plane field is copied back to the host.
//
// The math mirrors ControlModel::forward_internal (and the Python
// reference) exactly: same shift conventions, MFT coordinate/scale
// conventions, and dual-resolution (low-res FFT + high-res MFT) vortex
// propagation.

#include "lina/control_models.h"

#include <complex>
#include <cstddef>
#include <stdexcept>

#ifdef LINA_USE_CUDA

#include <cublas_v2.h>
#include <cuComplex.h>
#include <cufft.h>
#include <cuda_runtime.h>

#include <cmath>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace lina {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kTwoPi = 6.283185307179586476925286766559005768;

void check_cuda(cudaError_t s, const char* msg) {
    if (s != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(s));
    }
}
void check_cufft(cufftResult s, const char* msg) {
    if (s != CUFFT_SUCCESS) throw std::runtime_error(msg);
}
void check_cublas(cublasStatus_t s, const char* msg) {
    if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(msg);
}

// --------------------------------------------------------------------------
// Kernels
// --------------------------------------------------------------------------

__global__ void k_cmul_inplace(cuDoubleComplex* a, const cuDoubleComplex* b, std::size_t n) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    a[i] = cuCmul(a[i], b[i]);
}

__global__ void k_cmul(cuDoubleComplex* out, const cuDoubleComplex* a,
                       const cuDoubleComplex* b, std::size_t n) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = cuCmul(a[i], b[i]);
}

__global__ void k_cadd(cuDoubleComplex* out, const cuDoubleComplex* a,
                       const cuDoubleComplex* b, std::size_t n) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = cuCadd(a[i], b[i]);
}

__global__ void k_cscale(cuDoubleComplex* a, std::size_t n, double s) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    a[i] = make_cuDoubleComplex(a[i].x * s, a[i].y * s);
}

// out = a (complex) * r (real)
__global__ void k_real_mul(cuDoubleComplex* out, const cuDoubleComplex* a,
                           const double* r, std::size_t n) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double rr = r[i];
    out[i] = make_cuDoubleComplex(a[i].x * rr, a[i].y * rr);
}

// out = exp(i * coeff * Re(in))
__global__ void k_phasor_from_real(cuDoubleComplex* out, const cuDoubleComplex* in,
                                   std::size_t n, double coeff) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double phase = coeff * in[i].x;
    out[i] = make_cuDoubleComplex(cos(phase), sin(phase));
}

// out = ap * amp * exp(i * coeff * opd)   (entrance-pupil field)
__global__ void k_eep(cuDoubleComplex* out, const double* amp, const double* opd,
                      const double* ap, std::size_t n, double coeff) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double phase = coeff * opd[i];
    const double mag = ap[i] * amp[i];
    out[i] = make_cuDoubleComplex(mag * cos(phase), mag * sin(phase));
}

// Centered pad-or-crop, matching lina::pad_or_crop (out must be pre-zeroed).
__global__ void k_pad_or_crop(cuDoubleComplex* out, std::size_t outn,
                              const cuDoubleComplex* in, std::size_t inn) {
    const std::size_t c = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t r = blockIdx.y * blockDim.y + threadIdx.y;
    if (outn < inn) {
        // crop: out(r,c) = in(r+start, c+start)
        if (r >= outn || c >= outn) return;
        const std::size_t start = inn / 2 - outn / 2;
        out[r * outn + c] = in[(r + start) * inn + (c + start)];
    } else {
        // pad: out(r+start, c+start) = in(r,c)
        if (r >= inn || c >= inn) return;
        const std::size_t start = outn / 2 - inn / 2;
        out[(r + start) * outn + (c + start)] = in[r * inn + c];
    }
}

__global__ void shift2d_kernel(const cuDoubleComplex* __restrict__ in,
                               cuDoubleComplex* __restrict__ out,
                               std::size_t rows, std::size_t cols,
                               std::size_t shift_r, std::size_t shift_c) {
    const std::size_t c = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= rows || c >= cols) return;
    const std::size_t rr = (r + shift_r) % rows;
    const std::size_t cc = (c + shift_c) % cols;
    out[rr * cols + cc] = in[r * cols + c];
}

__global__ void shift2d_scaled_kernel(const cuDoubleComplex* __restrict__ in,
                                      cuDoubleComplex* __restrict__ out,
                                      std::size_t rows, std::size_t cols,
                                      std::size_t shift_r, std::size_t shift_c,
                                      double scale) {
    const std::size_t c = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= rows || c >= cols) return;
    const std::size_t rr = (r + shift_r) % rows;
    const std::size_t cc = (c + shift_c) % cols;
    const cuDoubleComplex v = in[r * cols + c];
    out[rr * cols + cc] = make_cuDoubleComplex(v.x * scale, v.y * scale);
}

__global__ void mft_matrix_kernel(cuDoubleComplex* __restrict__ M,
                                  const double* __restrict__ U,
                                  const double* __restrict__ X,
                                  std::size_t rows, std::size_t cols,
                                  double sign_two_pi) {
    const std::size_t b = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t a = blockIdx.y * blockDim.y + threadIdx.y;
    if (a >= rows || b >= cols) return;
    const double phase = sign_two_pi * U[a] * X[b];
    M[a * cols + b] = make_cuDoubleComplex(cos(phase), sin(phase));
}

inline dim3 grid1d(std::size_t n, unsigned block) {
    return dim3((static_cast<unsigned>(n) + block - 1) / block);
}
inline dim3 grid2d(std::size_t n, dim3 block) {
    return dim3((static_cast<unsigned>(n) + block.x - 1) / block.x,
                (static_cast<unsigned>(n) + block.y - 1) / block.y);
}

std::vector<double> mft_coords(std::size_t n, double pixelscale, const char* centering) {
    std::vector<double> out(n);
    const double half = static_cast<double>(n) / 2.0;
    const bool even = centering && std::string(centering) == "even";
    const double off = even ? 0.5 : 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = (static_cast<double>(i) - half + off) * pixelscale;
    }
    return out;
}

// Row-major device zgemm: C(m,n) = A(m,k) @ B(k,n).
void zgemm_rm(cublasHandle_t h, std::size_t m, std::size_t n, std::size_t k,
              const cuDoubleComplex* dA, const cuDoubleComplex* dB,
              cuDoubleComplex* dC) {
    const cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
    const cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);
    check_cublas(cublasZgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                             static_cast<int>(n), static_cast<int>(m), static_cast<int>(k),
                             &one, dB, static_cast<int>(n),
                             dA, static_cast<int>(k),
                             &zero, dC, static_cast<int>(n)),
                 "cublasZgemm failed");
}

// A device buffer of cuDoubleComplex, RAII.
struct DBuf {
    cuDoubleComplex* d = nullptr;
    std::size_t count = 0;
    void ensure(std::size_t n) {
        if (n <= count) return;
        if (d) cudaFree(d);
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d), sizeof(cuDoubleComplex) * n),
                   "cudaMalloc DBuf");
        count = n;
    }
    ~DBuf() { if (d) cudaFree(d); }
};

struct DBufR {
    double* d = nullptr;
    std::size_t count = 0;
    void ensure(std::size_t n) {
        if (n <= count) return;
        if (d) cudaFree(d);
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d), sizeof(double) * n),
                   "cudaMalloc DBufR");
        count = n;
    }
    ~DBufR() { if (d) cudaFree(d); }
};

cuDoubleComplex* as_cplx(const std::complex<double>* p) {
    return reinterpret_cast<cuDoubleComplex*>(const_cast<std::complex<double>*>(p));
}

} // namespace

// ===========================================================================
// Persistent device state for one ControlModel.
// ===========================================================================
struct ControlModelGpuState {
    cublasHandle_t cublas = nullptr;

    // Constant device arrays (uploaded once).
    DBuf d_mx_dm, d_my_dm;        // (nsurf x nact), (nact x nsurf)
    DBuf d_inf_fft;               // (nsurf x nsurf)
    DBufR d_aperture, d_lyot;     // (ndef x ndef)
    DBuf d_wv_lres, d_wv_hres;    // (nlres^2), (nhres^2)

    // Per-call prefpm (re-uploaded; cheap, and changes via setters).
    DBufR d_prefpm_amp, d_prefpm_opd;

    // Scratch (sized lazily, reused across calls).
    DBuf d_dmcmd;                 // nact^2
    DBuf d_tmp_na;                // nsurf*nact
    DBuf s_surfA, s_surfB, s_surfC; // nsurf^2
    DBuf d_ep, d_dmphasor_pad, d_dm, d_lp, d_lp_lres, d_lp_hres, d_ls; // ndef^2
    DBuf s_lresA, s_lresB, s_lresC; // nlres^2
    DBuf s_hresA, s_hresB;        // nhres^2
    DBuf d_efp;                   // ncamsci^2
    DBuf s_mftT;                  // gemm temp (max of stage T sizes)

    // FFT plans by square size.
    std::unordered_map<std::size_t, cufftHandle> plans;

    // MFT matrices (built lazily, cached). Device pointers + sizes.
    DBuf hres_fwd_Mx, hres_fwd_My;   // (nhres x ndef), (ndef x nhres)
    DBuf hres_rev_Mx, hres_rev_My;   // (ndef x nhres), (nhres x ndef)
    bool hres_built = false;
    DBuf focal_Mx, focal_My;         // (ncamsci x ndef), (ndef x ncamsci)
    double focal_wl = -1.0;          // wavelength the focal matrices were built for

    cufftHandle plan_for(std::size_t n) {
        auto it = plans.find(n);
        if (it != plans.end()) return it->second;
        cufftHandle p;
        check_cufft(cufftPlan2d(&p, static_cast<int>(n), static_cast<int>(n), CUFFT_Z2Z),
                    "cufftPlan2d");
        plans.emplace(n, p);
        return p;
    }

    ControlModelGpuState() {
        check_cublas(cublasCreate(&cublas), "cublasCreate");
    }
    ~ControlModelGpuState() {
        for (auto& kv : plans) cufftDestroy(kv.second);
        if (cublas) cublasDestroy(cublas);
    }
    ControlModelGpuState(const ControlModelGpuState&) = delete;
    ControlModelGpuState& operator=(const ControlModelGpuState&) = delete;
};

namespace {

// In-place centered FFT on a device buffer of size n x n, using two
// scratch buffers. Matches lina's shift convention.
void fft_device(ControlModelGpuState& st, cuDoubleComplex* d_buf, std::size_t n,
                bool inverse, cuDoubleComplex* tmp) {
    const std::size_t pre_r = inverse ? (n + 1) / 2 : n / 2;
    const std::size_t post_r = inverse ? n / 2 : (n + 1) / 2;
    const dim3 block(16, 16);
    const dim3 grid = grid2d(n, block);

    // pre-shift d_buf -> tmp
    shift2d_kernel<<<grid, block>>>(d_buf, tmp, n, n, pre_r, pre_r);
    cufftHandle plan = st.plan_for(n);
    check_cufft(cufftExecZ2Z(plan, tmp, tmp, inverse ? CUFFT_INVERSE : CUFFT_FORWARD),
                "cufftExecZ2Z");
    // post-shift tmp -> d_buf (+ 1/N scale for inverse)
    if (inverse) {
        const double norm = 1.0 / static_cast<double>(n * n);
        shift2d_scaled_kernel<<<grid, block>>>(tmp, d_buf, n, n, post_r, post_r, norm);
    } else {
        shift2d_kernel<<<grid, block>>>(tmp, d_buf, n, n, post_r, post_r);
    }
}

void build_mft_matrix(ControlModelGpuState& st, DBuf& M,
                      const std::vector<double>& U, const std::vector<double>& X,
                      double sign, DBufR& du, DBufR& dx) {
    M.ensure(U.size() * X.size());
    du.ensure(U.size());
    dx.ensure(X.size());
    check_cuda(cudaMemcpy(du.d, U.data(), sizeof(double) * U.size(), cudaMemcpyHostToDevice),
               "H2D mft U");
    check_cuda(cudaMemcpy(dx.d, X.data(), sizeof(double) * X.size(), cudaMemcpyHostToDevice),
               "H2D mft X");
    const dim3 block(16, 16);
    const dim3 grid((static_cast<unsigned>(X.size()) + block.x - 1) / block.x,
                    (static_cast<unsigned>(U.size()) + block.y - 1) / block.y);
    mft_matrix_kernel<<<grid, block>>>(M.d, du.d, dx.d, U.size(), X.size(), sign * kTwoPi);
    check_cuda(cudaGetLastError(), "mft_matrix_kernel");
}

// MFT forward: d_out(npsf x npsf) = (Mx @ W @ My) * (du/npix), W = d_in(N x N).
// `npix` is a sampling scale (dx = 1/npix), kept fractional to match Python.
void mft_forward_dev(ControlModelGpuState& st, cuDoubleComplex* d_in, std::size_t N,
                     double npix, std::size_t npsf, double du,
                     const DBuf& Mx, const DBuf& My, cuDoubleComplex* d_out) {
    st.s_mftT.ensure(npsf * N);
    // T = Mx(npsf x N) @ W(N x N) -> (npsf x N)
    zgemm_rm(st.cublas, npsf, N, N, Mx.d, d_in, st.s_mftT.d);
    // O = T(npsf x N) @ My(N x npsf) -> (npsf x npsf)
    zgemm_rm(st.cublas, npsf, npsf, N, st.s_mftT.d, My.d, d_out);
    const double scale = du / npix;
    k_cscale<<<grid1d(npsf * npsf, 256), 256>>>(d_out, npsf * npsf, scale);
}

// MFT reverse: d_out(N x N) = (Mx @ F @ My) * (du/npix), F = d_in(npsf x npsf).
void mft_reverse_dev(ControlModelGpuState& st, cuDoubleComplex* d_in, std::size_t npsf,
                     double npix, std::size_t N, double du,
                     const DBuf& Mx, const DBuf& My, cuDoubleComplex* d_out) {
    st.s_mftT.ensure(N * npsf);
    // T = Mx(N x npsf) @ F(npsf x npsf) -> (N x npsf)
    zgemm_rm(st.cublas, N, npsf, npsf, Mx.d, d_in, st.s_mftT.d);
    // O = T(N x npsf) @ My(npsf x N) -> (N x N)
    zgemm_rm(st.cublas, N, N, npsf, st.s_mftT.d, My.d, d_out);
    const double scale = du / npix;
    k_cscale<<<grid1d(N * N, 256), 256>>>(d_out, N * N, scale);
}

} // namespace

// ===========================================================================
// Public entry point.
// ===========================================================================
Array2D<std::complex<double>> control_model_forward_gpu(
    const ControlModel& model,
    const std::vector<double>& actuators,
    double wavelength,
    bool use_vortex) {

    // The angular-spectrum exit-pupil propagation is not yet ported to the
    // device-resident path. It is unused by the standard coronagraph model;
    // require CPU device for that rare configuration.
    if (model.exit_pupil_prop_dist_.has_value()) {
        throw std::runtime_error(
            "control_model_forward_gpu: exit_pupil_prop_dist is not supported on "
            "the GPU device-resident path; use set_device('cpu') for that model.");
    }

    const std::size_t nact = model.nact_;
    const std::size_t nsurf = model.nsurf_;
    const std::size_t ndef = model.ndef_;
    const std::size_t nlres = model.n_vortex_lres_;
    const std::size_t nhres = model.n_vortex_hres_;
    const std::size_t ncam = model.ncamsci_;

    if (!model.gpu_state_) {
        model.gpu_state_ = std::make_shared<ControlModelGpuState>();
        auto& st = *model.gpu_state_;
        // Upload constants once.
        st.d_mx_dm.ensure(nsurf * nact);
        st.d_my_dm.ensure(nact * nsurf);
        st.d_inf_fft.ensure(nsurf * nsurf);
        st.d_wv_lres.ensure(nlres * nlres);
        st.d_wv_hres.ensure(nhres * nhres);
        st.d_aperture.ensure(ndef * ndef);
        st.d_lyot.ensure(ndef * ndef);
        check_cuda(cudaMemcpy(st.d_mx_dm.d, as_cplx(model.mx_dm_.data()),
                              sizeof(cuDoubleComplex) * nsurf * nact, cudaMemcpyHostToDevice), "H2D mx");
        check_cuda(cudaMemcpy(st.d_my_dm.d, as_cplx(model.my_dm_.data()),
                              sizeof(cuDoubleComplex) * nact * nsurf, cudaMemcpyHostToDevice), "H2D my");
        check_cuda(cudaMemcpy(st.d_inf_fft.d, as_cplx(model.inf_fun_fft_.data()),
                              sizeof(cuDoubleComplex) * nsurf * nsurf, cudaMemcpyHostToDevice), "H2D inf_fft");
        check_cuda(cudaMemcpy(st.d_wv_lres.d, as_cplx(model.windowed_vortex_lres_.data()),
                              sizeof(cuDoubleComplex) * nlres * nlres, cudaMemcpyHostToDevice), "H2D wv_lres");
        check_cuda(cudaMemcpy(st.d_wv_hres.d, as_cplx(model.windowed_vortex_hres_.data()),
                              sizeof(cuDoubleComplex) * nhres * nhres, cudaMemcpyHostToDevice), "H2D wv_hres");
        check_cuda(cudaMemcpy(st.d_aperture.d, model.aperture_.data(),
                              sizeof(double) * ndef * ndef, cudaMemcpyHostToDevice), "H2D aperture");
        check_cuda(cudaMemcpy(st.d_lyot.d, model.lyotstop_.data(),
                              sizeof(double) * ndef * ndef, cudaMemcpyHostToDevice), "H2D lyot");
    }
    auto& st = *model.gpu_state_;

    // Per-call prefpm upload (cheap; changes via set_prefpm_*).
    st.d_prefpm_amp.ensure(ndef * ndef);
    st.d_prefpm_opd.ensure(ndef * ndef);
    check_cuda(cudaMemcpy(st.d_prefpm_amp.d, model.prefpm_amp_.data(),
                          sizeof(double) * ndef * ndef, cudaMemcpyHostToDevice), "H2D prefpm_amp");
    check_cuda(cudaMemcpy(st.d_prefpm_opd.d, model.prefpm_opd_.data(),
                          sizeof(double) * ndef * ndef, cudaMemcpyHostToDevice), "H2D prefpm_opd");

    // Build DM command (nact x nact) complex on host (tiny), upload.
    std::vector<std::complex<double>> dmcmd(nact * nact, {0.0, 0.0});
    {
        std::size_t idx = 0;
        for (std::size_t i = 0; i < model.dm_mask_.size(); ++i) {
            if (model.dm_mask_.data()[i]) dmcmd[i] = actuators[idx++];
        }
    }
    st.d_dmcmd.ensure(nact * nact);
    check_cuda(cudaMemcpy(st.d_dmcmd.d, as_cplx(dmcmd.data()),
                          sizeof(cuDoubleComplex) * nact * nact, cudaMemcpyHostToDevice), "H2D dmcmd");

    // Scratch sizing.
    st.d_tmp_na.ensure(nsurf * nact);
    st.s_surfA.ensure(nsurf * nsurf);
    st.s_surfB.ensure(nsurf * nsurf);
    st.s_surfC.ensure(nsurf * nsurf);
    st.d_ep.ensure(ndef * ndef);
    st.d_dmphasor_pad.ensure(ndef * ndef);
    st.d_dm.ensure(ndef * ndef);
    st.d_lp.ensure(ndef * ndef);
    st.d_lp_lres.ensure(ndef * ndef);
    st.d_lp_hres.ensure(ndef * ndef);
    st.d_ls.ensure(ndef * ndef);
    st.d_efp.ensure(ncam * ncam);

    const unsigned B = 256;

    // 1) mft_command = mx_dm @ dmcmd @ my_dm  (nsurf x nsurf), in s_surfA.
    zgemm_rm(st.cublas, nsurf, nact, nact, st.d_mx_dm.d, st.d_dmcmd.d, st.d_tmp_na.d);
    zgemm_rm(st.cublas, nsurf, nsurf, nact, st.d_tmp_na.d, st.d_my_dm.d, st.s_surfA.d);

    // 2) fourier_surf = inf_fft * mft_command  -> s_surfB
    k_cmul<<<grid1d(nsurf * nsurf, B), B>>>(st.s_surfB.d, st.d_inf_fft.d, st.s_surfA.d, nsurf * nsurf);

    // 3) dm_surf_c = ifft(fourier_surf)  (in-place on s_surfB, scratch s_surfC)
    fft_device(st, st.s_surfB.d, nsurf, /*inverse=*/true, st.s_surfC.d);

    // 4) dm_phasor = exp(i * 4pi/wl * Re(dm_surf_c)) -> s_surfA  (nsurf^2)
    k_phasor_from_real<<<grid1d(nsurf * nsurf, B), B>>>(
        st.s_surfA.d, st.s_surfB.d, nsurf * nsurf, 4.0 * kPi / wavelength);

    // 5) dm_phasor_pad = pad_or_crop(dm_phasor, ndef)
    check_cuda(cudaMemset(st.d_dmphasor_pad.d, 0, sizeof(cuDoubleComplex) * ndef * ndef),
               "memset dmphasor_pad");
    {
        const std::size_t mx = (ndef > nsurf) ? ndef : nsurf;
        const dim3 block(16, 16);
        k_pad_or_crop<<<grid2d(mx, block), block>>>(st.d_dmphasor_pad.d, ndef, st.s_surfA.d, nsurf);
    }

    // 6) e_ep = aperture * prefpm_amp * exp(i * 2pi/wl * prefpm_opd)  (ndef^2)
    k_eep<<<grid1d(ndef * ndef, B), B>>>(st.d_ep.d, st.d_prefpm_amp.d, st.d_prefpm_opd.d,
                                         st.d_aperture.d, ndef * ndef, 2.0 * kPi / wavelength);

    // 7) e_dm = e_ep * dm_phasor_pad
    k_cmul<<<grid1d(ndef * ndef, B), B>>>(st.d_dm.d, st.d_ep.d, st.d_dmphasor_pad.d, ndef * ndef);

    if (use_vortex) {
        // --- low-res FFT branch ---
        st.s_lresA.ensure(nlres * nlres);
        st.s_lresB.ensure(nlres * nlres);
        st.s_lresC.ensure(nlres * nlres);
        check_cuda(cudaMemset(st.s_lresA.d, 0, sizeof(cuDoubleComplex) * nlres * nlres),
                   "memset lres");
        {
            const dim3 block(16, 16);
            k_pad_or_crop<<<grid2d(nlres, block), block>>>(st.s_lresA.d, nlres, st.d_dm.d, ndef);
        }
        fft_device(st, st.s_lresA.d, nlres, /*inverse=*/false, st.s_lresB.d);
        k_cmul_inplace<<<grid1d(nlres * nlres, B), B>>>(st.s_lresA.d, st.d_wv_lres.d, nlres * nlres);
        fft_device(st, st.s_lresA.d, nlres, /*inverse=*/true, st.s_lresB.d);
        // crop to ndef -> d_lp_lres
        {
            const dim3 block(16, 16);
            const std::size_t mx = (ndef > nlres) ? ndef : nlres;
            k_pad_or_crop<<<grid2d(mx, block), block>>>(st.d_lp_lres.d, ndef, st.s_lresA.d, nlres);
        }

        // --- high-res MFT branch ---
        st.s_hresA.ensure(nhres * nhres);
        st.s_hresB.ensure(nhres * nhres);
        if (!st.hres_built) {
            DBufR du, dx;
            const auto Xs = mft_coords(ndef, 1.0 / static_cast<double>(model.npix_), "odd");
            const auto Us = mft_coords(nhres, model.hres_sampling_, "odd");
            // forward ('-'): Mx=build(Us,Xs), My=build(Xs,Us)
            build_mft_matrix(st, st.hres_fwd_Mx, Us, Xs, -1.0, du, dx);
            build_mft_matrix(st, st.hres_fwd_My, Xs, Us, -1.0, du, dx);
            // reverse ('+'): Mx=build(Xs,Us), My=build(Us,Xs)
            build_mft_matrix(st, st.hres_rev_Mx, Xs, Us, 1.0, du, dx);
            build_mft_matrix(st, st.hres_rev_My, Us, Xs, 1.0, du, dx);
            st.hres_built = true;
        }
        // e_fpm_hres = mft_forward(e_dm, npix_, nhres, hres_sampling) -> s_hresA
        mft_forward_dev(st, st.d_dm.d, ndef, model.npix_, nhres, model.hres_sampling_,
                        st.hres_fwd_Mx, st.hres_fwd_My, st.s_hresA.d);
        k_cmul_inplace<<<grid1d(nhres * nhres, B), B>>>(st.s_hresA.d, st.d_wv_hres.d, nhres * nhres);
        // e_lp_hres = mft_reverse(e_fpm_hres, hres_sampling, npix_, ndef) -> d_lp_hres
        mft_reverse_dev(st, st.s_hresA.d, nhres, model.npix_, ndef, model.hres_sampling_,
                        st.hres_rev_Mx, st.hres_rev_My, st.d_lp_hres.d);

        // e_lp = e_lp_lres + e_lp_hres
        k_cadd<<<grid1d(ndef * ndef, B), B>>>(st.d_lp.d, st.d_lp_lres.d, st.d_lp_hres.d, ndef * ndef);
    } else {
        check_cuda(cudaMemcpy(st.d_lp.d, st.d_dm.d, sizeof(cuDoubleComplex) * ndef * ndef,
                              cudaMemcpyDeviceToDevice), "D2D e_lp");
    }

    // e_ls = lyotstop * e_lp
    k_real_mul<<<grid1d(ndef * ndef, B), B>>>(st.d_ls.d, st.d_lp.d, st.d_lyot.d, ndef * ndef);

    // Focal-plane MFT. Rebuild matrices if wavelength changed.
    const double camsci_pxscl_lamD = model.camsci_pxscl_lamDc_ * model.wavelength_c_ / wavelength;
    // npix is a fractional sampling scale (dx = 1/npix), not an array size.
    const double npix_eff = static_cast<double>(model.npix_) * model.lyot_ratio_;
    if (st.focal_wl != wavelength) {
        DBufR du, dx;
        const auto Xs = mft_coords(ndef, 1.0 / npix_eff, "odd");
        const auto Us = mft_coords(ncam, camsci_pxscl_lamD, "odd");
        build_mft_matrix(st, st.focal_Mx, Us, Xs, -1.0, du, dx);
        build_mft_matrix(st, st.focal_My, Xs, Us, -1.0, du, dx);
        st.focal_wl = wavelength;
    }
    mft_forward_dev(st, st.d_ls.d, ndef, npix_eff, ncam, camsci_pxscl_lamD,
                    st.focal_Mx, st.focal_My, st.d_efp.d);

    // Copy focal-plane field back to host.
    Array2D<std::complex<double>> out(ncam, ncam, {0.0, 0.0});
    check_cuda(cudaDeviceSynchronize(), "device sync forward_gpu");
    check_cuda(cudaMemcpy(as_cplx(out.data()), st.d_efp.d,
                          sizeof(cuDoubleComplex) * ncam * ncam, cudaMemcpyDeviceToHost),
               "D2H e_fp");
    return out;
}

} // namespace lina

#else  // !LINA_USE_CUDA

namespace lina {
Array2D<std::complex<double>> control_model_forward_gpu(
    const ControlModel&,
    const std::vector<double>&,
    double,
    bool) {
    throw std::runtime_error(
        "control_model_forward_gpu unavailable: build with LINA_USE_CUDA=ON");
}
} // namespace lina

#endif
