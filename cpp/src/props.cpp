#include "lina/props.h"
#include "lina/grid2d.h"
#include "lina/utils.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef LINA_USE_FFTW
#include <fftw3.h>
#endif

#ifdef LINA_USE_OPENBLAS
#include <cblas.h>
#endif

namespace lina {
namespace {

constexpr double kTwoPi = 2.0 * M_PI;

// ---------------------------------------------------------------------------
// MFT matrix cache.
//
// Building M_pre / M_post is O(npix * npsf) complex exponentials. In a
// typical wavefront-control loop the *geometry* (npix, npsf, du,
// convention, centering) stays fixed across hundreds of iterations and
// only the wavefront changes, so caching the M matrices turns
// `mft_forward` into "two zgemms" -- matching the numpy reference that
// precomputes the matrices once outside its timed loop.
//
// The cache is keyed on every parameter that affects the matrix, has a
// modest LRU-style cap so it cannot grow unbounded in a long-running
// process, and is guarded by a mutex so concurrent threads do not race
// on the std::unordered_map.
// ---------------------------------------------------------------------------

struct MftKey {
    std::size_t npix;
    std::size_t npsf;
    std::size_t N;          // wavefront side length (for mft_forward) or
                            // pupil side length (for mft_reverse)
    double du;              // psf_pixelscale_lamD
    int sign;               // +1 or -1
    int role;               // 0=mft_forward pre, 1=mft_forward post,
                            // 2=mft_reverse Mx, 3=mft_reverse My
    char pp_first;          // first char of pp_centering ('o' or 'e')
    char fp_first;          // first char of fp_centering ('o' or 'e')

    bool operator==(const MftKey& o) const noexcept {
        return npix == o.npix && npsf == o.npsf && N == o.N
            && du == o.du && sign == o.sign && role == o.role
            && pp_first == o.pp_first && fp_first == o.fp_first;
    }
};

struct MftKeyHash {
    std::size_t operator()(const MftKey& k) const noexcept {
        // Mix the integral fields; double goes through std::hash.
        std::size_t h = std::hash<std::size_t>{}(k.npix);
        h ^= std::hash<std::size_t>{}(k.npsf) + 0x9e3779b97f4a7c15ULL
             + (h << 6) + (h >> 2);
        h ^= std::hash<std::size_t>{}(k.N)    + 0x9e3779b97f4a7c15ULL
             + (h << 6) + (h >> 2);
        h ^= std::hash<double>{}(k.du)        + 0x9e3779b97f4a7c15ULL
             + (h << 6) + (h >> 2);
        const std::size_t packed = static_cast<std::size_t>(k.sign + 2)
            | (static_cast<std::size_t>(k.role) << 4)
            | (static_cast<std::size_t>(k.pp_first) << 8)
            | (static_cast<std::size_t>(k.fp_first) << 16);
        h ^= packed + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h;
    }
};

using ComplexMatrix = Array2D<std::complex<double>>;
using ComplexMatrixPtr = std::shared_ptr<const ComplexMatrix>;

static std::mutex g_mft_cache_mu;
static std::unordered_map<MftKey, ComplexMatrixPtr, MftKeyHash> g_mft_cache;
constexpr std::size_t kMftCacheCap = 32;  // ~32 distinct geometries max

// Returns a shared_ptr so eviction does not free a matrix that another
// thread is still reading. The map holds one strong reference; each
// caller keeps another until it finishes its zgemm.
ComplexMatrixPtr mft_cache_get_or_build(
    const MftKey& key,
    std::size_t rows, std::size_t cols,
    double sign,
    const std::vector<double>& row_coords,
    const std::vector<double>& col_coords) {
    std::lock_guard<std::mutex> lk(g_mft_cache_mu);
    auto it = g_mft_cache.find(key);
    if (it != g_mft_cache.end()) {
        return it->second;
    }
    if (g_mft_cache.size() >= kMftCacheCap) {
        g_mft_cache.erase(g_mft_cache.begin());
    }
    auto M = std::make_shared<ComplexMatrix>(rows, cols,
                                             std::complex<double>{0.0, 0.0});
    const std::complex<double> j(0.0, 1.0);
    for (std::size_t r = 0; r < rows; ++r) {
        const double rc = row_coords[r];
        for (std::size_t c = 0; c < cols; ++c) {
            const double phase = sign * kTwoPi * rc * col_coords[c];
            (*M)(r, c) = std::exp(j * phase);
        }
    }
    auto [iter, inserted] = g_mft_cache.emplace(key, M);
    return iter->second;
}

void shift_into_buffer(const std::complex<double>* src,
                       std::complex<double>* dst,
                       std::size_t rows,
                       std::size_t cols,
                       std::size_t shift_r,
                       std::size_t shift_c) {
    for (std::size_t r = 0; r < rows; ++r) {
        const std::size_t rr = (r + shift_r) % rows;
        const std::complex<double>* src_row = src + r * cols;
        std::complex<double>* dst_row = dst + rr * cols;
        const std::size_t right = cols - shift_c;
        if (shift_c == 0) {
            std::copy(src_row, src_row + cols, dst_row);
        } else {
            std::copy(src_row, src_row + right, dst_row + shift_c);
            std::copy(src_row + right, src_row + cols, dst_row);
        }
    }
}

Array2D<std::complex<double>> shift2d(const Array2D<std::complex<double>>& in,
                                      std::size_t shift_r,
                                      std::size_t shift_c) {
    const std::size_t rows = in.rows();
    const std::size_t cols = in.cols();
    Array2D<std::complex<double>> out(rows, cols, {0.0, 0.0});
    shift_into_buffer(in.data(), out.data(), rows, cols, shift_r, shift_c);
    return out;
}

Array2D<std::complex<double>> fftshift(const Array2D<std::complex<double>>& in) {
    return shift2d(in, in.rows() / 2, in.cols() / 2);
}

Array2D<std::complex<double>> ifftshift(const Array2D<std::complex<double>>& in) {
    return shift2d(in, (in.rows() + 1) / 2, (in.cols() + 1) / 2);
}

#ifdef LINA_USE_FFTW
struct FftwPlanKey {
    std::size_t rows;
    std::size_t cols;
    bool inverse;

    bool operator==(const FftwPlanKey& other) const {
        return rows == other.rows && cols == other.cols && inverse == other.inverse;
    }
};

struct FftwPlanKeyHash {
    std::size_t operator()(const FftwPlanKey& key) const noexcept {
        return std::hash<std::size_t>()(key.rows) ^
               (std::hash<std::size_t>()(key.cols) << 1) ^
               (std::hash<bool>()(key.inverse) << 2);
    }
};

class FftwContext {
public:
    FftwContext() {
#ifdef LINA_USE_FFTW_THREADS
        fftw_init_threads();
#endif
        const char* threads_env = std::getenv("LINA_FFTW_THREADS");
        if (threads_env) {
            threads_ = std::max(1, std::atoi(threads_env));
        }
#ifdef LINA_USE_FFTW_THREADS
        fftw_plan_with_nthreads(threads_);
#endif
        const char* wisdom_dir = std::getenv("LINA_FFTW_WISDOM_DIR");
        if (wisdom_dir) {
            wisdom_dir_ = wisdom_dir;
        }
        const char* wisdom_env = std::getenv("LINA_FFTW_WISDOM_PATH");
        if (wisdom_env) {
            wisdom_path_ = wisdom_env;
        }
        const char* plan_env = std::getenv("LINA_FFTW_PLAN");
        if (plan_env && std::string(plan_env) == "MEASURE") {
            plan_flags_ = FFTW_MEASURE;
        }
    }

    ~FftwContext() {
        for (const auto& entry : plans_) {
            fftw_destroy_plan(entry.second);
        }
        for (const auto& entry : buffers_) {
            fftw_free(entry.second.data);
        }
#ifdef LINA_USE_FFTW_THREADS
        fftw_cleanup_threads();
#endif
    }

    fftw_plan get_plan(std::size_t rows, std::size_t cols, bool inverse) {
        std::lock_guard<std::mutex> lock(mutex_);
        const FftwPlanKey key{rows, cols, inverse};
        const auto it = plans_.find(key);
        if (it != plans_.end()) {
            return it->second;
        }

        const std::string wisdom_path = wisdom_path_for(rows, cols);
        if (!wisdom_path.empty()) {
            fftw_import_wisdom_from_filename(wisdom_path.c_str());
        }

        // The plan must be created in-place because fft_cpu/ifft_cpu call
        // fftw_execute_dft(plan, buffer, buffer) (same in and out). Per the
        // FFTW docs (New-array Execute Functions): "in-place plans must be
        // used in-place and out-of-place plans must be used out-of-place."
        // Using an out-of-place plan in-place silently produces garbage for
        // prime-factor sizes (e.g. odd N >= 33).
        std::vector<std::complex<double>> dummy(rows * cols);
        const int dir = inverse ? FFTW_BACKWARD : FFTW_FORWARD;
        fftw_plan plan = fftw_plan_dft_2d(
            static_cast<int>(rows),
            static_cast<int>(cols),
            reinterpret_cast<fftw_complex*>(dummy.data()),
            reinterpret_cast<fftw_complex*>(dummy.data()),
            dir,
            plan_flags_);
        plans_.emplace(key, plan);
        if (!wisdom_path.empty()) {
            fftw_export_wisdom_to_filename(wisdom_path.c_str());
        }
        return plan;
    }

    fftw_complex* get_buffer(std::size_t rows, std::size_t cols) {
        const std::size_t key = (rows << 32) ^ cols;
        const std::size_t count = rows * cols;
        const auto it = buffers_.find(key);
        if (it != buffers_.end() && it->second.count == count) {
            return it->second.data;
        }
        FftwBuffer buf{};
        buf.count = count;
        buf.data = reinterpret_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * count));
        if (!buf.data) {
            throw std::runtime_error("fftw_malloc failed");
        }
        buffers_[key] = buf;
        return buf.data;
    }

private:
    struct FftwBuffer {
        fftw_complex* data = nullptr;
        std::size_t count = 0;
    };

    std::string wisdom_path_for(std::size_t rows, std::size_t cols) const {
        if (!wisdom_dir_.empty()) {
            std::ostringstream oss;
            oss << wisdom_dir_ << "/fftw_wisdom_" << rows << "x" << cols << ".dat";
            return oss.str();
        }
        return wisdom_path_;
    }

    std::unordered_map<FftwPlanKey, fftw_plan, FftwPlanKeyHash> plans_;
    std::unordered_map<std::size_t, FftwBuffer> buffers_;
    std::mutex mutex_;
    int threads_ = 1;
    int plan_flags_ = FFTW_ESTIMATE;
    std::string wisdom_path_;
    std::string wisdom_dir_;
};

FftwContext& fftw_context() {
    static FftwContext ctx;
    return ctx;
}
#endif

Array2D<std::complex<double>> fft2d_naive(const Array2D<std::complex<double>>& in,
                                          bool inverse) {
    const std::size_t rows = in.rows();
    const std::size_t cols = in.cols();
    Array2D<std::complex<double>> out(rows, cols, {0.0, 0.0});

    const double norm = inverse ? 1.0 / static_cast<double>(rows * cols) : 1.0;
    const double sign = inverse ? 1.0 : -1.0;

    for (std::size_t u = 0; u < rows; ++u) {
        for (std::size_t v = 0; v < cols; ++v) {
            std::complex<double> sum(0.0, 0.0);
            for (std::size_t x = 0; x < rows; ++x) {
                for (std::size_t y = 0; y < cols; ++y) {
                    const double angle =
                        sign * kTwoPi *
                        (static_cast<double>(u) * x / rows +
                         static_cast<double>(v) * y / cols);
                    sum += in(x, y) * std::exp(std::complex<double>(0.0, angle));
                }
            }
            out(u, v) = sum * norm;
        }
    }
    return out;
}

std::vector<double> build_coordinates(std::size_t n,
                                      double scale,
                                      const char* centering) {
    std::vector<double> coords(n, 0.0);
    if (std::string(centering) == "even") {
        for (std::size_t i = 0; i < n; ++i) {
            coords[i] = (static_cast<double>(i) - static_cast<double>(n) / 2.0 + 0.5) * scale;
        }
    } else {
        for (std::size_t i = 0; i < n; ++i) {
            coords[i] = (static_cast<double>(i) - static_cast<double>(n) / 2.0) * scale;
        }
    }
    return coords;
}

} // namespace

Array2D<std::complex<double>> fft_cpu(const Array2D<std::complex<double>>& arr) {
#ifdef LINA_USE_FFTW
    const std::size_t rows = arr.rows();
    const std::size_t cols = arr.cols();
    Array2D<std::complex<double>> out(rows, cols, {0.0, 0.0});
    fftw_complex* buffer = fftw_context().get_buffer(rows, cols);
    shift_into_buffer(arr.data(),
                      reinterpret_cast<std::complex<double>*>(buffer),
                      rows, cols, rows / 2, cols / 2);
    fftw_plan plan = fftw_context().get_plan(rows, cols, false);
    fftw_execute_dft(plan, buffer, buffer);
    shift_into_buffer(reinterpret_cast<std::complex<double>*>(buffer),
                      out.data(),
                      rows, cols, (rows + 1) / 2, (cols + 1) / 2);
    return out;
#else
    const auto shifted = fftshift(arr);
    const auto freq = fft2d_naive(shifted, false);
    return ifftshift(freq);
#endif
}

Array2D<std::complex<double>> ifft_cpu(const Array2D<std::complex<double>>& arr) {
#ifdef LINA_USE_FFTW
    const std::size_t rows = arr.rows();
    const std::size_t cols = arr.cols();
    Array2D<std::complex<double>> out(rows, cols, {0.0, 0.0});
    fftw_complex* buffer = fftw_context().get_buffer(rows, cols);
    shift_into_buffer(arr.data(),
                      reinterpret_cast<std::complex<double>*>(buffer),
                      rows, cols, (rows + 1) / 2, (cols + 1) / 2);
    fftw_plan plan = fftw_context().get_plan(rows, cols, true);
    fftw_execute_dft(plan, buffer, buffer);
    auto buffer_c = reinterpret_cast<std::complex<double>*>(buffer);
    const double norm = 1.0 / static_cast<double>(rows * cols);
    for (std::size_t i = 0; i < rows * cols; ++i) {
        buffer_c[i] *= norm;
    }
    shift_into_buffer(buffer_c, out.data(), rows, cols, rows / 2, cols / 2);
    return out;
#else
    const auto shifted = ifftshift(arr);
    const auto spatial = fft2d_naive(shifted, true);
    return fftshift(spatial);
#endif
}

Array2D<std::complex<double>> fft(const Array2D<std::complex<double>>& arr) {
    return fft_cpu(arr);
}

Array2D<std::complex<double>> ifft(const Array2D<std::complex<double>>& arr) {
    return ifft_cpu(arr);
}

Array2D<std::complex<double>> ang_spec(const Array2D<std::complex<double>>& wavefront,
                                       double wavelength,
                                       double distance,
                                       double pixelscale) {
    const std::size_t n = wavefront.rows();
    const double delkx = kTwoPi / (static_cast<double>(n) * pixelscale);

    std::vector<double> kxy(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        kxy[i] = (static_cast<double>(i) - static_cast<double>(n) / 2.0 + 0.5) * delkx;
    }

    Array2D<std::complex<double>> wf_as = fft(wavefront);

    Array2D<std::complex<double>> out(n, n, {0.0, 0.0});
    const double k = kTwoPi / wavelength;
    for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t c = 0; c < n; ++c) {
            const double kx = kxy[c];
            const double ky = kxy[r];
            const std::complex<double> kz = std::sqrt(std::complex<double>(k * k - kx * kx - ky * ky, 0.0));
            const std::complex<double> tf = std::exp(std::complex<double>(0.0, 1.0) * kz * distance);
            out(r, c) = wf_as(r, c) * tf;
        }
    }

    return ifft(out);
}

Array2D<std::complex<double>> make_vortex_phase_mask(std::size_t npix,
                                                     int charge,
                                                     const char* grid) {
    Array2D<std::complex<double>> phasor(npix, npix, {0.0, 0.0});
    const double half = static_cast<double>(npix) / 2.0;
    const double offset = (std::string(grid) == "even") ? 0.5 : 0.0;
    for (std::size_t r = 0; r < npix; ++r) {
        for (std::size_t c = 0; c < npix; ++c) {
            const double x = static_cast<double>(c) - half + offset;
            const double y = static_cast<double>(r) - half + offset;
            const double th = std::atan2(y, x);
            const double phase = static_cast<double>(charge) * th;
            phasor(r, c) = std::exp(std::complex<double>(0.0, phase));
        }
    }
    return phasor;
}

Array2D<std::complex<double>> mft_forward(
    const Array2D<std::complex<double>>& wavefront,
    std::size_t npix,
    std::size_t npsf,
    double psf_pixelscale_lamD,
    char convention,
    const char* pp_centering,
    const char* fp_centering) {

    const std::size_t N = wavefront.rows();
    if (N != wavefront.cols()) {
        throw std::invalid_argument("mft_forward expects square wavefront");
    }

    const double dx = 1.0 / static_cast<double>(npix);
    const double du = psf_pixelscale_lamD;

    const auto Xs = build_coordinates(N, dx, pp_centering);
    const auto Us = build_coordinates(npsf, du, fp_centering);

    const double sign = (convention == '-') ? -1.0 : 1.0;
    const double scale = psf_pixelscale_lamD / static_cast<double>(npix);

    // Strategy: precompute the two MFT matrices ONCE per geometry
    // (cached across calls -- see mft_cache_get_or_build), then
    // dispatch the two matmuls to BLAS (zgemm). This mirrors what
    // numpy does for ``M_pre @ wavefront @ M_post`` and brings the CPU
    // MFT from O(npsf * npix^2) complex-exp evaluations down to
    // (essentially zero matrix-build work + two highly-optimised zgemm
    // calls) for the steady-state case where geometry is fixed.
    //
    //   M_pre[u, y] = exp(j * sign * 2π * Us[u] * Xs[y])        (npsf, N)
    //   M_post[x, v] = exp(j * sign * 2π * Xs[x] * Us[v])        (N, npsf)
    //
    //   temp = M_pre @ wavefront                                 (npsf, N)
    //   out  = scale * temp @ M_post                             (npsf, npsf)
    const int isign = (convention == '-') ? -1 : 1;
    const char pp0 = pp_centering ? pp_centering[0] : 'o';
    const char fp0 = fp_centering ? fp_centering[0] : 'o';

    MftKey k_pre {npix, npsf, N, du, isign, /*role=*/0, pp0, fp0};
    auto M_pre_ptr = mft_cache_get_or_build(
        k_pre, npsf, N, sign, Us, Xs);
    const auto& M_pre = *M_pre_ptr;

    MftKey k_post{npix, npsf, N, du, isign, /*role=*/1, pp0, fp0};
    auto M_post_ptr = mft_cache_get_or_build(
        k_post, N, npsf, sign, Xs, Us);
    const auto& M_post = *M_post_ptr;

    Array2D<std::complex<double>> temp(npsf, N, {0.0, 0.0});
    Array2D<std::complex<double>> out(npsf, npsf, {0.0, 0.0});

#ifdef LINA_USE_OPENBLAS
    // temp = 1 * M_pre @ wavefront + 0 * temp
    {
        const std::complex<double> alpha(1.0, 0.0);
        const std::complex<double> beta(0.0, 0.0);
        cblas_zgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            static_cast<int>(npsf), static_cast<int>(N), static_cast<int>(N),
            &alpha,
            M_pre.data(),     static_cast<int>(N),
            wavefront.data(), static_cast<int>(N),
            &beta,
            temp.data(),      static_cast<int>(N));
    }
    // out = scale * temp @ M_post + 0 * out
    {
        const std::complex<double> alpha(scale, 0.0);
        const std::complex<double> beta(0.0, 0.0);
        cblas_zgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            static_cast<int>(npsf), static_cast<int>(npsf), static_cast<int>(N),
            &alpha,
            temp.data(),   static_cast<int>(N),
            M_post.data(), static_cast<int>(npsf),
            &beta,
            out.data(),    static_cast<int>(npsf));
    }
#else
    // Fallback hand-rolled matmuls. Still much faster than the prior
    // exp-in-inner-loop implementation because the M matrices are
    // built once. Acceptable because users without OpenBLAS get a
    // working (if not blazing) MFT.
    for (std::size_t u = 0; u < npsf; ++u) {
        for (std::size_t x = 0; x < N; ++x) {
            std::complex<double> sum(0.0, 0.0);
            for (std::size_t y = 0; y < N; ++y) {
                sum += M_pre(u, y) * wavefront(y, x);
            }
            temp(u, x) = sum;
        }
    }
    for (std::size_t u = 0; u < npsf; ++u) {
        for (std::size_t v = 0; v < npsf; ++v) {
            std::complex<double> sum(0.0, 0.0);
            for (std::size_t x = 0; x < N; ++x) {
                sum += temp(u, x) * M_post(x, v);
            }
            out(u, v) = sum * scale;
        }
    }
#endif

    return out;
}

Array2D<std::complex<double>> mft_reverse(
    const Array2D<std::complex<double>>& fpwf,
    double psf_pixelscale_lamD,
    std::size_t npix,
    std::size_t N,
    char convention,
    const char* pp_centering,
    const char* fp_centering) {

    const std::size_t npsf = fpwf.rows();
    if (npsf != fpwf.cols()) {
        throw std::invalid_argument("mft_reverse expects square focal plane");
    }

    const double du = psf_pixelscale_lamD;
    const double dx = 1.0 / static_cast<double>(npix);

    const auto Us = build_coordinates(npsf, du, fp_centering);
    const auto Xs = build_coordinates(N, dx, pp_centering);

    const double sign = (convention == '+') ? 1.0 : -1.0;
    const double scale = psf_pixelscale_lamD / static_cast<double>(npix);

    // Same precompute-then-BLAS strategy as mft_forward, with caching.
    //   Mx[x, u]  = exp(j * sign * 2π * Xs[x] * Us[u])       (N, npsf)
    //   My[v, y]  = exp(j * sign * 2π * Us[v] * Xs[y])       (npsf, N)
    //   temp      = Mx @ fpwf                                 (N, npsf)
    //   out       = scale * temp @ My                         (N, N)
    const int isign = (convention == '+') ? 1 : -1;
    const char pp0 = pp_centering ? pp_centering[0] : 'o';
    const char fp0 = fp_centering ? fp_centering[0] : 'o';

    MftKey k_x {npix, npsf, N, du, isign, /*role=*/2, pp0, fp0};
    auto Mx_ptr = mft_cache_get_or_build(
        k_x, N, npsf, sign, Xs, Us);
    const auto& Mx = *Mx_ptr;

    MftKey k_y {npix, npsf, N, du, isign, /*role=*/3, pp0, fp0};
    auto My_ptr = mft_cache_get_or_build(
        k_y, npsf, N, sign, Us, Xs);
    const auto& My = *My_ptr;

    Array2D<std::complex<double>> temp(N, npsf, {0.0, 0.0});
    Array2D<std::complex<double>> out(N, N, {0.0, 0.0});

#ifdef LINA_USE_OPENBLAS
    {
        const std::complex<double> alpha(1.0, 0.0);
        const std::complex<double> beta(0.0, 0.0);
        cblas_zgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            static_cast<int>(N), static_cast<int>(npsf), static_cast<int>(npsf),
            &alpha,
            Mx.data(),   static_cast<int>(npsf),
            fpwf.data(), static_cast<int>(npsf),
            &beta,
            temp.data(), static_cast<int>(npsf));
    }
    {
        const std::complex<double> alpha(scale, 0.0);
        const std::complex<double> beta(0.0, 0.0);
        cblas_zgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            static_cast<int>(N), static_cast<int>(N), static_cast<int>(npsf),
            &alpha,
            temp.data(), static_cast<int>(npsf),
            My.data(),   static_cast<int>(N),
            &beta,
            out.data(),  static_cast<int>(N));
    }
#else
    for (std::size_t x = 0; x < N; ++x) {
        for (std::size_t v = 0; v < npsf; ++v) {
            std::complex<double> sum(0.0, 0.0);
            for (std::size_t u = 0; u < npsf; ++u) {
                sum += Mx(x, u) * fpwf(u, v);
            }
            temp(x, v) = sum;
        }
    }
    for (std::size_t x = 0; x < N; ++x) {
        for (std::size_t y = 0; y < N; ++y) {
            std::complex<double> sum(0.0, 0.0);
            for (std::size_t v = 0; v < npsf; ++v) {
                sum += temp(x, v) * My(v, y);
            }
            out(x, y) = sum * scale;
        }
    }
#endif

    return out;
}

Array2D<std::complex<double>> get_fresnel_TF(double dz,
                                             std::size_t n,
                                             double wavelength,
                                             double fnum) {
    // df is the spatial-frequency sample spacing of the unshifted pupil
    // FFT result for an array of `n` pixels at this wavelength and f/#.
    const double df = 1.0 / (static_cast<double>(n) * wavelength * fnum);
    // The Python reference uses center=(N-1)/2.0 and shift=False. That
    // maps exactly to lina::Centering::Even with pixelscale=df:
    //   coord_i = (i - n/2 + 0.5) * df = (i - (n-1)/2) * df.
    const Grid2D grid(n, df, Centering::Even);

    Array2D<std::complex<double>> tf(n, n, {0.0, 0.0});
    const std::complex<double> j(0.0, 1.0);
    const double pi = 3.141592653589793;
    for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t c = 0; c < n; ++c) {
            const double rp = grid.r(r, c);
            const double phase = -pi * dz * wavelength * rp * rp;
            tf(r, c) = std::exp(j * phase);
        }
    }
    return tf;
}

} // namespace lina
