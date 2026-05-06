#include "lina/props.h"
#include "lina/grid2d.h"
#include "lina/utils.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef LINA_USE_FFTW
#include <fftw3.h>
#endif

namespace lina {
namespace {

constexpr double kTwoPi = 2.0 * M_PI;

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
    const std::complex<double> j(0.0, 1.0);

    Array2D<std::complex<double>> temp(npsf, N, {0.0, 0.0});
    for (std::size_t u = 0; u < npsf; ++u) {
        for (std::size_t x = 0; x < N; ++x) {
            std::complex<double> sum(0.0, 0.0);
            for (std::size_t y = 0; y < N; ++y) {
                const double phase = sign * kTwoPi * Us[u] * Xs[y];
                sum += wavefront(y, x) * std::exp(j * phase);
            }
            temp(u, x) = sum;
        }
    }

    Array2D<std::complex<double>> out(npsf, npsf, {0.0, 0.0});
    for (std::size_t u = 0; u < npsf; ++u) {
        for (std::size_t v = 0; v < npsf; ++v) {
            std::complex<double> sum(0.0, 0.0);
            for (std::size_t x = 0; x < N; ++x) {
                const double phase = sign * kTwoPi * Xs[x] * Us[v];
                sum += temp(u, x) * std::exp(j * phase);
            }
            out(u, v) = sum * (psf_pixelscale_lamD / static_cast<double>(npix));
        }
    }

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
    const std::complex<double> j(0.0, 1.0);

    // Computes out = Mx @ fpwf @ My with
    //   Mx[x, u] = exp(j * sign * 2π * Xs[x] * Us[u])
    //   My[v, y] = exp(j * sign * 2π * Us[v] * Xs[y])
    // (matches Python lina.props.make_mft_reverse_matrices). The previous
    // factorization produced the transpose of the correct result.
    Array2D<std::complex<double>> temp(N, npsf, {0.0, 0.0});
    for (std::size_t x = 0; x < N; ++x) {
        for (std::size_t v = 0; v < npsf; ++v) {
            std::complex<double> sum(0.0, 0.0);
            for (std::size_t u = 0; u < npsf; ++u) {
                const double phase = sign * kTwoPi * Xs[x] * Us[u];
                sum += std::exp(j * phase) * fpwf(u, v);
            }
            temp(x, v) = sum;
        }
    }

    Array2D<std::complex<double>> out(N, N, {0.0, 0.0});
    for (std::size_t x = 0; x < N; ++x) {
        for (std::size_t y = 0; y < N; ++y) {
            std::complex<double> sum(0.0, 0.0);
            for (std::size_t v = 0; v < npsf; ++v) {
                const double phase = sign * kTwoPi * Us[v] * Xs[y];
                sum += temp(x, v) * std::exp(j * phase);
            }
            out(x, y) = sum * (psf_pixelscale_lamD / static_cast<double>(npix));
        }
    }

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
