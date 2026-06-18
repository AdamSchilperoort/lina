#include "lina/control_models.h"

#include "lina/dm.h"
#include "lina/props.h"
#include "lina/utils.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>

#ifdef LINA_USE_LBFGS
#include <lbfgs.h>
#endif

#ifdef LINA_USE_OPENBLAS
#include <cblas.h>
#endif

namespace lina {
namespace {

Array2D<double> circular_aperture(std::size_t npix, double radius, double pixelscale) {
    Array2D<double> ap(npix, npix, 0.0);
    const double half = static_cast<double>(npix) / 2.0;
    for (std::size_t r = 0; r < npix; ++r) {
        for (std::size_t c = 0; c < npix; ++c) {
            const double x = (static_cast<double>(c) - half + 0.5) * pixelscale;
            const double y = (static_cast<double>(r) - half + 0.5) * pixelscale;
            if (std::hypot(x, y) <= radius) {
                ap(r, c) = 1.0;
            }
        }
    }
    return ap;
}

std::vector<double> linspace(double start, double stop, std::size_t num) {
    std::vector<double> vals(num, 0.0);
    if (num == 1) {
        vals[0] = start;
        return vals;
    }
    const double step = (stop - start) / static_cast<double>(num - 1);
    for (std::size_t i = 0; i < num; ++i) {
        vals[i] = start + step * static_cast<double>(i);
    }
    return vals;
}

std::vector<double> fftfreq(std::size_t n) {
    std::vector<double> freq(n, 0.0);
    const double inv = 1.0 / static_cast<double>(n);
    const std::size_t half = n / 2;
    for (std::size_t i = 0; i < n; ++i) {
        if (i <= half) {
            freq[i] = static_cast<double>(i) * inv;
        } else {
            freq[i] = -static_cast<double>(n - i) * inv;
        }
    }
    return freq;
}

Array2D<std::complex<double>> exp_outer(const std::vector<double>& a,
                                        const std::vector<double>& b,
                                        double sign) {
    Array2D<std::complex<double>> out(a.size(), b.size(), {0.0, 0.0});
    const std::complex<double> j(0.0, 1.0);
    for (std::size_t r = 0; r < a.size(); ++r) {
        for (std::size_t c = 0; c < b.size(); ++c) {
            const double phase = sign * 2.0 * M_PI * a[r] * b[c];
            out(r, c) = std::exp(j * phase);
        }
    }
    return out;
}

std::vector<double> tukey_window(std::size_t n, double alpha) {
    std::vector<double> w(n, 1.0);
    if (alpha <= 0.0) {
        return w;
    }
    if (alpha >= 1.0) {
        for (std::size_t i = 0; i < n; ++i) {
            w[i] = 0.5 * (1.0 - std::cos(2.0 * M_PI * static_cast<double>(i) / (n - 1)));
        }
        return w;
    }
    const std::size_t edge = static_cast<std::size_t>(alpha * (n - 1) / 2.0);
    for (std::size_t i = 0; i < edge; ++i) {
        w[i] = 0.5 * (1.0 + std::cos(M_PI * (2.0 * static_cast<double>(i) / (alpha * (n - 1)) - 1.0)));
        w[n - i - 1] = w[i];
    }
    return w;
}

template <typename T>
Array2D<T> rotate_bilinear(const Array2D<T>& in, double angle_deg) {
    if (angle_deg == 0.0) {
        return in;
    }
    const double radians = angle_deg * M_PI / 180.0;
    const double cos_r = std::cos(radians);
    const double sin_r = std::sin(radians);
    const double cx = (in.cols() - 1) / 2.0;
    const double cy = (in.rows() - 1) / 2.0;
    Array2D<T> out(in.rows(), in.cols(), T{});

    for (std::size_t r = 0; r < in.rows(); ++r) {
        for (std::size_t c = 0; c < in.cols(); ++c) {
            const double x = static_cast<double>(c) - cx;
            const double y = static_cast<double>(r) - cy;
            const double xr = x * cos_r + y * sin_r + cx;
            const double yr = -x * sin_r + y * cos_r + cy;
            if (xr < 0.0 || yr < 0.0 || xr >= in.cols() - 1 || yr >= in.rows() - 1) {
                continue;
            }
            const std::size_t x0 = static_cast<std::size_t>(std::floor(xr));
            const std::size_t y0 = static_cast<std::size_t>(std::floor(yr));
            const double dx = xr - static_cast<double>(x0);
            const double dy = yr - static_cast<double>(y0);

            const T v00 = in(y0, x0);
            const T v10 = in(y0, x0 + 1);
            const T v01 = in(y0 + 1, x0);
            const T v11 = in(y0 + 1, x0 + 1);

            out(r, c) = static_cast<double>(1 - dx) * static_cast<double>(1 - dy) * v00 +
                        dx * (1 - dy) * v10 +
                        (1 - dx) * dy * v01 +
                        dx * dy * v11;
        }
    }
    return out;
}

} // namespace

ControlModel::ControlModel(double wavelength_c,
                           std::optional<double> wavelength,
                           std::size_t npix,
                           std::size_t ndef,
                           std::size_t n_vortex_lres,
                           double vortex_win_diam,
                           double vortex_hres_sampling,
                           double vortex_dot_mask_diam_lamDc,
                           double dm_beam_diam,
                           double lyot_pupil_diam,
                           double lyot_stop_diam,
                           std::optional<double> exit_pupil_prop_dist,
                           double camsci_pxscl_lamDc,
                           std::size_t ncamsci,
                           std::size_t nact,
                           double act_spacing,
                           double act_coupling)
    : wavelength_c_(wavelength_c),
      wavelength_(wavelength.value_or(wavelength_c)),
      dm_beam_diam_(dm_beam_diam),
      lyot_pupil_diam_(lyot_pupil_diam),
      lyot_stop_diam_(lyot_stop_diam),
      lyot_ratio_(lyot_stop_diam / lyot_pupil_diam),
      exit_pupil_prop_dist_(exit_pupil_prop_dist),
      camsci_pxscl_lamDc_(camsci_pxscl_lamDc),
      camsci_pxscl_lamD_(camsci_pxscl_lamDc_ * wavelength_c_ / wavelength_),
      npix_(npix),
      ndef_(ndef),
      def_oversample_(static_cast<double>(ndef) / npix),
      ncamsci_(ncamsci),
      exit_pupil_pxscl_(lyot_pupil_diam_ / static_cast<double>(npix)),
      nact_(nact),
      nacts_(0),
      act_spacing_(act_spacing),
      dm_pxscl_(dm_beam_diam_ / static_cast<double>(npix)),
      inf_sampling_(act_spacing_ / dm_pxscl_),
      n_vortex_lres_(n_vortex_lres),
      vortex_win_diam_(vortex_win_diam),
      hres_sampling_(vortex_hres_sampling),
      vortex_dot_mask_diam_lamDc_(vortex_dot_mask_diam_lamDc),
      vortex_dot_mask_diam_lamD_(vortex_dot_mask_diam_lamDc_ * wavelength_c_ / wavelength_) {
    aperture_ = circular_aperture(ndef_, lyot_pupil_diam_ / 2.0, lyot_pupil_diam_ / npix_);
    lyotstop_ = circular_aperture(ndef_, lyot_stop_diam_ / 2.0, lyot_pupil_diam_ / npix_);
    prefpm_amp_ = Array2D<double>(ndef_, ndef_, 1.0);
    prefpm_opd_ = Array2D<double>(ndef_, ndef_, 0.0);

    dm_mask_ = create_mask(nact_);
    for (std::size_t i = 0; i < dm_mask_.size(); ++i) {
        if (dm_mask_.data()[i]) {
            ++nacts_;
        }
    }

    inf_fun_ = make_gaussian_inf_fun(act_spacing_, inf_sampling_, act_coupling, nact_ + 2);
    nsurf_ = inf_fun_.rows();
    Array2D<std::complex<double>> inf_fun_c(nsurf_, nsurf_, {0.0, 0.0});
    for (std::size_t i = 0; i < inf_fun_.size(); ++i) {
        inf_fun_c.data()[i] = inf_fun_.data()[i];
    }
    inf_fun_fft_ = fft_backend(inf_fun_c);

    const auto xc = linspace(-static_cast<double>(nact_) / 2.0,
                             static_cast<double>(nact_) / 2.0 - 1.0,
                             nact_);
    const auto yc = linspace(-static_cast<double>(nact_) / 2.0,
                             static_cast<double>(nact_) / 2.0 - 1.0,
                             nact_);
    std::vector<double> xc_scaled(nact_, 0.0);
    std::vector<double> yc_scaled(nact_, 0.0);
    for (std::size_t i = 0; i < nact_; ++i) {
        xc_scaled[i] = inf_sampling_ * (xc[i] + 0.5);
        yc_scaled[i] = inf_sampling_ * (yc[i] + 0.5);
    }

    auto fx = fftfreq(nsurf_);
    auto fy = fftfreq(nsurf_);

    mx_dm_ = exp_outer(fx, xc_scaled, -1.0);
    my_dm_ = exp_outer(yc_scaled, fy, -1.0);
    mx_dm_back_ = exp_outer(xc_scaled, fx, 1.0);
    my_dm_back_ = exp_outer(fy, yc_scaled, 1.0);

    oversample_vortex_ = static_cast<double>(n_vortex_lres_) / npix_;
    lres_sampling_ = 1.0 / oversample_vortex_;
    lres_win_size_ = static_cast<std::size_t>(vortex_win_diam_ / lres_sampling_);
    const auto w1d = tukey_window(lres_win_size_, 1.0);
    Array2D<double> outer(lres_win_size_, lres_win_size_, 0.0);
    for (std::size_t r = 0; r < lres_win_size_; ++r) {
        for (std::size_t c = 0; c < lres_win_size_; ++c) {
            outer(r, c) = w1d[r] * w1d[c];
        }
    }
    lres_window_ = pad_or_crop(outer, n_vortex_lres_);
    vortex_lres_ = make_vortex_phase_mask(n_vortex_lres_);

    n_vortex_hres_ = static_cast<std::size_t>(std::round(vortex_win_diam_ / hres_sampling_));
    hres_win_size_ = static_cast<std::size_t>(vortex_win_diam_ / hres_sampling_);
    const auto w1d_h = tukey_window(hres_win_size_, 1.0);
    Array2D<double> outer_h(hres_win_size_, hres_win_size_, 0.0);
    for (std::size_t r = 0; r < hres_win_size_; ++r) {
        for (std::size_t c = 0; c < hres_win_size_; ++c) {
            outer_h(r, c) = w1d_h[r] * w1d_h[c];
        }
    }
    hres_window_ = pad_or_crop(outer_h, n_vortex_hres_);
    vortex_hres_ = make_vortex_phase_mask(n_vortex_hres_);

    hres_dot_mask_ = circular_aperture(
        n_vortex_hres_, (vortex_dot_mask_diam_lamD_ / 2.0) * hres_sampling_, hres_sampling_);
    for (std::size_t i = 0; i < hres_dot_mask_.size(); ++i) {
        hres_dot_mask_.data()[i] = 1.0 - hres_dot_mask_.data()[i];
    }

    windowed_vortex_lres_ = Array2D<std::complex<double>>(n_vortex_lres_, n_vortex_lres_, {0.0, 0.0});
    for (std::size_t i = 0; i < windowed_vortex_lres_.size(); ++i) {
        windowed_vortex_lres_.data()[i] = vortex_lres_.data()[i] * (1.0 - lres_window_.data()[i]);
    }
    windowed_vortex_hres_ = Array2D<std::complex<double>>(n_vortex_hres_, n_vortex_hres_, {0.0, 0.0});
    for (std::size_t i = 0; i < windowed_vortex_hres_.size(); ++i) {
        windowed_vortex_hres_.data()[i] = vortex_hres_.data()[i] * hres_window_.data()[i] * hres_dot_mask_.data()[i];
    }
}

void ControlModel::set_device(const std::string& device) {
    if (device == "cpu") {
        use_gpu_ = false;
    } else if (device == "gpu") {
        // Probe once so users get an immediate clear error if CUDA is unavailable.
        Array2D<std::complex<double>> probe(2, 2, {0.0, 0.0});
        (void)fft_gpu(probe);
        use_gpu_ = true;
    } else {
        throw std::invalid_argument("device must be 'cpu' or 'gpu'");
    }

    // Rebuild backend-dependent cached FFT so subsequent calls stay on the
    // selected compute path -- unless the DM model was supplied externally
    // (in which case inf_fun_fft_ is reference data, not backend-dependent).
    if (!dm_model_external_) {
        Array2D<std::complex<double>> inf_fun_c(nsurf_, nsurf_, {0.0, 0.0});
        for (std::size_t i = 0; i < inf_fun_.size(); ++i) {
            inf_fun_c.data()[i] = inf_fun_.data()[i];
        }
        inf_fun_fft_ = fft_backend(inf_fun_c);
    }
}

Array2D<std::complex<double>> ControlModel::fft_backend(
    const Array2D<std::complex<double>>& arr) const {
    return use_gpu_ ? fft_gpu(arr) : fft(arr);
}

Array2D<std::complex<double>> ControlModel::ifft_backend(
    const Array2D<std::complex<double>>& arr) const {
    return use_gpu_ ? ifft_gpu(arr) : ifft(arr);
}

Array2D<std::complex<double>> ControlModel::ang_spec_backend(
    const Array2D<std::complex<double>>& wavefront,
    double wavelength,
    double distance,
    double pixelscale) const {
    return use_gpu_
               ? ang_spec_gpu(wavefront, wavelength, distance, pixelscale)
               : ang_spec(wavefront, wavelength, distance, pixelscale);
}

Array2D<std::complex<double>> ControlModel::mft_forward_backend(
    const Array2D<std::complex<double>>& wavefront,
    double npix,
    std::size_t npsf,
    double psf_pixelscale_lamD,
    char convention,
    const char* pp_centering,
    const char* fp_centering) const {
    if (!use_gpu_) {
        return mft_forward(
            wavefront, npix, npsf, psf_pixelscale_lamD,
            convention, pp_centering, fp_centering);
    }

    try {
        return mft_forward_gpu(
            wavefront, npix, npsf, psf_pixelscale_lamD,
            convention, pp_centering, fp_centering);
    } catch (const std::exception& err) {
        const std::string msg = err.what();
        if (msg.find("cublasZgemm failed") == std::string::npos) {
            throw;
        }
        // Retry once for transient cuBLAS failures observed on some stacks.
        return mft_forward_gpu(
            wavefront, npix, npsf, psf_pixelscale_lamD,
            convention, pp_centering, fp_centering);
    }
}

Array2D<std::complex<double>> ControlModel::mft_reverse_backend(
    const Array2D<std::complex<double>>& fpwf,
    double psf_pixelscale_lamD,
    double npix,
    std::size_t N,
    char convention,
    const char* pp_centering,
    const char* fp_centering) const {
    if (!use_gpu_) {
        return mft_reverse(
            fpwf, psf_pixelscale_lamD, npix, N,
            convention, pp_centering, fp_centering);
    }

    try {
        return mft_reverse_gpu(
            fpwf, psf_pixelscale_lamD, npix, N,
            convention, pp_centering, fp_centering);
    } catch (const std::exception& err) {
        const std::string msg = err.what();
        if (msg.find("cublasZgemm failed") == std::string::npos) {
            throw;
        }
        // Retry once for transient cuBLAS failures observed on some stacks.
        return mft_reverse_gpu(
            fpwf, psf_pixelscale_lamD, npix, N,
            convention, pp_centering, fp_centering);
    }
}

Array2D<std::complex<double>> ControlModel::matmul(
    const Array2D<std::complex<double>>& a,
    const Array2D<std::complex<double>>& b) const {
    if (a.cols() != b.rows()) {
        throw std::invalid_argument("complex matmul dimension mismatch");
    }
    const std::size_t m = a.rows();
    const std::size_t k = a.cols();
    const std::size_t n = b.cols();
    Array2D<std::complex<double>> out(m, n, {0.0, 0.0});

#ifdef LINA_USE_OPENBLAS
    // BLAS complex GEMM (row-major). std::complex<double> is layout-compatible
    // with the two-double representation CBLAS expects. This is ~50x faster
    // than the naive triple loop for the nsurf-sized DM-model matrices.
    const std::complex<double> alpha(1.0, 0.0);
    const std::complex<double> beta(0.0, 0.0);
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                &alpha, a.data(), static_cast<int>(k),
                b.data(), static_cast<int>(n),
                &beta, out.data(), static_cast<int>(n));
#else
    for (std::size_t r = 0; r < m; ++r) {
        for (std::size_t c = 0; c < n; ++c) {
            std::complex<double> sum(0.0, 0.0);
            for (std::size_t kk = 0; kk < k; ++kk) {
                sum += a(r, kk) * b(kk, c);
            }
            out(r, c) = sum;
        }
    }
#endif
    return out;
}

ControlModel::ForwardResult ControlModel::forward_internal(const std::vector<double>& actuators,
                                                           double wavelength,
                                                           bool use_vortex) const {
    Array2D<double> dm_command(nact_, nact_, 0.0);
    std::size_t idx = 0;
    for (std::size_t i = 0; i < dm_mask_.size(); ++i) {
        if (dm_mask_.data()[i]) {
            dm_command.data()[i] = actuators[idx++];
        }
    }

    Array2D<std::complex<double>> dm_command_c(nact_, nact_, {0.0, 0.0});
    for (std::size_t i = 0; i < dm_command.size(); ++i) {
        dm_command_c.data()[i] = dm_command.data()[i];
    }

    const auto mft_command = matmul(matmul(mx_dm_, dm_command_c), my_dm_);
    Array2D<std::complex<double>> fourier_surf(nsurf_, nsurf_, {0.0, 0.0});
    for (std::size_t i = 0; i < fourier_surf.size(); ++i) {
        fourier_surf.data()[i] = inf_fun_fft_.data()[i] * mft_command.data()[i];
    }
    Array2D<std::complex<double>> dm_surf_c = ifft_backend(fourier_surf);
    Array2D<double> dm_surf(nsurf_, nsurf_, 0.0);
    for (std::size_t i = 0; i < dm_surf.size(); ++i) {
        dm_surf.data()[i] = dm_surf_c.data()[i].real();
    }

    Array2D<std::complex<double>> dm_phasor(nsurf_, nsurf_, {0.0, 0.0});
    for (std::size_t i = 0; i < dm_surf.size(); ++i) {
        const double phase = 4.0 * M_PI / wavelength * dm_surf.data()[i];
        dm_phasor.data()[i] = std::exp(std::complex<double>(0.0, phase));
    }

    Array2D<std::complex<double>> wfe(ndef_, ndef_, {0.0, 0.0});
    for (std::size_t i = 0; i < wfe.size(); ++i) {
        const double phase = 2.0 * M_PI / wavelength * prefpm_opd_.data()[i];
        wfe.data()[i] = prefpm_amp_.data()[i] * std::exp(std::complex<double>(0.0, phase));
    }

    Array2D<std::complex<double>> e_ep(ndef_, ndef_, {0.0, 0.0});
    for (std::size_t i = 0; i < e_ep.size(); ++i) {
        e_ep.data()[i] = aperture_.data()[i] * wfe.data()[i];
    }

    Array2D<std::complex<double>> dm_phasor_pad = pad_or_crop(dm_phasor, ndef_);
    Array2D<std::complex<double>> e_dm(ndef_, ndef_, {0.0, 0.0});
    for (std::size_t i = 0; i < e_dm.size(); ++i) {
        e_dm.data()[i] = e_ep.data()[i] * dm_phasor_pad.data()[i];
    }

    Array2D<std::complex<double>> e_lp(ndef_, ndef_, {0.0, 0.0});
    if (use_vortex) {
        Array2D<std::complex<double>> e_dm_lres = pad_or_crop(e_dm, n_vortex_lres_);
        Array2D<std::complex<double>> e_fpm_lres = fft_backend(e_dm_lres);
        for (std::size_t i = 0; i < e_fpm_lres.size(); ++i) {
            e_fpm_lres.data()[i] *= windowed_vortex_lres_.data()[i];
        }
        Array2D<std::complex<double>> e_lp_lres = ifft_backend(e_fpm_lres);
        e_lp_lres = pad_or_crop(e_lp_lres, ndef_);

        Array2D<std::complex<double>> e_fpm_hres = mft_forward_backend(
            e_dm, npix_, n_vortex_hres_, hres_sampling_, '-', "odd", "odd");
        for (std::size_t i = 0; i < e_fpm_hres.size(); ++i) {
            e_fpm_hres.data()[i] *= windowed_vortex_hres_.data()[i];
        }
        Array2D<std::complex<double>> e_lp_hres = mft_reverse_backend(
            e_fpm_hres, hres_sampling_, npix_, ndef_, '+', "odd", "odd");

        for (std::size_t i = 0; i < e_lp.size(); ++i) {
            e_lp.data()[i] = e_lp_lres.data()[i] + e_lp_hres.data()[i];
        }
    } else {
        e_lp = e_dm;
    }

    Array2D<std::complex<double>> e_ls(ndef_, ndef_, {0.0, 0.0});
    for (std::size_t i = 0; i < e_ls.size(); ++i) {
        e_ls.data()[i] = lyotstop_.data()[i] * e_lp.data()[i];
    }

    Array2D<std::complex<double>> e_fffp = e_ls;
    if (exit_pupil_prop_dist_.has_value()) {
        Array2D<std::complex<double>> e_ls_pad = pad_or_crop(e_ls, 2 * ndef_);
        e_fffp = ang_spec_backend(
            e_ls_pad, wavelength, exit_pupil_prop_dist_.value(), exit_pupil_pxscl_);
    }

    const double camsci_pxscl_lamD = camsci_pxscl_lamDc_ * wavelength_c_ / wavelength;
    // npix here is a sampling scale (dx = 1/npix), not an array size, so keep
    // it fractional to match the Python reference (npix * lyot_ratio).
    Array2D<std::complex<double>> e_fp = mft_forward_backend(
        e_fffp, static_cast<double>(npix_) * lyot_ratio_, ncamsci_,
        camsci_pxscl_lamD, '-', "odd", "odd");

    if (camsci_rotation_ != 0.0) {
        e_fp = rotate_bilinear(e_fp, camsci_rotation_);
    }
    return {e_fp, e_ep, dm_phasor};
}

Array2D<std::complex<double>> ControlModel::forward(const std::vector<double>& actuators,
                                                    double wavelength,
                                                    bool use_vortex) {
#ifdef LINA_USE_CUDA
    if (use_gpu_) {
        // Device-resident pipeline: all intermediates stay on the GPU,
        // only the focal-plane field is copied back to the host.
        return control_model_forward_gpu(*this, actuators, wavelength, use_vortex);
    }
#endif
    return forward_internal(actuators, wavelength, use_vortex).e_fp;
}

void ControlModel::set_prefpm_amp(const Array2D<double>& amp) {
    if (amp.rows() != ndef_ || amp.cols() != ndef_) {
        throw std::invalid_argument("set_prefpm_amp shape mismatch");
    }
    prefpm_amp_ = amp;
}

void ControlModel::set_prefpm_opd(const Array2D<double>& opd) {
    if (opd.rows() != ndef_ || opd.cols() != ndef_) {
        throw std::invalid_argument("set_prefpm_opd shape mismatch");
    }
    prefpm_opd_ = opd;
}

void ControlModel::set_aperture(const Array2D<double>& aperture) {
    if (aperture.rows() != ndef_ || aperture.cols() != ndef_) {
        throw std::invalid_argument("set_aperture shape mismatch");
    }
    aperture_ = aperture;
    // Invalidate any cached GPU constants so the next forward re-uploads.
    gpu_state_.reset();
}

void ControlModel::set_lyotstop(const Array2D<double>& lyotstop) {
    if (lyotstop.rows() != ndef_ || lyotstop.cols() != ndef_) {
        throw std::invalid_argument("set_lyotstop shape mismatch");
    }
    lyotstop_ = lyotstop;
    gpu_state_.reset();
}

void ControlModel::set_windowed_vortex_lres(const Array2D<std::complex<double>>& m) {
    if (m.rows() != n_vortex_lres_ || m.cols() != n_vortex_lres_) {
        throw std::invalid_argument("set_windowed_vortex_lres shape mismatch");
    }
    windowed_vortex_lres_ = m;
    gpu_state_.reset();
}

void ControlModel::set_windowed_vortex_hres(const Array2D<std::complex<double>>& m) {
    if (m.rows() != n_vortex_hres_ || m.cols() != n_vortex_hres_) {
        throw std::invalid_argument("set_windowed_vortex_hres shape mismatch");
    }
    windowed_vortex_hres_ = m;
    gpu_state_.reset();
}

void ControlModel::set_dm_model(const Array2D<std::complex<double>>& inf_fun_fft,
                                const Array2D<std::complex<double>>& mx_dm,
                                const Array2D<std::complex<double>>& my_dm,
                                const Array2D<std::complex<double>>& mx_dm_back,
                                const Array2D<std::complex<double>>& my_dm_back) {
    if (inf_fun_fft.rows() != nsurf_ || inf_fun_fft.cols() != nsurf_) {
        throw std::invalid_argument("set_dm_model: inf_fun_fft shape mismatch");
    }
    if (mx_dm.rows() != nsurf_ || mx_dm.cols() != nact_ ||
        my_dm.rows() != nact_ || my_dm.cols() != nsurf_ ||
        mx_dm_back.rows() != nact_ || mx_dm_back.cols() != nsurf_ ||
        my_dm_back.rows() != nsurf_ || my_dm_back.cols() != nact_) {
        throw std::invalid_argument("set_dm_model: DM matrix shape mismatch");
    }
    inf_fun_fft_ = inf_fun_fft;
    mx_dm_ = mx_dm;
    my_dm_ = my_dm;
    mx_dm_back_ = mx_dm_back;
    my_dm_back_ = my_dm_back;
    dm_model_external_ = true;
    gpu_state_.reset();
}

double dm_val_and_grad(const ControlModel& model,
                       const std::vector<double>& del_acts,
                       const Array2D<double>& opd,
                       std::vector<double>& grad_out) {
    const std::size_t nact = model.nact_;
    const std::size_t nsurf = model.nsurf_;
    const std::size_t ndef = opd.rows();
    if (opd.cols() != ndef) {
        throw std::invalid_argument("dm_val_and_grad expects square OPD");
    }

    // del_command (nact x nact) from the masked actuator vector.
    Array2D<std::complex<double>> dm_command_c(nact, nact, {0.0, 0.0});
    std::size_t idx = 0;
    for (std::size_t i = 0; i < model.dm_mask_.size(); ++i) {
        if (model.dm_mask_.data()[i]) {
            dm_command_c.data()[i] = del_acts[idx++];
        }
    }

    // dm_surf = Re( ifft( inf_fun_fft * (Mx_dm @ dm_command @ My_dm) ) )
    // Use the CPU FFT here: this is a small (nsurf^2) one-time flat-DM solve,
    // and the CPU path avoids the per-call GPU host<->device round-trips that
    // dominate runtime when this is driven by an optimizer (hundreds of evals).
    // Numerically identical to the GPU FFT to ~1e-12.
    const auto mft_command = model.matmul(model.matmul(model.mx_dm_, dm_command_c), model.my_dm_);
    Array2D<std::complex<double>> fourier_surf(nsurf, nsurf, {0.0, 0.0});
    for (std::size_t i = 0; i < fourier_surf.size(); ++i) {
        fourier_surf.data()[i] = model.inf_fun_fft_.data()[i] * mft_command.data()[i];
    }
    Array2D<std::complex<double>> dm_surf_c = ifft(fourier_surf);
    Array2D<double> dm_surf(nsurf, nsurf, 0.0);
    for (std::size_t i = 0; i < dm_surf.size(); ++i) {
        dm_surf.data()[i] = dm_surf_c.data()[i].real();
    }
    Array2D<double> dm_surf_pad = pad_or_crop(dm_surf, ndef);

    // Beam-aperture mask = aperture > 0 (already ndef x ndef).
    const Array2D<double>& ap = model.aperture_;

    double opd_l2norm = 0.0;
    for (std::size_t i = 0; i < opd.size(); ++i) {
        if (ap.data()[i] > 0.0) {
            const double v = opd.data()[i];
            opd_l2norm += v * v;
        }
    }
    if (opd_l2norm == 0.0) {
        opd_l2norm = 1.0;
    }

    double j_num = 0.0;
    Array2D<double> total_opd(ndef, ndef, 0.0);
    for (std::size_t i = 0; i < opd.size(); ++i) {
        const double t = opd.data()[i] + 2.0 * dm_surf_pad.data()[i];
        total_opd.data()[i] = t;
        if (ap.data()[i] > 0.0) {
            j_num += t * t;
        }
    }
    const double J = j_num / opd_l2norm;

    // dJ/dOPD = 2 * mask * total_opd / opd_l2norm
    Array2D<double> dJ_dOPD(ndef, ndef, 0.0);
    for (std::size_t i = 0; i < opd.size(); ++i) {
        if (ap.data()[i] > 0.0) {
            dJ_dOPD.data()[i] = 2.0 * total_opd.data()[i] / opd_l2norm;
        }
    }

    // Back-propagate to actuators (adjoint of the forward DM-surface model).
    Array2D<double> dJ_dS_DM = pad_or_crop(dJ_dOPD, nsurf);
    Array2D<std::complex<double>> dJ_dS_DM_c(nsurf, nsurf, {0.0, 0.0});
    for (std::size_t i = 0; i < dJ_dS_DM.size(); ++i) {
        dJ_dS_DM_c.data()[i] = dJ_dS_DM.data()[i];
    }
    Array2D<std::complex<double>> x2_bar = fft(dJ_dS_DM_c);
    Array2D<std::complex<double>> x1_bar(nsurf, nsurf, {0.0, 0.0});
    for (std::size_t i = 0; i < x1_bar.size(); ++i) {
        x1_bar.data()[i] = std::conj(model.inf_fun_fft_.data()[i]) * x2_bar.data()[i];
    }
    Array2D<std::complex<double>> dJ_dA1 =
        model.matmul(model.matmul(model.mx_dm_back_, x1_bar), model.my_dm_back_);
    const double norm = static_cast<double>(nsurf * nact * nact);

    grad_out.assign(model.nacts_, 0.0);
    std::size_t act_idx = 0;
    for (std::size_t i = 0; i < model.dm_mask_.size(); ++i) {
        if (model.dm_mask_.data()[i]) {
            grad_out[act_idx++] = dJ_dA1.data()[i].real() / norm;
        }
    }
    return J;
}

namespace {

using ObjFn = std::function<double(const std::vector<double>&, std::vector<double>&)>;

double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

double inf_norm(const std::vector<double>& v) {
    double m = 0.0;
    for (double x : v) m = std::max(m, std::abs(x));
    return m;
}

// ---------------------------------------------------------------------------
// Linear conjugate gradient for the (convex, quadratic) flat-DM objective.
//
// J(x) is quadratic in the actuators, so its gradient is affine:
//     grad(x) = A x - b   (A symmetric PSD).
// Linear CG is the natural, parameter-free solver here: starting from x = 0 it
// converges to the minimum-norm solution within the Krylov subspace and never
// excites the degenerate (zero-curvature) directions that make a generic
// L-BFGS line search drift into large, meaningless actuator strokes. This
// reproduces SciPy L-BFGS-B's well-behaved (small-stroke) solution.
//
// We need only the gradient oracle: A d = grad(x + d) - grad(x), evaluated via
// one extra objective call per iteration. Any uniform scaling of grad (the
// analytic objective returns a consistently scaled gradient) leaves the
// solution unchanged, since CG then solves (sA) x = (s b) <=> A x = b.
// ---------------------------------------------------------------------------
std::vector<double> cg_minimize(std::size_t n,
                                const ObjFn& objective,
                                double gtol,
                                int max_iter) {
    if (max_iter <= 0) max_iter = static_cast<int>(n) + 50;

    std::vector<double> x(n, 0.0);
    std::vector<double> g(n, 0.0);
    (void)objective(x, g);          // g = grad(0) = -b (up to scale)

    // Relative gradient stop: the flat-DM objective's analytic gradient is a
    // (consistent) constant scaling of the true gradient, so an absolute
    // threshold is meaningless. Stop when the gradient inf-norm has dropped by
    // `gtol` relative to its initial value (default 1e-4 -> 4 orders), which
    // captures the well-observed DM modes and ignores the degenerate tail.
    const double g0 = std::max(inf_norm(g), 1e-300);
    const double grad_stop = gtol * g0;

    std::vector<double> r(n, 0.0);  // residual = -grad(x)
    for (std::size_t i = 0; i < n; ++i) r[i] = -g[i];
    std::vector<double> d = r;
    double rs_old = dot(r, r);

    std::vector<double> x_trial(n, 0.0), g_trial(n, 0.0), Ad(n, 0.0);

    for (int iter = 0; iter < max_iter; ++iter) {
        if (inf_norm(g) <= grad_stop) break;

        // A d = grad(x + d) - grad(x).
        for (std::size_t i = 0; i < n; ++i) x_trial[i] = x[i] + d[i];
        (void)objective(x_trial, g_trial);
        for (std::size_t i = 0; i < n; ++i) Ad[i] = g_trial[i] - g[i];

        const double dAd = dot(d, Ad);
        if (!(dAd > 0.0)) break;  // no positive curvature left (degenerate)

        const double alpha = rs_old / dAd;
        for (std::size_t i = 0; i < n; ++i) {
            x[i] += alpha * d[i];
            r[i] -= alpha * Ad[i];
            g[i] = -r[i];           // grad(x) updated consistently
        }

        if (inf_norm(g) <= grad_stop) break;
        const double rs_new = dot(r, r);
        const double beta = rs_new / rs_old;
        for (std::size_t i = 0; i < n; ++i) d[i] = r[i] + beta * d[i];
        rs_old = rs_new;
    }
    return x;
}

} // namespace

std::vector<double> solve_flat_command(const ControlModel& model,
                                       const Array2D<double>& opd,
                                       double tol,
                                       int max_iter) {
    // The flat-DM objective is a convex quadratic in the actuators, so linear
    // conjugate gradient is the natural solver: it is parameter-free, invariant
    // to the constant scaling of the analytic gradient, converges to the
    // minimum-norm (small-stroke) solution that matches SciPy L-BFGS-B, and
    // avoids the line-search fragility of a generic quasi-Newton method on a
    // dimensionally-tiny, ill-conditioned problem.
    const std::size_t n = model.nacts();
    auto objective = [&](const std::vector<double>& x, std::vector<double>& grad) -> double {
        return dm_val_and_grad(model, x, opd, grad);
    };
    return cg_minimize(n, objective, tol, max_iter);
}

double val_and_grad(const std::vector<double>& del_acts,
                    const ControlModel& model,
                    const std::vector<double>& current_acts,
                    const Array2D<std::complex<double>>& e_ab,
                    const Array2D<std::complex<double>>& e_fp_nom,
                    const Array2D<std::uint8_t>& control_mask,
                    double wavelength,
                    double r_cond,
                    std::vector<double>& grad_out) {
    const std::size_t nacts = del_acts.size();
    std::vector<double> del_acts_waves(nacts, 0.0);
    for (std::size_t i = 0; i < nacts; ++i) {
        del_acts_waves[i] = del_acts[i] / model.wavelength();
    }

    double e_ab_l2norm = 0.0;
    for (std::size_t i = 0; i < control_mask.size(); ++i) {
        if (control_mask.data()[i]) {
            e_ab_l2norm += std::norm(e_ab.data()[i]);
        }
    }

    std::vector<double> acts_sum(nacts, 0.0);
    for (std::size_t i = 0; i < nacts; ++i) {
        acts_sum[i] = current_acts[i] + del_acts[i];
    }
    const auto forward_res = model.forward_internal(acts_sum, wavelength, true);
    const auto& e_fp_with_del = forward_res.e_fp;
    const auto& e_ep = forward_res.e_ep;
    const auto& dm_phasor = forward_res.dm_phasor;

    Array2D<std::complex<double>> delta_e(e_fp_with_del.rows(), e_fp_with_del.cols(), {0.0, 0.0});
    for (std::size_t i = 0; i < delta_e.size(); ++i) {
        delta_e.data()[i] = e_fp_with_del.data()[i] - e_fp_nom.data()[i];
    }

    double j_del_e = 0.0;
    Array2D<std::complex<double>> e_predicted(e_fp_with_del.rows(), e_fp_with_del.cols(), {0.0, 0.0});
    for (std::size_t i = 0; i < e_predicted.size(); ++i) {
        e_predicted.data()[i] = e_ab.data()[i] + delta_e.data()[i];
        if (control_mask.data()[i]) {
            j_del_e += std::norm(e_predicted.data()[i]);
        }
    }

    double j_c = 0.0;
    for (double v : del_acts_waves) {
        j_c += v * v;
    }
    j_c *= r_cond;
    const double j_total = (j_del_e + j_c) / e_ab_l2norm;

    Array2D<std::complex<double>> e_predicted_masked(e_predicted.rows(), e_predicted.cols(), {0.0, 0.0});
    for (std::size_t i = 0; i < e_predicted_masked.size(); ++i) {
        if (control_mask.data()[i]) {
            e_predicted_masked.data()[i] = e_predicted.data()[i];
        }
    }

    Array2D<std::complex<double>> dJ_ddeltaE = rotate_bilinear(e_predicted_masked, -model.camsci_rotation_);
    for (std::size_t i = 0; i < dJ_ddeltaE.size(); ++i) {
        dJ_ddeltaE.data()[i] *= (2.0 / e_ab_l2norm);
    }

    const double camsci_pxscl_lamD = model.camsci_pxscl_lamDc_ * model.wavelength_c_ / wavelength;
    Array2D<std::complex<double>> dJ_dE_FFFP = model.mft_reverse_backend(
        dJ_ddeltaE, camsci_pxscl_lamD,
        static_cast<double>(model.npix_) * model.lyot_ratio_,
        2 * model.ndef_, '+', "odd", "odd");

    Array2D<std::complex<double>> dJ_dE_LS = dJ_dE_FFFP;
    if (model.exit_pupil_prop_dist_.has_value()) {
        dJ_dE_LS = model.ang_spec_backend(
            dJ_dE_FFFP, wavelength, -model.exit_pupil_prop_dist_.value(),
            model.exit_pupil_pxscl_);
        dJ_dE_LS = pad_or_crop(dJ_dE_LS, model.ndef_);
    } else {
        dJ_dE_LS = pad_or_crop(dJ_dE_LS, model.ndef_);
    }

    Array2D<std::complex<double>> dJ_dE_LP(dJ_dE_LS.rows(), dJ_dE_LS.cols(), {0.0, 0.0});
    for (std::size_t i = 0; i < dJ_dE_LP.size(); ++i) {
        dJ_dE_LP.data()[i] = dJ_dE_LS.data()[i] * model.lyotstop_.data()[i];
    }

    Array2D<std::complex<double>> dJ_dE_LP_fft = pad_or_crop(dJ_dE_LP, model.n_vortex_lres_);
    Array2D<std::complex<double>> dJ_dE_FPM_fft = model.fft_backend(dJ_dE_LP_fft);
    Array2D<std::complex<double>> dJ_dE_FP_fft(dJ_dE_FPM_fft.rows(), dJ_dE_FPM_fft.cols(), {0.0, 0.0});
    for (std::size_t i = 0; i < dJ_dE_FP_fft.size(); ++i) {
        dJ_dE_FP_fft.data()[i] =
            std::conj(model.vortex_lres_.data()[i]) * (1.0 - model.lres_window_.data()[i]) *
            dJ_dE_FPM_fft.data()[i];
    }
    Array2D<std::complex<double>> dJ_dE_PUP_fft = model.ifft_backend(dJ_dE_FP_fft);
    dJ_dE_PUP_fft = pad_or_crop(dJ_dE_PUP_fft, model.ndef_);

    Array2D<std::complex<double>> dJ_dE_FPM_mft = model.mft_forward_backend(
        dJ_dE_LP, model.npix_, model.n_vortex_hres_, model.hres_sampling_, '-', "odd", "odd");
    Array2D<std::complex<double>> dJ_dE_FP_mft(dJ_dE_FPM_mft.rows(), dJ_dE_FPM_mft.cols(), {0.0, 0.0});
    for (std::size_t i = 0; i < dJ_dE_FP_mft.size(); ++i) {
        dJ_dE_FP_mft.data()[i] =
            std::conj(model.vortex_hres_.data()[i]) * model.hres_window_.data()[i] *
            model.hres_dot_mask_.data()[i] * dJ_dE_FPM_mft.data()[i];
    }
    Array2D<std::complex<double>> dJ_dE_PUP_mft = model.mft_reverse_backend(
        dJ_dE_FP_mft, model.hres_sampling_, model.npix_, model.ndef_, '+', "odd", "odd");

    Array2D<std::complex<double>> dJ_dE_PUP(dJ_dE_PUP_fft.rows(), dJ_dE_PUP_fft.cols(), {0.0, 0.0});
    for (std::size_t i = 0; i < dJ_dE_PUP.size(); ++i) {
        dJ_dE_PUP.data()[i] = dJ_dE_PUP_fft.data()[i] + dJ_dE_PUP_mft.data()[i];
    }

    Array2D<std::complex<double>> dm_phasor_pad = pad_or_crop(dm_phasor, model.ndef_);
    Array2D<double> dJ_dS_DM(model.ndef_, model.ndef_, 0.0);
    for (std::size_t i = 0; i < dJ_dS_DM.size(); ++i) {
        const std::complex<double> val = dJ_dE_PUP.data()[i] *
                                         std::conj(e_ep.data()[i]) *
                                         std::conj(dm_phasor_pad.data()[i]);
        dJ_dS_DM.data()[i] = 4.0 * M_PI / wavelength * std::imag(val);
    }

    Array2D<double> dJ_dS_DM_pad = pad_or_crop(dJ_dS_DM, model.nsurf_);
    Array2D<std::complex<double>> dJ_dS_DM_c(model.nsurf_, model.nsurf_, {0.0, 0.0});
    for (std::size_t i = 0; i < dJ_dS_DM_pad.size(); ++i) {
        dJ_dS_DM_c.data()[i] = dJ_dS_DM_pad.data()[i];
    }
    Array2D<std::complex<double>> x2_bar = model.fft_backend(dJ_dS_DM_c);
    Array2D<std::complex<double>> x1_bar(model.nsurf_, model.nsurf_, {0.0, 0.0});
    for (std::size_t i = 0; i < x1_bar.size(); ++i) {
        x1_bar.data()[i] = std::conj(model.inf_fun_fft_.data()[i]) * x2_bar.data()[i];
    }

    Array2D<std::complex<double>> dJ_dA =
        model.matmul(model.matmul(model.mx_dm_back_, x1_bar), model.my_dm_back_);
    const double norm = static_cast<double>(model.nsurf_ * model.nact_ * model.nact_);
    for (std::size_t i = 0; i < dJ_dA.size(); ++i) {
        dJ_dA.data()[i] /= norm;
    }

    grad_out.assign(nacts, 0.0);
    std::size_t act_idx = 0;
    for (std::size_t i = 0; i < model.dm_mask_.size(); ++i) {
        if (model.dm_mask_.data()[i]) {
            grad_out[act_idx] = dJ_dA.data()[i].real() + r_cond * 2.0 * del_acts_waves[act_idx];
            ++act_idx;
        }
    }
    return j_total;
}

double val_and_grad_bb(const std::vector<double>& del_acts,
                       const ControlModel& model,
                       const std::vector<double>& actuators,
                       const std::vector<Array2D<std::complex<double>>>& e_abs,
                       const Array2D<std::uint8_t>& control_mask,
                       const std::vector<double>& waves,
                       double r_cond,
                       std::vector<double>& grad_out) {
    const std::size_t nwaves = waves.size();
    grad_out.assign(del_acts.size(), 0.0);
    double j_sum = 0.0;
    for (std::size_t i = 0; i < nwaves; ++i) {
        std::vector<double> grad_mono;
        const auto e_fp_nom = model.forward_internal(actuators, waves[i], true).e_fp;
        const double j_mono = val_and_grad(del_acts, model, actuators, e_abs[i], e_fp_nom, control_mask,
                                           waves[i], 0.0, grad_mono);
        j_sum += j_mono;
        for (std::size_t k = 0; k < grad_out.size(); ++k) {
            grad_out[k] += grad_mono[k];
        }
    }
    const double inv_lam = 1.0 / model.wavelength();
    for (std::size_t i = 0; i < del_acts.size(); ++i) {
        grad_out[i] += r_cond * 2.0 * del_acts[i] * inv_lam;
    }
    return j_sum / static_cast<double>(nwaves);
}

} // namespace lina
