#include "lina/control_models.h"

#include "lina/dm.h"
#include "lina/props.h"
#include "lina/utils.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

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
    inf_fun_fft_ = fft(inf_fun_c);

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

Array2D<std::complex<double>> ControlModel::matmul(
    const Array2D<std::complex<double>>& a,
    const Array2D<std::complex<double>>& b) const {
    if (a.cols() != b.rows()) {
        throw std::invalid_argument("complex matmul dimension mismatch");
    }
    Array2D<std::complex<double>> out(a.rows(), b.cols(), {0.0, 0.0});
    for (std::size_t r = 0; r < a.rows(); ++r) {
        for (std::size_t c = 0; c < b.cols(); ++c) {
            std::complex<double> sum(0.0, 0.0);
            for (std::size_t k = 0; k < a.cols(); ++k) {
                sum += a(r, k) * b(k, c);
            }
            out(r, c) = sum;
        }
    }
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
    Array2D<std::complex<double>> dm_surf_c = ifft(fourier_surf);
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
        Array2D<std::complex<double>> e_fpm_lres = fft(e_dm_lres);
        for (std::size_t i = 0; i < e_fpm_lres.size(); ++i) {
            e_fpm_lres.data()[i] *= windowed_vortex_lres_.data()[i];
        }
        Array2D<std::complex<double>> e_lp_lres = ifft(e_fpm_lres);
        e_lp_lres = pad_or_crop(e_lp_lres, ndef_);

        Array2D<std::complex<double>> e_fpm_hres = mft_forward(
            e_dm, npix_, n_vortex_hres_, hres_sampling_, '-', "odd", "odd");
        for (std::size_t i = 0; i < e_fpm_hres.size(); ++i) {
            e_fpm_hres.data()[i] *= windowed_vortex_hres_.data()[i];
        }
        Array2D<std::complex<double>> e_lp_hres = mft_reverse(
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
        e_fffp = ang_spec(e_ls_pad, wavelength, exit_pupil_prop_dist_.value(), exit_pupil_pxscl_);
    }

    const double camsci_pxscl_lamD = camsci_pxscl_lamDc_ * wavelength_c_ / wavelength;
    Array2D<std::complex<double>> e_fp = mft_forward(
        e_fffp, static_cast<std::size_t>(npix_ * lyot_ratio_), ncamsci_, camsci_pxscl_lamD, '-', "odd", "odd");

    if (camsci_rotation_ != 0.0) {
        e_fp = rotate_bilinear(e_fp, camsci_rotation_);
    }
    return {e_fp, e_ep, dm_phasor};
}

Array2D<std::complex<double>> ControlModel::forward(const std::vector<double>& actuators,
                                                    double wavelength,
                                                    bool use_vortex) {
    return forward_internal(actuators, wavelength, use_vortex).e_fp;
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
    Array2D<std::complex<double>> dJ_dE_FFFP = mft_reverse(
        dJ_ddeltaE, camsci_pxscl_lamD, static_cast<std::size_t>(model.npix_ * model.lyot_ratio_),
        2 * model.ndef_, '+', "odd", "odd");

    Array2D<std::complex<double>> dJ_dE_LS = dJ_dE_FFFP;
    if (model.exit_pupil_prop_dist_.has_value()) {
        dJ_dE_LS = ang_spec(dJ_dE_FFFP, wavelength, -model.exit_pupil_prop_dist_.value(),
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
    Array2D<std::complex<double>> dJ_dE_FPM_fft = fft(dJ_dE_LP_fft);
    Array2D<std::complex<double>> dJ_dE_FP_fft(dJ_dE_FPM_fft.rows(), dJ_dE_FPM_fft.cols(), {0.0, 0.0});
    for (std::size_t i = 0; i < dJ_dE_FP_fft.size(); ++i) {
        dJ_dE_FP_fft.data()[i] =
            std::conj(model.vortex_lres_.data()[i]) * (1.0 - model.lres_window_.data()[i]) *
            dJ_dE_FPM_fft.data()[i];
    }
    Array2D<std::complex<double>> dJ_dE_PUP_fft = ifft(dJ_dE_FP_fft);
    dJ_dE_PUP_fft = pad_or_crop(dJ_dE_PUP_fft, model.ndef_);

    Array2D<std::complex<double>> dJ_dE_FPM_mft = mft_forward(
        dJ_dE_LP, model.npix_, model.n_vortex_hres_, model.hres_sampling_, '-', "odd", "odd");
    Array2D<std::complex<double>> dJ_dE_FP_mft(dJ_dE_FPM_mft.rows(), dJ_dE_FPM_mft.cols(), {0.0, 0.0});
    for (std::size_t i = 0; i < dJ_dE_FP_mft.size(); ++i) {
        dJ_dE_FP_mft.data()[i] =
            std::conj(model.vortex_hres_.data()[i]) * model.hres_window_.data()[i] *
            model.hres_dot_mask_.data()[i] * dJ_dE_FPM_mft.data()[i];
    }
    Array2D<std::complex<double>> dJ_dE_PUP_mft = mft_reverse(
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
    Array2D<std::complex<double>> x2_bar = fft(dJ_dS_DM_c);
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
