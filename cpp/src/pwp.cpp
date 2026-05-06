#include "lina/pwp.h"

#include "lina/linalg.h"

#include <cmath>
#include <stdexcept>

namespace lina {
namespace {

std::vector<std::size_t> mask_indices(const Array2D<std::uint8_t>& mask) {
    std::vector<std::size_t> indices;
    indices.reserve(mask.size());
    for (std::size_t i = 0; i < mask.size(); ++i) {
        if (mask.data()[i]) {
            indices.push_back(i);
        }
    }
    return indices;
}

template <typename T>
Array2D<T> shift_bilinear(const Array2D<T>& in, double shift_x, double shift_y) {
    Array2D<T> out(in.rows(), in.cols(), T{});
    for (std::size_t r = 0; r < in.rows(); ++r) {
        for (std::size_t c = 0; c < in.cols(); ++c) {
            const double src_y = static_cast<double>(r) - shift_y;
            const double src_x = static_cast<double>(c) - shift_x;
            if (src_x < 0.0 || src_y < 0.0 || src_x >= in.cols() - 1 || src_y >= in.rows() - 1) {
                continue;
            }
            const std::size_t x0 = static_cast<std::size_t>(std::floor(src_x));
            const std::size_t y0 = static_cast<std::size_t>(std::floor(src_y));
            const double dx = src_x - static_cast<double>(x0);
            const double dy = src_y - static_cast<double>(y0);

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

Array2D<double> pinv_2x2(const Array2D<double>& mat, double rcond) {
    if (mat.rows() != 2 || mat.cols() != 2) {
        throw std::invalid_argument("pinv_2x2 expects 2x2");
    }
    Array2D<double> a = mat;
    const auto svd_res = svd(a);
    const double max_s = std::max(svd_res.s[0], svd_res.s[1]);
    Array2D<double> s_inv(2, 2, 0.0);
    for (std::size_t i = 0; i < 2; ++i) {
        if (svd_res.s[i] > rcond * max_s) {
            s_inv(i, i) = 1.0 / svd_res.s[i];
        }
    }
    Array2D<double> vt_t(2, 2, 0.0);
    for (std::size_t r = 0; r < 2; ++r) {
        for (std::size_t c = 0; c < 2; ++c) {
            vt_t(r, c) = svd_res.vt(c, r);
        }
    }
    const Array2D<double> u = svd_res.u;
    Array2D<double> temp = gemm(vt_t, s_inv);
    return gemm(temp, u, false, true);
}

} // namespace

PwpSolver::PwpSolver(const Array2D<double>& probes,
                     const Array2D<std::uint8_t>& control_mask,
                     const Array2D<std::uint8_t>& dm_mask,
                     std::size_t nframes,
                     double probe_amp,
                     double reg_cond,
                     double gain,
                     double dm_scale)
    : probes_(probes),
      control_mask_(control_mask),
      dm_mask_(dm_mask),
      nframes_(nframes),
      probe_amp_(probe_amp),
      reg_cond_(reg_cond),
      gain_(gain),
      dm_scale_(dm_scale) {}

void PwpSolver::set_model(EfcModel* model, double wavelength) {
    model_ = model;
    wavelength_ = wavelength;
}

void PwpSolver::set_jacobian(const Array2D<double>& jacobian) {
    jacobian_ = jacobian;
}

void PwpSolver::set_fp_shift(double x_shift, double y_shift) {
    fp_shift_ = std::make_pair(x_shift, y_shift);
}

void PwpSolver::set_e_fp_nom(const Array2D<std::complex<double>>& e_fp_nom) {
    e_fp_nom_ = e_fp_nom;
}

PwpResult PwpSolver::run(Stream2D& camsci,
                         Stream2D& dm,
                         const ImParams& im_params,
                         const ImParams& ref_params) {
    const std::vector<std::size_t> mask_idx = mask_indices(control_mask_);
    const std::size_t nmask = mask_idx.size();
    const std::size_t nprobes = probes_.rows();
    const std::size_t nact = static_cast<std::size_t>(std::sqrt(probes_.cols()));

    Array2D<double> current_command = dm.grab_latest();
    for (std::size_t i = 0; i < current_command.size(); ++i) {
        current_command.data()[i] *= dm_scale_;
    }

    std::vector<Array2D<double>> diff_ims;
    diff_ims.reserve(nprobes);
    std::vector<Array2D<std::complex<double>>> e_probes;
    e_probes.reserve(nprobes);

    std::vector<double> current_acts;
    current_acts.reserve(nmask);
    for (std::size_t i = 0; i < dm_mask_.size(); ++i) {
        if (dm_mask_.data()[i]) {
            current_acts.push_back(current_command.data()[i]);
        }
    }

    for (std::size_t i = 0; i < nprobes; ++i) {
        Array2D<double> dm_probe(nact, nact, 0.0);
        for (std::size_t r = 0; r < nact; ++r) {
            for (std::size_t c = 0; c < nact; ++c) {
                dm_probe(r, c) = probe_amp_ * probes_(i, r * nact + c);
            }
        }

        Array2D<double> cmd_pos(nact, nact, 0.0);
        Array2D<double> cmd_neg(nact, nact, 0.0);
        for (std::size_t idx = 0; idx < cmd_pos.size(); ++idx) {
            cmd_pos.data()[idx] = current_command.data()[idx] + dm_probe.data()[idx];
            cmd_neg.data()[idx] = current_command.data()[idx] - dm_probe.data()[idx];
        }

        Array2D<double> write_pos = cmd_pos;
        Array2D<double> write_neg = cmd_neg;
        for (std::size_t idx = 0; idx < write_pos.size(); ++idx) {
            write_pos.data()[idx] /= dm_scale_;
            write_neg.data()[idx] /= dm_scale_;
        }
        dm.write(write_pos);
        Array2D<double> im_pos = camsci.grab_mean(nframes_);
        dm.write(write_neg);
        Array2D<double> im_neg = camsci.grab_mean(nframes_);

        Array2D<double> diff(im_pos.rows(), im_pos.cols(), 0.0);
        for (std::size_t idx = 0; idx < diff.size(); ++idx) {
            diff.data()[idx] = im_pos.data()[idx] - im_neg.data()[idx];
        }
        Array2D<double> diff_ni = normalize_coro_im(diff, im_params, ref_params, 0.0);
        if (fp_shift_.has_value()) {
            diff_ni = shift_bilinear(diff_ni, fp_shift_->first, fp_shift_->second);
        }
        diff_ims.push_back(diff_ni);

        Array2D<std::complex<double>> e_probe(camsci.rows(), camsci.cols(), {0.0, 0.0});
        if (jacobian_) {
            std::vector<double> probe_acts;
            probe_acts.reserve(nmask);
            for (std::size_t idx = 0; idx < dm_mask_.size(); ++idx) {
                if (dm_mask_.data()[idx]) {
                    probe_acts.push_back(dm_probe.data()[idx]);
                }
            }
            const std::vector<double> e_probe_vec = gemv(*jacobian_, probe_acts);
            for (std::size_t k = 0; k < nmask; ++k) {
                const std::size_t idx = mask_idx[k];
                e_probe.data()[idx] = std::complex<double>(e_probe_vec[2 * k], e_probe_vec[2 * k + 1]);
            }
        } else if (model_) {
            if (!e_fp_nom_.has_value()) {
                e_fp_nom_ = model_->forward(current_acts, wavelength_, true);
            }
            const std::vector<double> probe_acts = current_acts;
            std::vector<double> probe_acts_sum = probe_acts;
            std::size_t act_idx = 0;
            for (std::size_t idx = 0; idx < dm_mask_.size(); ++idx) {
                if (dm_mask_.data()[idx]) {
                    probe_acts_sum[act_idx] += dm_probe.data()[idx];
                    ++act_idx;
                }
            }
            const auto e_with_probe = model_->forward(probe_acts_sum, wavelength_, true);
            for (std::size_t idx = 0; idx < e_probe.size(); ++idx) {
                e_probe.data()[idx] = e_with_probe.data()[idx] - e_fp_nom_.value().data()[idx];
            }
        } else {
            throw std::runtime_error("PWP requires a Jacobian or a model");
        }
        e_probes.push_back(e_probe);
    }

    Array2D<double> write_current = current_command;
    for (std::size_t idx = 0; idx < write_current.size(); ++idx) {
        write_current.data()[idx] /= dm_scale_;
    }
    dm.write(write_current);

    Array2D<std::complex<double>> e_est_2d(camsci.rows(), camsci.cols(), {0.0, 0.0});
    std::vector<std::complex<double>> e_est(nmask, {0.0, 0.0});

    for (std::size_t i = 0; i < nmask; ++i) {
        std::vector<double> delI(nprobes, 0.0);
        Array2D<double> H(nprobes, 2, 0.0);
        for (std::size_t p = 0; p < nprobes; ++p) {
            delI[p] = diff_ims[p].data()[mask_idx[i]];
            const std::complex<double> ep = e_probes[p].data()[mask_idx[i]];
            H(p, 0) = 4.0 * ep.real();
            H(p, 1) = 4.0 * ep.imag();
        }

        Array2D<double> HtH(2, 2, 0.0);
        for (std::size_t r = 0; r < 2; ++r) {
            for (std::size_t c = 0; c < 2; ++c) {
                double sum = 0.0;
                for (std::size_t p = 0; p < nprobes; ++p) {
                    sum += H(p, r) * H(p, c);
                }
                HtH(r, c) = sum;
            }
        }

        const Array2D<double> HtH_inv = pinv_2x2(HtH, reg_cond_);
        Array2D<double> Ht(2, nprobes, 0.0);
        for (std::size_t r = 0; r < 2; ++r) {
            for (std::size_t p = 0; p < nprobes; ++p) {
                Ht(r, p) = H(p, r);
            }
        }
        Array2D<double> Hinv = gemm(HtH_inv, Ht);
        std::vector<double> est(2, 0.0);
        for (std::size_t r = 0; r < 2; ++r) {
            double sum = 0.0;
            for (std::size_t p = 0; p < nprobes; ++p) {
                sum += Hinv(r, p) * delI[p];
            }
            est[r] = sum;
        }

        e_est[i] = std::complex<double>(est[0], est[1]) * gain_;
        e_est_2d.data()[mask_idx[i]] = e_est[i];
    }

    return {e_est_2d, e_est};
}

PwpResult PwpSolver::run_with_jacobian(Stream2D& camsci,
                                       Stream2D& dm,
                                       const ImParams& im_params,
                                       const ImParams& ref_params) {
    return run(camsci, dm, im_params, ref_params);
}

} // namespace lina
