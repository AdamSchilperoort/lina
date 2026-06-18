#include "lina/iefc.h"
#include "lina/coro_utils.h"
#include "lina/linalg.h"
#include "lina/props.h"
#include "lina/utils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <thread>

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

std::vector<double> command_to_actuators(
    const Array2D<double>& command,
    const Array2D<std::uint8_t>& dm_mask) {
    if (command.rows() != dm_mask.rows() || command.cols() != dm_mask.cols()) {
        throw std::invalid_argument("command_to_actuators shape mismatch");
    }
    std::vector<double> acts;
    acts.reserve(dm_mask.size());
    for (std::size_t i = 0; i < dm_mask.size(); ++i) {
        if (dm_mask.data()[i]) {
            acts.push_back(command.data()[i]);
        }
    }
    return acts;
}

Array2D<double> model_intensity_image(
    ControlModel& model,
    const Array2D<double>& total_command,
    bool use_vortex,
    double imax_ref) {
    const auto acts = command_to_actuators(total_command, model.dm_mask());
    const auto e_fp = model.forward(acts, model.wavelength(), use_vortex);
    const double norm = (imax_ref > 0.0) ? imax_ref : 1.0;
    Array2D<double> im(e_fp.rows(), e_fp.cols(), 0.0);
    for (std::size_t i = 0; i < e_fp.size(); ++i) {
        im.data()[i] = std::norm(e_fp.data()[i]) / norm;
    }
    return im;
}

Array2D<double> measure_probe_response_control_model(
    ControlModel& model,
    const Array2D<double>& base_command,
    const Array2D<double>& probe_modes,
    double probe_amplitude,
    std::size_t image_size,
    bool use_vortex,
    double imax_ref) {
    const std::size_t nprobes = probe_modes.rows();
    const std::size_t nact = static_cast<std::size_t>(std::sqrt(probe_modes.cols()));
    if (nact * nact != probe_modes.cols()) {
        throw std::invalid_argument("probe_modes second dimension must be square");
    }
    if (base_command.rows() != nact || base_command.cols() != nact) {
        throw std::invalid_argument("base_command shape mismatch");
    }
    if (probe_amplitude == 0.0) {
        throw std::invalid_argument("probe_amplitude must be non-zero");
    }

    Array2D<double> responses(nprobes, image_size, 0.0);

    for (std::size_t p = 0; p < nprobes; ++p) {
        Array2D<double> probe(nact, nact, 0.0);
        for (std::size_t r = 0; r < nact; ++r) {
            for (std::size_t c = 0; c < nact; ++c) {
                probe(r, c) = probe_amplitude * probe_modes(p, r * nact + c);
            }
        }

        Array2D<double> cmd_pos(nact, nact, 0.0);
        Array2D<double> cmd_neg(nact, nact, 0.0);
        for (std::size_t r = 0; r < nact; ++r) {
            for (std::size_t c = 0; c < nact; ++c) {
                cmd_pos(r, c) = base_command(r, c) + probe(r, c);
                cmd_neg(r, c) = base_command(r, c) - probe(r, c);
            }
        }

        const Array2D<double> im_pos = model_intensity_image(
            model, cmd_pos, use_vortex, imax_ref);
        const Array2D<double> im_neg = model_intensity_image(
            model, cmd_neg, use_vortex, imax_ref);
        if (im_pos.size() != image_size || im_neg.size() != image_size) {
            throw std::runtime_error("ControlModel image size mismatch in probe response");
        }

        for (std::size_t idx = 0; idx < im_pos.size(); ++idx) {
            responses(p, idx) = (im_pos.data()[idx] - im_neg.data()[idx]) /
                                (2.0 * probe_amplitude);
        }
    }
    return responses;
}

} // namespace

std::vector<Array2D<double>> measure_probe_response(Stream2D& camsci,
                                                    std::size_t ncamsci,
                                                    Stream2D& dm,
                                                    const ImParams& im_params,
                                                    const ImParams& ref_params,
                                                    const Array2D<double>& probe_modes,
                                                    double probe_amplitude,
                                                    double delay_s,
                                                    double dm_scale) {
    const std::size_t nprobes = probe_modes.rows();
    const std::size_t nact = static_cast<std::size_t>(std::sqrt(probe_modes.cols()));

    Array2D<double> current_command = dm.grab_latest();
    for (std::size_t j = 0; j < current_command.size(); ++j) {
        current_command.data()[j] *= dm_scale;
    }

    std::vector<Array2D<double>> responses;
    responses.reserve(nprobes);

    for (std::size_t i = 0; i < nprobes; ++i) {
        Array2D<double> probe(nact, nact, 0.0);
        for (std::size_t r = 0; r < nact; ++r) {
            for (std::size_t c = 0; c < nact; ++c) {
                probe(r, c) = probe_amplitude * probe_modes(i, r * nact + c);
            }
        }

        Array2D<double> cmd_pos(nact, nact, 0.0);
        Array2D<double> cmd_neg(nact, nact, 0.0);
        for (std::size_t r = 0; r < nact; ++r) {
            for (std::size_t c = 0; c < nact; ++c) {
                cmd_pos(r, c) = current_command(r, c) + probe(r, c);
                cmd_neg(r, c) = current_command(r, c) - probe(r, c);
            }
        }

        Array2D<double> write_pos = cmd_pos;
        for (std::size_t idx = 0; idx < write_pos.size(); ++idx) {
            write_pos.data()[idx] /= dm_scale;
        }
        dm.write(write_pos);
        std::this_thread::sleep_for(std::chrono::duration<double>(delay_s));
        Array2D<double> im_pos = camsci.grab_mean(ncamsci);

        Array2D<double> write_neg = cmd_neg;
        for (std::size_t idx = 0; idx < write_neg.size(); ++idx) {
            write_neg.data()[idx] /= dm_scale;
        }
        dm.write(write_neg);
        std::this_thread::sleep_for(std::chrono::duration<double>(delay_s));
        Array2D<double> im_neg = camsci.grab_mean(ncamsci);

        Array2D<double> diff(im_pos.rows(), im_pos.cols(), 0.0);
        for (std::size_t idx = 0; idx < diff.size(); ++idx) {
            diff.data()[idx] = im_pos.data()[idx] - im_neg.data()[idx];
        }

        Array2D<double> diff_ni = normalize_coro_im(diff, im_params, ref_params, 0.0);
        for (std::size_t idx = 0; idx < diff_ni.size(); ++idx) {
            diff_ni.data()[idx] /= (2.0 * probe_amplitude);
        }
        responses.push_back(diff_ni);
    }

    Array2D<double> write_current = current_command;
    for (std::size_t idx = 0; idx < write_current.size(); ++idx) {
        write_current.data()[idx] /= dm_scale;
    }
    dm.write(write_current);
    return responses;
}

Array2D<double> calibrate(Stream2D& camsci,
                          std::size_t ncamsci,
                          Stream2D& dm,
                          const ImParams& im_params,
                          const ImParams& ref_params,
                          const Array2D<std::uint8_t>& control_mask,
                          double probe_amplitude,
                          const Array2D<double>& probe_modes,
                          double calibration_amplitude,
                          const Array2D<double>& calibration_modes,
                          double delay_s,
                          double dm_scale) {
    const std::size_t nact = static_cast<std::size_t>(std::sqrt(probe_modes.cols()));
    const std::size_t nprobes = probe_modes.rows();
    const std::size_t nmodes = calibration_modes.rows();

    const std::vector<std::size_t> mask_idx = mask_indices(control_mask);
    const std::size_t nmask = mask_idx.size();

    Array2D<double> current_command = dm.grab_latest();
    for (std::size_t j = 0; j < current_command.size(); ++j) {
        current_command.data()[j] *= dm_scale;
    }

    Array2D<double> response_matrix(nmodes, nprobes * nmask, 0.0);

    for (std::size_t i = 0; i < nmodes; ++i) {
        Array2D<double> response(nprobes, nact * nact, 0.0);
        for (int s : {-1, 1}) {
            Array2D<double> calib_mode(nact, nact, 0.0);
            for (std::size_t r = 0; r < nact; ++r) {
                for (std::size_t c = 0; c < nact; ++c) {
                    calib_mode(r, c) = calibration_amplitude * calibration_modes(i, r * nact + c);
                }
            }

            Array2D<double> cmd(nact, nact, 0.0);
            for (std::size_t r = 0; r < nact; ++r) {
                for (std::size_t c = 0; c < nact; ++c) {
                    cmd(r, c) = current_command(r, c) + s * calib_mode(r, c);
                }
            }
            Array2D<double> write_cmd = cmd;
            for (std::size_t idx = 0; idx < write_cmd.size(); ++idx) {
                write_cmd.data()[idx] /= dm_scale;
            }
            dm.write(write_cmd);
            std::this_thread::sleep_for(std::chrono::duration<double>(delay_s));

            const auto probed = measure_probe_response(
                camsci, ncamsci, dm, im_params, ref_params, probe_modes, probe_amplitude, delay_s, dm_scale);

            for (std::size_t p = 0; p < nprobes; ++p) {
                for (std::size_t idx = 0; idx < probed[p].size(); ++idx) {
                    response(p, idx) += static_cast<double>(s) * probed[p].data()[idx] /
                                        (2.0 * calibration_amplitude);
                }
            }
        }
        Array2D<double> write_current = current_command;
        for (std::size_t idx = 0; idx < write_current.size(); ++idx) {
            write_current.data()[idx] /= dm_scale;
        }
        dm.write(write_current);

        for (std::size_t p = 0; p < nprobes; ++p) {
            for (std::size_t k = 0; k < nmask; ++k) {
                response_matrix(i, p * nmask + k) =
                    response(p, mask_idx[k]);
            }
        }
    }
    return response_matrix;
}

IefcCalibrationResult calibrate_control_model(
    ControlModel& model,
    const Array2D<std::uint8_t>& control_mask,
    double probe_amplitude,
    const Array2D<double>& probe_modes,
    double calibration_amplitude,
    const Array2D<double>& calibration_modes,
    const std::vector<double>& scale_factors,
    std::optional<Array2D<double>> initial_command,
    bool use_vortex,
    double imax_ref,
    const std::function<void(std::size_t, std::size_t)>& progress) {
    const std::size_t ncam = control_mask.rows();
    if (control_mask.cols() != ncam) {
        throw std::invalid_argument("control_mask must be square");
    }
    const std::size_t nact = static_cast<std::size_t>(std::sqrt(probe_modes.cols()));
    if (nact * nact != probe_modes.cols()) {
        throw std::invalid_argument("probe_modes second dimension must be square");
    }
    if (calibration_modes.cols() != nact * nact) {
        throw std::invalid_argument("calibration_modes second dimension mismatch");
    }
    const std::size_t nprobes = probe_modes.rows();
    const std::size_t nmodes = calibration_modes.rows();
    if (!scale_factors.empty() && scale_factors.size() != nmodes) {
        throw std::invalid_argument("scale_factors length must match calibration modes");
    }
    if (calibration_amplitude == 0.0) {
        throw std::invalid_argument("calibration_amplitude must be non-zero");
    }

    Array2D<double> base0 = initial_command.value_or(Array2D<double>(nact, nact, 0.0));
    if (base0.rows() != nact || base0.cols() != nact) {
        throw std::invalid_argument("initial_command shape mismatch");
    }

    const std::vector<std::size_t> mask_idx = mask_indices(control_mask);
    const std::size_t nmask = mask_idx.size();

    IefcCalibrationResult out;
    out.response_matrix = Array2D<double>(nprobes * nmask, nmodes, 0.0);
    out.response_cube.reserve(nmodes);
    out.nprobes = nprobes;
    out.ncamsci = ncam;

    for (std::size_t m = 0; m < nmodes; ++m) {
        Array2D<double> dm_mode(nact, nact, 0.0);
        for (std::size_t r = 0; r < nact; ++r) {
            for (std::size_t c = 0; c < nact; ++c) {
                dm_mode(r, c) = calibration_modes(m, r * nact + c);
            }
        }

        const double amp = calibration_amplitude *
            (scale_factors.empty() ? 1.0 : scale_factors[m]);
        if (amp == 0.0) {
            throw std::invalid_argument("effective calibration amplitude cannot be zero");
        }

        Array2D<double> response(nprobes, ncam * ncam, 0.0);
        for (int s : {1, -1}) {
            Array2D<double> base = base0;
            for (std::size_t idx = 0; idx < base.size(); ++idx) {
                base.data()[idx] += static_cast<double>(s) * amp * dm_mode.data()[idx];
            }

            const Array2D<double> probed = measure_probe_response_control_model(
                model,
                base,
                probe_modes,
                probe_amplitude,
                ncam * ncam,
                use_vortex,
                imax_ref);

            const double coeff = static_cast<double>(s) / (2.0 * amp);
            for (std::size_t idx = 0; idx < response.size(); ++idx) {
                response.data()[idx] += coeff * probed.data()[idx];
            }
        }

        out.response_cube.push_back(response);
        for (std::size_t p = 0; p < nprobes; ++p) {
            for (std::size_t k = 0; k < nmask; ++k) {
                out.response_matrix(p * nmask + k, m) = response(p, mask_idx[k]);
            }
        }

        if (progress) {
            progress(m + 1, nmodes);
        }
    }

    return out;
}

void run(IefcData& iefc_data,
         Stream2D& camsci,
         std::size_t ncamsci,
         Stream2D& dm,
         const ImParams& im_params,
         const ImParams& ref_params,
         const Array2D<double>& dark_im,
         const Array2D<double>& control_matrix,
         double probe_amplitude,
         const Array2D<double>& probe_modes,
         const Array2D<double>& calib_modes,
         const Array2D<std::uint8_t>& control_mask,
         double delay_s,
         std::size_t num_iterations,
         double gain,
         double leakage,
         double dm_scale) {
    const std::size_t nact = static_cast<std::size_t>(std::sqrt(probe_modes.cols()));
    const std::size_t nmodes = calib_modes.rows();
    const std::vector<std::size_t> mask_idx = mask_indices(control_mask);
    const std::size_t nmask = mask_idx.size();

    Array2D<double> modal_matrix(nact * nact, nmodes, 0.0);
    for (std::size_t m = 0; m < nmodes; ++m) {
        for (std::size_t r = 0; r < nact; ++r) {
            for (std::size_t c = 0; c < nact; ++c) {
                modal_matrix(r * nact + c, m) = calib_modes(m, r * nact + c);
            }
        }
    }

    Array2D<double> total_command(nact, nact, 0.0);
    if (!iefc_data.commands.empty()) {
        total_command = iefc_data.commands.back();
    }

    for (std::size_t i = 0; i < num_iterations; ++i) {
        const auto diff_ims = measure_probe_response(
            camsci, ncamsci, dm, im_params, ref_params, probe_modes, probe_amplitude, delay_s, dm_scale);

        std::vector<double> measurement_vector;
        measurement_vector.reserve(diff_ims.size() * nmask);
        for (const auto& im : diff_ims) {
            for (std::size_t k = 0; k < nmask; ++k) {
                measurement_vector.push_back(im.data()[mask_idx[k]]);
            }
        }

        std::vector<double> modal_coeff = gemv(control_matrix, measurement_vector);
        for (double& v : modal_coeff) {
            v = -v;
        }
        std::vector<double> del_command_vec = gemv(modal_matrix, modal_coeff);

        Array2D<double> del_command(nact, nact, 0.0);
        for (std::size_t r = 0; r < nact; ++r) {
            for (std::size_t c = 0; c < nact; ++c) {
                del_command(r, c) = gain * del_command_vec[r * nact + c];
            }
        }

        for (std::size_t idx = 0; idx < total_command.size(); ++idx) {
            total_command.data()[idx] = (1.0 - leakage) * total_command.data()[idx] +
                                         del_command.data()[idx];
        }

        Array2D<double> write_cmd = total_command;
        for (std::size_t idx = 0; idx < write_cmd.size(); ++idx) {
            write_cmd.data()[idx] /= dm_scale;
        }
        dm.write(write_cmd);
        std::this_thread::sleep_for(std::chrono::duration<double>(delay_s));

        const Array2D<double> coro_im = camsci.grab_mean(ncamsci);
        const Array2D<double> coro_im_ni = normalize_coro_im(coro_im, im_params, ref_params, dark_im);
        const ContrastResult contrast = compute_contrast(coro_im_ni, control_mask);

        iefc_data.raw_images.push_back(coro_im);
        iefc_data.dark_images.push_back(dark_im);
        iefc_data.ni_images.push_back(coro_im_ni);
        iefc_data.contrasts.push_back(contrast.contrast);
        iefc_data.commands.push_back(total_command);
        iefc_data.del_commands.push_back(del_command);
    }
}

std::vector<double> compute_hadamard_scale_factors(const Array2D<double>& had_modes,
                                                   double scale_exp,
                                                   double scale_thresh,
                                                   double iwa,
                                                   double owa,
                                                   std::size_t oversamp) {
    const std::size_t nact = static_cast<std::size_t>(std::sqrt(had_modes.cols()));
    const std::size_t nmodes = had_modes.rows();

    std::vector<double> sum_vals(nmodes, 0.0);
    std::vector<double> max_vals(nmodes, 0.0);

    const std::size_t n = nact * oversamp;
    for (std::size_t i = 0; i < nmodes; ++i) {
        Array2D<std::complex<double>> mode(nact, nact, {0.0, 0.0});
        for (std::size_t r = 0; r < nact; ++r) {
            for (std::size_t c = 0; c < nact; ++c) {
                mode(r, c) = had_modes(i, r * nact + c);
            }
        }
        Array2D<std::complex<double>> padded = pad_or_crop(mode, n);
        const auto ft = fft(padded);

        const double pixel_scale = 1.0 / static_cast<double>(oversamp);
        const double half = static_cast<double>(n) / 2.0;

        double sum = 0.0;
        double maxv = 0.0;
        for (std::size_t r = 0; r < n; ++r) {
            for (std::size_t c = 0; c < n; ++c) {
                const double x = (static_cast<double>(c) - half + 0.5) * pixel_scale;
                const double y = (static_cast<double>(r) - half + 0.5) * pixel_scale;
                const double rr = std::hypot(x, y);
                if (rr > iwa && rr < owa) {
                    const double mag = std::abs(ft(r, c));
                    sum += mag;
                    maxv = std::max(maxv, mag * mag);
                }
            }
        }
        sum_vals[i] = sum;
        max_vals[i] = maxv;
    }

    const double biggest_max = *std::max_element(max_vals.begin(), max_vals.end());
    std::vector<double> scale_factors(nmodes, 1.0);
    for (std::size_t i = 0; i < nmodes; ++i) {
        const double scale = std::pow(biggest_max / max_vals[i], scale_exp);
        scale_factors[i] = std::min(scale, scale_thresh);
    }
    return scale_factors;
}

} // namespace lina
