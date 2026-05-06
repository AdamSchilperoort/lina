#include "lina/iefc.h"
#include "lina/coro_utils.h"
#include "lina/linalg.h"
#include "lina/props.h"
#include "lina/utils.h"

#include <algorithm>
#include <chrono>
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
