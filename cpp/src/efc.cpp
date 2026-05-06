#include "lina/efc.h"
#include "lina/coro_utils.h"
#include "lina/linalg.h"

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

Array2D<double> build_command_from_acts(const Array2D<std::uint8_t>& dm_mask,
                                        const std::vector<double>& acts,
                                        std::size_t nact) {
    Array2D<double> cmd(nact, nact, 0.0);
    std::size_t idx = 0;
    for (std::size_t i = 0; i < dm_mask.size(); ++i) {
        if (dm_mask.data()[i]) {
            cmd.data()[i] = acts[idx++];
        }
    }
    return cmd;
}

std::vector<double> extract_acts_from_command(const Array2D<std::uint8_t>& dm_mask,
                                              const Array2D<double>& cmd) {
    std::vector<double> acts;
    acts.reserve(dm_mask.size());
    for (std::size_t i = 0; i < dm_mask.size(); ++i) {
        if (dm_mask.data()[i]) {
            acts.push_back(cmd.data()[i]);
        }
    }
    return acts;
}

} // namespace

Array2D<double> compute_jacobian(EfcModel& model,
                                 const Array2D<std::uint8_t>& control_mask,
                                 double amp,
                                 const std::vector<double>* current_acts) {
    const std::size_t nacts = model.nacts();
    const std::vector<std::size_t> mask_idx = mask_indices(control_mask);
    const std::size_t nmask = mask_idx.size();

    std::vector<double> acts = current_acts ? *current_acts : std::vector<double>(nacts, 0.0);

    Array2D<double> jac(2 * nmask, nacts, 0.0);

    for (std::size_t i = 0; i < nacts; ++i) {
        std::vector<double> act_poke(nacts, 0.0);
        act_poke[i] = amp;

        std::vector<double> acts_pos(nacts, 0.0);
        std::vector<double> acts_neg(nacts, 0.0);
        for (std::size_t j = 0; j < nacts; ++j) {
            acts_pos[j] = acts[j] + act_poke[j];
            acts_neg[j] = acts[j] - act_poke[j];
        }

        const auto e_pos = model.forward(acts_pos, model.wavelength(), true);
        const auto e_neg = model.forward(acts_neg, model.wavelength(), true);

        for (std::size_t k = 0; k < nmask; ++k) {
            const std::size_t idx = mask_idx[k];
            const std::complex<double> resp = (e_pos.data()[idx] - e_neg.data()[idx]) / (2.0 * amp);
            jac(2 * k, i) = resp.real();
            jac(2 * k + 1, i) = resp.imag();
        }
    }

    return jac;
}

std::pair<Array2D<double>, std::vector<Array2D<double>>> compute_jacobian_bb(
    std::vector<EfcModel*>& models,
    const Array2D<std::uint8_t>& control_mask,
    double amp,
    const std::vector<double>* current_acts) {
    const std::size_t nwaves = models.size();
    const std::vector<std::size_t> mask_idx = mask_indices(control_mask);
    const std::size_t nmask = mask_idx.size();
    const std::size_t nacts = models.front()->nacts();

    Array2D<double> jac(nwaves * 2 * nmask, nacts, 0.0);
    std::vector<Array2D<double>> mono_jacs;
    mono_jacs.reserve(nwaves);

    for (std::size_t i = 0; i < nwaves; ++i) {
        Array2D<double> mono = compute_jacobian(*models[i], control_mask, amp, current_acts);
        mono_jacs.push_back(mono);
        for (std::size_t r = 0; r < mono.rows(); ++r) {
            for (std::size_t c = 0; c < mono.cols(); ++c) {
                jac(i * 2 * nmask + r, c) = mono(r, c);
            }
        }
    }

    return {jac, mono_jacs};
}

void run(EfcData& efc_data,
         Stream2D& camsci,
         Stream2D& dm,
         const ImParams& im_params,
         const ImParams& ref_params,
         std::size_t nframes,
         const Array2D<double>& dark_im,
         const Array2D<std::uint8_t>& control_mask,
         const Array2D<std::uint8_t>& dm_mask,
         const Array2D<double>& control_matrix,
         PwpEstimator& pwp,
         std::size_t nitr,
         double gain,
         double leakage,
         double delay_s,
         double dm_scale) {
    const std::vector<std::size_t> mask_idx = mask_indices(control_mask);
    const std::size_t nmask = mask_idx.size();

    Array2D<double> del_command(dm.rows(), dm.cols(), 0.0);
    std::vector<double> e_ab_vec(2 * nmask, 0.0);

    for (std::size_t i = 0; i < nitr; ++i) {
        Array2D<double> current_command = dm.grab_latest();
        for (std::size_t j = 0; j < current_command.size(); ++j) {
            current_command.data()[j] *= dm_scale;
        }

        const PwpResult pwp_result = pwp.run(camsci, dm, im_params, ref_params);

        for (std::size_t k = 0; k < nmask; ++k) {
            const std::complex<double> val = pwp_result.field_vec[k];
            e_ab_vec[2 * k] = val.real();
            e_ab_vec[2 * k + 1] = val.imag();
        }

        const std::vector<double> del_acts = gemv(control_matrix, e_ab_vec);

        std::vector<double> scaled_del_acts(del_acts.size(), 0.0);
        for (std::size_t k = 0; k < del_acts.size(); ++k) {
            scaled_del_acts[k] = -gain * del_acts[k];
        }

        del_command = build_command_from_acts(dm_mask, scaled_del_acts, dm.rows());

        Array2D<double> total_command(dm.rows(), dm.cols(), 0.0);
        for (std::size_t j = 0; j < total_command.size(); ++j) {
            total_command.data()[j] = (1.0 - leakage) * current_command.data()[j] + del_command.data()[j];
        }
        for (std::size_t j = 0; j < total_command.size(); ++j) {
            total_command.data()[j] /= dm_scale;
        }
        dm.write(total_command);
        std::this_thread::sleep_for(std::chrono::duration<double>(delay_s));

        const Array2D<double> metric_im = camsci.grab_mean(nframes);
        const Array2D<double> metric_im_ni = normalize_coro_im(metric_im, im_params, ref_params, dark_im);
        const ContrastResult contrast = compute_contrast(metric_im_ni, control_mask);

        efc_data.images.push_back(metric_im_ni);
        efc_data.contrasts.push_back(contrast.contrast);
        efc_data.efields.push_back(pwp_result.field);
        efc_data.commands.push_back(total_command);
        efc_data.del_commands.push_back(del_command);
    }
}

void run_bb(EfcData& efc_data,
            Stream2D& camsci,
            Stream2D& dm,
            const ImParams& im_params,
            const ImParams& ref_params,
            std::size_t nframes,
            const Array2D<double>& dark_im,
            const Array2D<std::uint8_t>& control_mask,
            const Array2D<std::uint8_t>& dm_mask,
            const Array2D<double>& control_matrix,
            PwpEstimator& pwp,
            std::size_t nwaves,
            std::size_t nitr,
            double gain,
            double leakage,
            double dm_delay_s,
            double filter_delay_s,
            double dm_scale) {
    const std::vector<std::size_t> mask_idx = mask_indices(control_mask);
    const std::size_t nmask = mask_idx.size();

    Array2D<double> del_command(dm.rows(), dm.cols(), 0.0);
    std::vector<double> e_ab_vec(nwaves * 2 * nmask, 0.0);

    for (std::size_t i = 0; i < nitr; ++i) {
        Array2D<double> current_command = dm.grab_latest();
        for (std::size_t j = 0; j < current_command.size(); ++j) {
            current_command.data()[j] *= dm_scale;
        }

        for (std::size_t w = 0; w < nwaves; ++w) {
            std::this_thread::sleep_for(std::chrono::duration<double>(filter_delay_s));
            const PwpResult pwp_result = pwp.run_with_jacobian(camsci, dm, im_params, ref_params);
            for (std::size_t k = 0; k < nmask; ++k) {
                const std::complex<double> val = pwp_result.field_vec[k];
                e_ab_vec[w * 2 * nmask + 2 * k] = val.real();
                e_ab_vec[w * 2 * nmask + 2 * k + 1] = val.imag();
            }
        }

        const std::vector<double> del_acts = gemv(control_matrix, e_ab_vec);
        std::vector<double> scaled_del_acts(del_acts.size(), 0.0);
        for (std::size_t k = 0; k < del_acts.size(); ++k) {
            scaled_del_acts[k] = -gain * del_acts[k];
        }
        del_command = build_command_from_acts(dm_mask, scaled_del_acts, dm.rows());

        Array2D<double> total_command(dm.rows(), dm.cols(), 0.0);
        for (std::size_t j = 0; j < total_command.size(); ++j) {
            total_command.data()[j] = (1.0 - leakage) * current_command.data()[j] + del_command.data()[j];
            total_command.data()[j] /= dm_scale;
        }
        dm.write(total_command);
        std::this_thread::sleep_for(std::chrono::duration<double>(dm_delay_s));

        const Array2D<double> metric_im = camsci.grab_mean(nframes);
        const Array2D<double> metric_im_ni = normalize_coro_im(metric_im, im_params, ref_params, dark_im);
        const ContrastResult contrast = compute_contrast(metric_im_ni, control_mask);

        efc_data.images.push_back(metric_im_ni);
        efc_data.contrasts.push_back(contrast.contrast);
        efc_data.commands.push_back(total_command);
        efc_data.del_commands.push_back(del_command);
    }
}

} // namespace lina
