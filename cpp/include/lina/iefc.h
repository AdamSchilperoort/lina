#pragma once

#include "lina/array.h"
#include "lina/coro_utils.h"
#include "lina/stream.h"

#include <cstddef>
#include <vector>

namespace lina {

struct IefcData {
    std::vector<Array2D<double>> raw_images;
    std::vector<Array2D<double>> dark_images;
    std::vector<Array2D<double>> ni_images;
    std::vector<double> contrasts;
    std::vector<Array2D<double>> commands;
    std::vector<Array2D<double>> del_commands;
};

std::vector<Array2D<double>> measure_probe_response(Stream2D& camsci,
                                                    std::size_t ncamsci,
                                                    Stream2D& dm,
                                                    const ImParams& im_params,
                                                    const ImParams& ref_params,
                                                    const Array2D<double>& probe_modes,
                                                    double probe_amplitude,
                                                    double delay_s = 0.01,
                                                    double dm_scale = 1e-6);

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
                          double delay_s = 0.01,
                          double dm_scale = 1e-6);

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
         double delay_s = 0.01,
         std::size_t num_iterations = 3,
         double gain = 0.75,
         double leakage = 0.0,
         double dm_scale = 1e-6);

std::vector<double> compute_hadamard_scale_factors(const Array2D<double>& had_modes,
                                                   double scale_exp = 1.0 / 6.0,
                                                   double scale_thresh = 4.0,
                                                   double iwa = 2.5,
                                                   double owa = 13.0,
                                                   std::size_t oversamp = 4);

} // namespace lina
