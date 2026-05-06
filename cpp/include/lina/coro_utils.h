#pragma once

#include "lina/array.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace lina {

struct ImParams {
    double exp_time = 1.0;
    double gain = 0.0;
    double atten = 0.0;
    double Imax = 1.0;
};

Array2D<double> normalize_coro_im(const Array2D<double>& raw_im,
                                  const ImParams& im_params,
                                  const ImParams& ref_params,
                                  double dark_im = 0.0);

Array2D<double> normalize_coro_im(const Array2D<double>& raw_im,
                                  const ImParams& im_params,
                                  const ImParams& ref_params,
                                  const Array2D<double>& dark_im);

struct ContrastResult {
    double contrast = 0.0;
    std::size_t n_mask = 0;
    std::size_t n_positive = 0;
};

ContrastResult compute_contrast(const Array2D<double>& ni_im,
                                const std::vector<std::uint8_t>& mask);

ContrastResult compute_contrast(const Array2D<double>& ni_im,
                                const Array2D<std::uint8_t>& mask);

} // namespace lina
