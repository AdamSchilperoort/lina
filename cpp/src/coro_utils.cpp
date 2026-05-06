#include "lina/coro_utils.h"

#include <cmath>
#include <stdexcept>

namespace lina {

Array2D<double> normalize_coro_im(const Array2D<double>& raw_im,
                                  const ImParams& im_params,
                                  const ImParams& ref_params,
                                  double dark_im) {
    const double exp_time_factor =
        ref_params.exp_time / im_params.exp_time;
    const double gain_factor =
        std::pow(10.0, ref_params.gain / 20.0 * 0.1) /
        std::pow(10.0, im_params.gain / 20.0 * 0.1);
    const double fiber_atten_factor =
        std::pow(10.0, -ref_params.atten / 10.0) /
        std::pow(10.0, -im_params.atten / 10.0);

    Array2D<double> out(raw_im.rows(), raw_im.cols(), 0.0);
    for (std::size_t r = 0; r < raw_im.rows(); ++r) {
        for (std::size_t c = 0; c < raw_im.cols(); ++c) {
            const double ds = raw_im(r, c) - dark_im;
            out(r, c) = ds * exp_time_factor * gain_factor *
                        fiber_atten_factor / ref_params.Imax;
        }
    }
    return out;
}

Array2D<double> normalize_coro_im(const Array2D<double>& raw_im,
                                  const ImParams& im_params,
                                  const ImParams& ref_params,
                                  const Array2D<double>& dark_im) {
    if (raw_im.rows() != dark_im.rows() || raw_im.cols() != dark_im.cols()) {
        throw std::invalid_argument("dark_im size mismatch");
    }
    const double exp_time_factor =
        ref_params.exp_time / im_params.exp_time;
    const double gain_factor =
        std::pow(10.0, ref_params.gain / 20.0 * 0.1) /
        std::pow(10.0, im_params.gain / 20.0 * 0.1);
    const double fiber_atten_factor =
        std::pow(10.0, -ref_params.atten / 10.0) /
        std::pow(10.0, -im_params.atten / 10.0);

    Array2D<double> out(raw_im.rows(), raw_im.cols(), 0.0);
    for (std::size_t r = 0; r < raw_im.rows(); ++r) {
        for (std::size_t c = 0; c < raw_im.cols(); ++c) {
            const double ds = raw_im(r, c) - dark_im(r, c);
            out(r, c) = ds * exp_time_factor * gain_factor *
                        fiber_atten_factor / ref_params.Imax;
        }
    }
    return out;
}

ContrastResult compute_contrast(const Array2D<double>& ni_im,
                                const std::vector<std::uint8_t>& mask) {
    const std::size_t rows = ni_im.rows();
    const std::size_t cols = ni_im.cols();
    if (mask.size() != rows * cols) {
        throw std::invalid_argument("mask size mismatch");
    }

    double sum = 0.0;
    std::size_t count = 0;
    std::size_t n_mask = 0;
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            const std::size_t idx = r * cols + c;
            if (!mask[idx]) {
                continue;
            }
            ++n_mask;
            const double val = ni_im(r, c);
            if (val > 0.0) {
                sum += val;
                ++count;
            }
        }
    }

    ContrastResult result;
    result.contrast = count == 0 ? 0.0 : sum / static_cast<double>(count);
    result.n_mask = n_mask;
    result.n_positive = count;
    return result;
}

ContrastResult compute_contrast(const Array2D<double>& ni_im,
                                const Array2D<std::uint8_t>& mask) {
    const std::size_t rows = ni_im.rows();
    const std::size_t cols = ni_im.cols();
    if (mask.rows() != rows || mask.cols() != cols) {
        throw std::invalid_argument("mask size mismatch");
    }

    double sum = 0.0;
    std::size_t count = 0;
    std::size_t n_mask = 0;
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            if (!mask(r, c)) {
                continue;
            }
            ++n_mask;
            const double val = ni_im(r, c);
            if (val > 0.0) {
                sum += val;
                ++count;
            }
        }
    }

    ContrastResult result;
    result.contrast = count == 0 ? 0.0 : sum / static_cast<double>(count);
    result.n_mask = n_mask;
    result.n_positive = count;
    return result;
}

} // namespace lina
