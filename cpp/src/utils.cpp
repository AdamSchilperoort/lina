#include "lina/utils.h"

#include <cmath>
#include <numeric>

namespace lina {

double mean(const Array2D<double>& array,
            const std::vector<std::uint8_t>* mask) {
    const std::size_t rows = array.rows();
    const std::size_t cols = array.cols();
    double sum = 0.0;
    std::size_t count = 0;

    if (mask) {
        if (mask->size() != rows * cols) {
            throw std::invalid_argument("mask size mismatch");
        }
        for (std::size_t r = 0; r < rows; ++r) {
            for (std::size_t c = 0; c < cols; ++c) {
                const std::size_t idx = r * cols + c;
                if ((*mask)[idx]) {
                    sum += array(r, c);
                    ++count;
                }
            }
        }
    } else {
        for (std::size_t r = 0; r < rows; ++r) {
            for (std::size_t c = 0; c < cols; ++c) {
                sum += array(r, c);
                ++count;
            }
        }
    }

    return count == 0 ? 0.0 : sum / static_cast<double>(count);
}

double rms(const Array2D<double>& array,
           const std::vector<std::uint8_t>* mask) {
    const std::size_t rows = array.rows();
    const std::size_t cols = array.cols();
    double sum_sq = 0.0;
    std::size_t count = 0;

    if (mask) {
        if (mask->size() != rows * cols) {
            throw std::invalid_argument("mask size mismatch");
        }
        for (std::size_t r = 0; r < rows; ++r) {
            for (std::size_t c = 0; c < cols; ++c) {
                const std::size_t idx = r * cols + c;
                if ((*mask)[idx]) {
                    const double v = array(r, c);
                    sum_sq += v * v;
                    ++count;
                }
            }
        }
    } else {
        for (std::size_t r = 0; r < rows; ++r) {
            for (std::size_t c = 0; c < cols; ++c) {
                const double v = array(r, c);
                sum_sq += v * v;
                ++count;
            }
        }
    }

    return count == 0 ? 0.0 : std::sqrt(sum_sq / static_cast<double>(count));
}

std::pair<Array2D<double>, Array2D<double>> make_grid(std::size_t npix,
                                                      double pixelscale,
                                                      bool half_shift) {
    Array2D<double> x(npix, npix, 0.0);
    Array2D<double> y(npix, npix, 0.0);

    const double offset = half_shift ? 0.5 : 0.0;
    const double center = static_cast<double>(npix) / 2.0 - offset;

    for (std::size_t r = 0; r < npix; ++r) {
        for (std::size_t c = 0; c < npix; ++c) {
            const double yy = (static_cast<double>(r) - center) * pixelscale;
            const double xx = (static_cast<double>(c) - center) * pixelscale;
            y(r, c) = yy;
            x(r, c) = xx;
        }
    }

    return {x, y};
}

Array2D<std::uint8_t> create_annular_mask(std::size_t n,
                                          double pixelscale,
                                          double irad,
                                          double orad,
                                          double edge,
                                          double x_shift,
                                          double y_shift,
                                          double rotation_deg) {
    const double half = static_cast<double>(n) / 2.0;
    Array2D<std::uint8_t> mask(n, n, 0);

    const double radians = rotation_deg * M_PI / 180.0;
    const double cos_r = std::cos(radians);
    const double sin_r = std::sin(radians);

    for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t c = 0; c < n; ++c) {
            const double x = (static_cast<double>(c) - half + 0.5) * pixelscale;
            const double y = (static_cast<double>(r) - half + 0.5) * pixelscale;

            const double xr = (x - x_shift) * cos_r + (y - y_shift) * sin_r;
            const double yr = -(x - x_shift) * sin_r + (y - y_shift) * cos_r;

            const double rr = std::hypot(xr, yr);
            // Always apply edge as a real cut on xr. Pass a very-negative
            // value (the default) to disable the cut. This matches Python
            // semantics where `edge=0` means "filter at x>0" and
            // `edge=None` (here mapped to default) means "no filter".
            if (rr > irad && rr < orad && xr > edge) {
                mask(r, c) = 1;
            }
        }
    }
    return mask;
}

Array2D<std::uint8_t> create_annular_focal_plane_mask(std::size_t npsf,
                                                      double psf_pixelscale,
                                                      double irad,
                                                      double orad,
                                                      double edge,
                                                      const char* centering,
                                                      double rotation_deg,
                                                      double x_shift,
                                                      double y_shift) {
    const double half = static_cast<double>(npsf) / 2.0;
    const double offset = (std::string(centering) == "even") ? 0.5 : 0.0;

    Array2D<std::uint8_t> mask(npsf, npsf, 0);
    const double radians = rotation_deg * M_PI / 180.0;
    const double cos_r = std::cos(radians);
    const double sin_r = std::sin(radians);

    for (std::size_t r = 0; r < npsf; ++r) {
        for (std::size_t c = 0; c < npsf; ++c) {
            const double x = (static_cast<double>(c) - half + offset) * psf_pixelscale;
            const double y = (static_cast<double>(r) - half + offset) * psf_pixelscale;

            const double xr = (x - x_shift) * cos_r + (y - y_shift) * sin_r;
            const double yr = -(x - x_shift) * sin_r + (y - y_shift) * cos_r;

            const double rr = std::hypot(xr, yr);
            if (rr > irad && rr < orad && xr > edge) {
                mask(r, c) = 1;
            }
        }
    }
    return mask;
}

} // namespace lina
