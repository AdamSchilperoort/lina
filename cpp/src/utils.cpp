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

namespace {

// Nearest-neighbour raster rotate matching scipy.ndimage.rotate(
//   input, angle_deg, reshape=False, order=0, mode='constant', cval=0).
// For each output index o, the source index is i = R @ (o - center) + center
// with R = [[cos, sin], [-sin, cos]] and center = (shape - 1) / 2, sampled with
// floor(coord + 0.5) (scipy's order-0 spline rounding). This reproduces lina's
// exact pixels for axis-aligned rotations (e.g. 90 deg) and is within a couple
// of boundary pixels of scipy for arbitrary angles (inherent nearest-neighbour
// tie-breaking).
Array2D<std::uint8_t> ndimage_rotate_nn(const Array2D<std::uint8_t>& in,
                                        double angle_deg) {
    const std::size_t n0 = in.rows();
    const std::size_t n1 = in.cols();
    const double a = angle_deg * M_PI / 180.0;
    const double c = std::cos(a);
    const double s = std::sin(a);
    const double cen0 = (static_cast<double>(n0) - 1.0) / 2.0;
    const double cen1 = (static_cast<double>(n1) - 1.0) / 2.0;

    Array2D<std::uint8_t> out(n0, n1, 0);
    for (std::size_t r = 0; r < n0; ++r) {
        for (std::size_t col = 0; col < n1; ++col) {
            const double o0 = static_cast<double>(r) - cen0;
            const double o1 = static_cast<double>(col) - cen1;
            const double i0 = c * o0 + s * o1 + cen0;
            const double i1 = -s * o0 + c * o1 + cen1;
            const long ir = static_cast<long>(std::floor(i0 + 0.5));
            const long ic = static_cast<long>(std::floor(i1 + 0.5));
            if (ir >= 0 && ir < static_cast<long>(n0) &&
                ic >= 0 && ic < static_cast<long>(n1)) {
                out(r, col) = in(static_cast<std::size_t>(ir),
                                 static_cast<std::size_t>(ic));
            }
        }
    }
    return out;
}

// Nearest-neighbour raster shift matching scipy.ndimage.shift(
//   input, (y_shift, x_shift), order=0, mode='constant', cval=0).
Array2D<std::uint8_t> ndimage_shift_nn(const Array2D<std::uint8_t>& in,
                                       double y_shift,
                                       double x_shift) {
    const std::size_t n0 = in.rows();
    const std::size_t n1 = in.cols();
    Array2D<std::uint8_t> out(n0, n1, 0);
    for (std::size_t r = 0; r < n0; ++r) {
        for (std::size_t col = 0; col < n1; ++col) {
            const long ir = static_cast<long>(
                std::floor(static_cast<double>(r) - y_shift + 0.5));
            const long ic = static_cast<long>(
                std::floor(static_cast<double>(col) - x_shift + 0.5));
            if (ir >= 0 && ir < static_cast<long>(n0) &&
                ic >= 0 && ic < static_cast<long>(n1)) {
                out(r, col) = in(static_cast<std::size_t>(ir),
                                 static_cast<std::size_t>(ic));
            }
        }
    }
    return out;
}

}  // namespace

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

    // Build the mask in the unrotated frame, exactly like lina.utils:
    //   x = (linspace(-N/2, N/2-1, N) + 1/2) * pixelscale; r = hypot(x, y);
    //   mask = (r > irad) & (r < orad); if edge: mask &= (x > edge)
    // The edge cut is applied on the UNROTATED x; the rotate/shift below are
    // raster operations applied to the rasterized mask (see lina.utils).
    for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t c = 0; c < n; ++c) {
            const double x = (static_cast<double>(c) - half + 0.5) * pixelscale;
            const double y = (static_cast<double>(r) - half + 0.5) * pixelscale;
            const double rr = std::hypot(x, y);
            if (rr > irad && rr < orad && x > edge) {
                mask(r, c) = 1;
            }
        }
    }

    if (rotation_deg != 0.0) {
        mask = ndimage_rotate_nn(mask, rotation_deg);
    }
    if (x_shift != 0.0 || y_shift != 0.0) {
        mask = ndimage_shift_nn(mask, y_shift, x_shift);
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

    // Unrotated frame (matches lina.utils.create_annular_focal_plane_mask):
    //   odd : x = linspace(-npsf/2, npsf/2-1, npsf) * pixelscale
    //   even: x = (linspace(...) + 1/2) * pixelscale
    //   mask = (r > irad) & (r < orad); if edge: mask &= (x > edge)
    for (std::size_t r = 0; r < npsf; ++r) {
        for (std::size_t c = 0; c < npsf; ++c) {
            const double x = (static_cast<double>(c) - half + offset) * psf_pixelscale;
            const double y = (static_cast<double>(r) - half + offset) * psf_pixelscale;
            const double rr = std::hypot(x, y);
            if (rr > irad && rr < orad && x > edge) {
                mask(r, c) = 1;
            }
        }
    }

    // lina applies ndimage.rotate(order=0) then ndimage.shift(order=0) to the
    // rasterized mask -- replicate those raster ops here for exact parity.
    if (rotation_deg != 0.0) {
        mask = ndimage_rotate_nn(mask, rotation_deg);
    }
    if (x_shift != 0.0 || y_shift != 0.0) {
        mask = ndimage_shift_nn(mask, y_shift, x_shift);
    }
    return mask;
}

} // namespace lina
