#pragma once

#include "lina/array.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace lina {

// Centering convention for 1D coordinate vectors. This matches the
// language used throughout the Python package and the MagAO-X /
// scoob optical models:
//
//   "odd"  -> samples are at integer multiples of `pixelscale`,
//             with one sample exactly at zero (DC):
//               x_i = (i - n/2) * pixelscale     (using integer n/2)
//             For n=4 this gives [-2, -1, 0, 1] * dx.
//
//   "even" -> samples are at half-integer multiples of `pixelscale`,
//             with no sample at zero:
//               x_i = (i - n/2 + 1/2) * pixelscale
//             For n=4 this gives [-1.5, -0.5, 0.5, 1.5] * dx.
//
// Most physical-optics quantities in lina (annular masks, vortex,
// PSF mft, fresnel TF) want "even" centering at the focal plane and
// "odd" at the pupil, so callers pass the appropriate string.
enum class Centering { Odd, Even };

inline Centering parse_centering(const std::string& s) {
    if (s == "even") return Centering::Even;
    return Centering::Odd; // default
}

inline Centering parse_centering(const char* s) {
    return s ? parse_centering(std::string(s)) : Centering::Odd;
}

// One-dimensional coordinate vector. Useful for separable kernels
// (mft, fresnel TF) so we don't materialize a full 2D array.
inline std::vector<double> make_coords(std::size_t n,
                                       double pixelscale,
                                       Centering centering) {
    std::vector<double> out(n);
    const double offset = (centering == Centering::Even) ? 0.5 : 0.0;
    const double half   = static_cast<double>(n) / 2.0;
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = (static_cast<double>(i) - half + offset) * pixelscale;
    }
    return out;
}

inline std::vector<double> make_coords(std::size_t n,
                                       double pixelscale,
                                       const std::string& centering) {
    return make_coords(n, pixelscale, parse_centering(centering));
}

// Grid2D: a single, lazy holder for a 2D coordinate grid (x, y, r,
// optionally rotated). Used by every routine that builds a mask or
// a phase pattern from coordinates so the centering / pixelscale /
// rotation logic only lives in ONE place.
class Grid2D {
public:
    Grid2D(std::size_t n,
           double pixelscale = 1.0,
           Centering centering = Centering::Even,
           double rotation_deg = 0.0,
           double x_shift = 0.0,
           double y_shift = 0.0)
        : n_(n),
          pixelscale_(pixelscale),
          centering_(centering),
          rotation_deg_(rotation_deg),
          x_shift_(x_shift),
          y_shift_(y_shift),
          coords_(make_coords(n, pixelscale, centering)),
          cos_r_(0.0),
          sin_r_(0.0) {
        if (rotation_deg_ != 0.0) {
            const double rad = rotation_deg_ * 3.141592653589793 / 180.0;
            cos_r_ = std::cos(rad);
            sin_r_ = std::sin(rad);
        } else {
            cos_r_ = 1.0;
            sin_r_ = 0.0;
        }
    }

    std::size_t n() const { return n_; }
    double pixelscale() const { return pixelscale_; }
    Centering centering() const { return centering_; }
    double rotation_deg() const { return rotation_deg_; }
    const std::vector<double>& coords() const { return coords_; }

    // 2D coordinate at (r, c) (row index, column index). y is row, x is
    // column. After applying optional shift and rotation:
    //
    //     x_raw = coords_[c] - x_shift
    //     y_raw = coords_[r] - y_shift
    //     x_rot =  x_raw*cos(theta) + y_raw*sin(theta)
    //     y_rot = -x_raw*sin(theta) + y_raw*cos(theta)
    //
    // Returns (x_rot, y_rot).
    std::pair<double, double> xy(std::size_t r, std::size_t c) const {
        const double x_raw = coords_[c] - x_shift_;
        const double y_raw = coords_[r] - y_shift_;
        if (rotation_deg_ == 0.0) return {x_raw, y_raw};
        return { x_raw * cos_r_ + y_raw * sin_r_,
                -x_raw * sin_r_ + y_raw * cos_r_};
    }

    // Radial distance to the (rotated, shifted) origin.
    double r(std::size_t row, std::size_t col) const {
        const auto [x, y] = xy(row, col);
        return std::sqrt(x * x + y * y);
    }

    // Polar angle (atan2(y, x)).
    double theta(std::size_t row, std::size_t col) const {
        const auto [x, y] = xy(row, col);
        return std::atan2(y, x);
    }

    // Materialize a 2D x grid (column coordinate broadcast).
    Array2D<double> x_grid() const {
        Array2D<double> out(n_, n_, 0.0);
        for (std::size_t r = 0; r < n_; ++r)
            for (std::size_t c = 0; c < n_; ++c)
                out(r, c) = xy(r, c).first;
        return out;
    }

    // Materialize a 2D y grid (row coordinate broadcast).
    Array2D<double> y_grid() const {
        Array2D<double> out(n_, n_, 0.0);
        for (std::size_t r = 0; r < n_; ++r)
            for (std::size_t c = 0; c < n_; ++c)
                out(r, c) = xy(r, c).second;
        return out;
    }

    // Materialize a radial-distance grid.
    Array2D<double> r_grid() const {
        Array2D<double> out(n_, n_, 0.0);
        for (std::size_t row = 0; row < n_; ++row)
            for (std::size_t col = 0; col < n_; ++col)
                out(row, col) = r(row, col);
        return out;
    }

private:
    std::size_t n_;
    double pixelscale_;
    Centering centering_;
    double rotation_deg_;
    double x_shift_;
    double y_shift_;
    std::vector<double> coords_;  // 1D coordinate axis (same for x and y)
    double cos_r_;
    double sin_r_;
};

} // namespace lina
