#pragma once

#include "lina/array.h"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace lina {

double mean(const Array2D<double>& array,
            const std::vector<std::uint8_t>* mask = nullptr);

double rms(const Array2D<double>& array,
           const std::vector<std::uint8_t>* mask = nullptr);

std::pair<Array2D<double>, Array2D<double>> make_grid(std::size_t npix,
                                                      double pixelscale = 1.0,
                                                      bool half_shift = false);

// Sentinel for "no edge filter" (matches Python's edge=None). Any caller
// passing an explicit numeric edge (including 0.0) gets a real cut at xr>edge.
constexpr double kNoEdgeFilter = -std::numeric_limits<double>::max();

Array2D<std::uint8_t> create_annular_mask(std::size_t n,
                                          double pixelscale,
                                          double irad,
                                          double orad,
                                          double edge = kNoEdgeFilter,
                                          double x_shift = 0.0,
                                          double y_shift = 0.0,
                                          double rotation_deg = 0.0);

Array2D<std::uint8_t> create_annular_focal_plane_mask(std::size_t npsf,
                                                      double psf_pixelscale,
                                                      double irad,
                                                      double orad,
                                                      double edge = kNoEdgeFilter,
                                                      const char* centering = "odd",
                                                      double rotation_deg = 0.0,
                                                      double x_shift = 0.0,
                                                      double y_shift = 0.0);

// FITS I/O. Compiled when LINA_USE_CFITSIO=ON (the default if cfitsio
// is found at configure time). When the library was built without
// cfitsio support, calling these throws std::runtime_error.
//
// A FITS header is represented as an ordered list of (key, value) string
// pairs. cfitsio decides on type from the value text (e.g. "1.0" -> real,
// "'foo'" -> string). For the common case of ASCII metadata, just pass
// strings; numeric values are accepted as their string form.
using FitsHeader = std::vector<std::pair<std::string, std::string>>;

// Save a 2D real-valued array (double precision) as the primary HDU of
// a FITS file. Always writes BITPIX=-64 (IEEE double), NAXIS=2.
void save_fits(const std::string& fpath,
               const Array2D<double>& data,
               const FitsHeader& header = {},
               bool overwrite = true);

void save_fits(const std::string& fpath,
               const Array2D<float>& data,
               const FitsHeader& header = {},
               bool overwrite = true);

void save_fits(const std::string& fpath,
               const Array2D<std::int32_t>& data,
               const FitsHeader& header = {},
               bool overwrite = true);

void save_fits(const std::string& fpath,
               const Array2D<std::uint8_t>& data,
               const FitsHeader& header = {},
               bool overwrite = true);

// Load the primary HDU of a FITS file as Array2D<double>. Higher-bit
// integer types are converted automatically by cfitsio. Throws if the
// file isn't a 2D image.
Array2D<double> load_fits_double(const std::string& fpath);
Array2D<float> load_fits_float(const std::string& fpath);

// Same as above but also returns the header (key/value strings).
std::pair<Array2D<double>, FitsHeader> load_fits_with_header(const std::string& fpath);

// Whether this build of liblina actually links against cfitsio. Use to
// branch in callers that want a fallback path. (always-true at runtime
// when LINA_USE_CFITSIO=ON, always-false otherwise.)
bool fits_available();

template <typename T>
Array2D<T> pad_or_crop(const Array2D<T>& input, std::size_t npix) {
    const std::size_t in_rows = input.rows();
    const std::size_t in_cols = input.cols();
    if (in_rows != in_cols) {
        throw std::invalid_argument("pad_or_crop expects square input");
    }
    if (in_rows == npix) {
        return input;
    }

    Array2D<T> output(npix, npix, T());
    if (npix < in_rows) {
        const std::size_t start = in_rows / 2 - npix / 2;
        for (std::size_t r = 0; r < npix; ++r) {
            for (std::size_t c = 0; c < npix; ++c) {
                output(r, c) = input(r + start, c + start);
            }
        }
    } else {
        const std::size_t start = npix / 2 - in_rows / 2;
        for (std::size_t r = 0; r < in_rows; ++r) {
            for (std::size_t c = 0; c < in_cols; ++c) {
                output(r + start, c + start) = input(r, c);
            }
        }
    }
    return output;
}

} // namespace lina
