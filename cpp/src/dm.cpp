#include "lina/dm.h"
#include "lina/utils.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace lina {
namespace {

Array2D<double> shift_bilinear(const Array2D<double>& in, double shift_x, double shift_y) {
    const std::size_t rows = in.rows();
    const std::size_t cols = in.cols();
    Array2D<double> out(rows, cols, 0.0);

    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            const double src_y = static_cast<double>(r) - shift_y;
            const double src_x = static_cast<double>(c) - shift_x;
            if (src_x < 0.0 || src_y < 0.0 || src_x >= cols - 1 || src_y >= rows - 1) {
                continue;
            }
            const std::size_t x0 = static_cast<std::size_t>(std::floor(src_x));
            const std::size_t y0 = static_cast<std::size_t>(std::floor(src_y));
            const double dx = src_x - static_cast<double>(x0);
            const double dy = src_y - static_cast<double>(y0);

            const double v00 = in(y0, x0);
            const double v10 = in(y0, x0 + 1);
            const double v01 = in(y0 + 1, x0);
            const double v11 = in(y0 + 1, x0 + 1);

            out(r, c) = (1 - dx) * (1 - dy) * v00 +
                        dx * (1 - dy) * v10 +
                        (1 - dx) * dy * v01 +
                        dx * dy * v11;
        }
    }
    return out;
}

Array2D<double> hadamard_matrix(std::size_t n) {
    if (n == 0 || (n & (n - 1)) != 0) {
        throw std::invalid_argument("hadamard size must be power of two");
    }
    Array2D<double> h(1, 1, 1.0);
    while (h.rows() < n) {
        const std::size_t size = h.rows();
        Array2D<double> next(size * 2, size * 2, 0.0);
        for (std::size_t r = 0; r < size; ++r) {
            for (std::size_t c = 0; c < size; ++c) {
                const double val = h(r, c);
                next(r, c) = val;
                next(r, c + size) = val;
                next(r + size, c) = val;
                next(r + size, c + size) = -val;
            }
        }
        h = next;
    }
    return h;
}

} // namespace

Array2D<std::uint8_t> create_mask(std::size_t nact) {
    Array2D<std::uint8_t> mask(nact, nact, 0);
    const double half = static_cast<double>(nact) / 2.0;
    for (std::size_t r = 0; r < nact; ++r) {
        for (std::size_t c = 0; c < nact; ++c) {
            const double x = static_cast<double>(c) - half + 0.5;
            const double y = static_cast<double>(r) - half + 0.5;
            const double rmag = std::hypot(x, y);
            if (rmag < (static_cast<double>(nact) / 2.0 + 0.5)) {
                mask(r, c) = 1;
            }
        }
    }
    return mask;
}

Array2D<double> make_gaussian_inf_fun(double act_spacing,
                                      double sampling,
                                      double coupling,
                                      std::size_t nact) {
    const std::size_t ng = static_cast<std::size_t>(sampling * nact);
    const double pxscl = act_spacing / sampling;
    Array2D<double> inf(ng, ng, 0.0);

    const double half = static_cast<double>(ng) / 2.0;
    const double d = act_spacing / std::sqrt(-std::log(coupling));
    for (std::size_t r = 0; r < ng; ++r) {
        for (std::size_t c = 0; c < ng; ++c) {
            const double x = (static_cast<double>(c) - half + 0.5) * pxscl;
            const double y = (static_cast<double>(r) - half + 0.5) * pxscl;
            const double rr = std::hypot(x, y);
            inf(r, c) = std::exp(-std::pow(rr / d, 2.0));
        }
    }
    return inf;
}

Array2D<double> create_hadamard_modes(const Array2D<std::uint8_t>& dm_mask) {
    const std::size_t nact = dm_mask.rows();
    std::size_t nacts = 0;
    for (std::size_t i = 0; i < dm_mask.size(); ++i) {
        if (dm_mask.data()[i]) {
            ++nacts;
        }
    }
    std::size_t np2 = 1;
    while (np2 < nacts) {
        np2 <<= 1;
    }

    const Array2D<double> h = hadamard_matrix(np2);
    Array2D<double> modes(np2, nact * nact, 0.0);

    std::vector<std::size_t> inds;
    inds.reserve(nacts);
    for (std::size_t i = 0; i < dm_mask.size(); ++i) {
        if (dm_mask.data()[i]) {
            inds.push_back(i);
        }
    }

    for (std::size_t row = 0; row < np2; ++row) {
        for (std::size_t i = 0; i < nacts; ++i) {
            modes(row, inds[i]) = h(row, i);
        }
    }
    return modes;
}

Array2D<double> create_fourier_modes(const Array2D<std::uint8_t>& dm_mask,
                                     std::size_t npsf,
                                     double psf_pixelscale_lamD,
                                     double iwa,
                                     double owa,
                                     double rotation,
                                     double fourier_sampling,
                                     const char* which) {
    const std::size_t nact = dm_mask.rows();
    std::size_t nfg = static_cast<std::size_t>(std::round(npsf * psf_pixelscale_lamD / fourier_sampling));
    if (nfg % 2 == 1) {
        ++nfg;
    }

    // Match Python lina/dm.py: edge=0 (filter at xr>0), NOT edge=iwa-fs
    // (the older commented-out version). See cpp/AUDIT.md.
    const auto fourier_mask = create_annular_mask(
        nfg, fourier_sampling, iwa - fourier_sampling, owa + fourier_sampling,
        0.0, 0.0, 0.0, rotation);

    std::vector<std::pair<double, double>> sampled_fs;
    sampled_fs.reserve(nfg * nfg);
    const double half = static_cast<double>(nfg) / 2.0;
    for (std::size_t r = 0; r < nfg; ++r) {
        for (std::size_t c = 0; c < nfg; ++c) {
            if (fourier_mask(r, c)) {
                const double fx = (static_cast<double>(c) - half + 0.5) * fourier_sampling;
                const double fy = (static_cast<double>(r) - half + 0.5) * fourier_sampling;
                sampled_fs.emplace_back(fx, fy);
            }
        }
    }

    const bool use_cos = std::string(which) == "both" || std::string(which) == "cos";
    const bool use_sin = std::string(which) == "both" || std::string(which) == "sin";

    const std::size_t modes_per = (use_cos ? 1 : 0) + (use_sin ? 1 : 0);
    Array2D<double> modes(sampled_fs.size() * modes_per, nact * nact, 0.0);

    const double half_act = static_cast<double>(nact) / 2.0;
    std::size_t mode_idx = 0;
    for (const auto& fs : sampled_fs) {
        const double fx = fs.first;
        const double fy = fs.second;
        if (use_cos) {
            for (std::size_t r = 0; r < nact; ++r) {
                for (std::size_t c = 0; c < nact; ++c) {
                    const double x = static_cast<double>(c) - half_act + 0.5;
                    const double y = static_cast<double>(r) - half_act + 0.5;
                    const double val = std::cos(2.0 * M_PI * (fx * x + fy * y) / static_cast<double>(nact));
                    modes(mode_idx, r * nact + c) = dm_mask(r, c) ? val : 0.0;
                }
            }
            ++mode_idx;
        }
        if (use_sin) {
            for (std::size_t r = 0; r < nact; ++r) {
                for (std::size_t c = 0; c < nact; ++c) {
                    const double x = static_cast<double>(c) - half_act + 0.5;
                    const double y = static_cast<double>(r) - half_act + 0.5;
                    const double val = std::sin(2.0 * M_PI * (fx * x + fy * y) / static_cast<double>(nact));
                    modes(mode_idx, r * nact + c) = dm_mask(r, c) ? val : 0.0;
                }
            }
            ++mode_idx;
        }
    }
    return modes;
}

Array2D<double> create_fourier_probes(const Array2D<std::uint8_t>& dm_mask,
                                      std::size_t npsf,
                                      double psf_pixelscale_lamD,
                                      double iwa,
                                      double owa,
                                      double rotation,
                                      double fourier_sampling,
                                      std::size_t nprobes) {
    const Array2D<double> cos_modes = create_fourier_modes(
        dm_mask, npsf, psf_pixelscale_lamD, iwa, owa, rotation, fourier_sampling, "cos");
    const Array2D<double> sin_modes = create_fourier_modes(
        dm_mask, npsf, psf_pixelscale_lamD, iwa, owa, rotation, fourier_sampling, "sin");

    const std::size_t nact = dm_mask.rows();
    const std::size_t nmodes = cos_modes.rows();

    Array2D<double> sum_cos(nact, nact, 0.0);
    Array2D<double> sum_sin(nact, nact, 0.0);
    for (std::size_t i = 0; i < nmodes; ++i) {
        for (std::size_t r = 0; r < nact; ++r) {
            for (std::size_t c = 0; c < nact; ++c) {
                sum_cos(r, c) += cos_modes(i, r * nact + c);
                sum_sin(r, c) += sin_modes(i, r * nact + c);
            }
        }
    }

    Array2D<double> probes(nprobes, nact * nact, 0.0);
    for (std::size_t p = 0; p < nprobes; ++p) {
        const double cos_w = 1.0 - static_cast<double>(p) / std::max<std::size_t>(1, nprobes - 1);
        const double sin_w = 1.0 - cos_w;
        Array2D<double> probe(nact, nact, 0.0);
        double max_val = 0.0;
        for (std::size_t r = 0; r < nact; ++r) {
            for (std::size_t c = 0; c < nact; ++c) {
                const double val = cos_w * sum_cos(r, c) + sin_w * sum_sin(r, c);
                probe(r, c) = val;
                max_val = std::max(max_val, std::abs(val));
            }
        }
        if (max_val == 0.0) {
            max_val = 1.0;
        }
        probe = shift_bilinear(probe, 0.0, 0.0);
        for (std::size_t r = 0; r < nact; ++r) {
            for (std::size_t c = 0; c < nact; ++c) {
                probes(p, r * nact + c) = probe(r, c) / max_val;
            }
        }
    }
    return probes;
}

Array2D<double> make_fourier_command(std::size_t x_cpa,
                                     std::size_t y_cpa,
                                     std::size_t nact,
                                     double phase) {
    if (x_cpa > nact / 2 || y_cpa > nact / 2) {
        throw std::invalid_argument("cycles per aperture too high");
    }
    Array2D<double> cmd(nact, nact, 0.0);
    const double half = static_cast<double>(nact) / 2.0;
    for (std::size_t r = 0; r < nact; ++r) {
        for (std::size_t c = 0; c < nact; ++c) {
            const double x = static_cast<double>(c) - half;
            const double y = static_cast<double>(r) - half;
            cmd(r, c) = std::cos(2.0 * M_PI * (x_cpa * x + y_cpa * y) / static_cast<double>(nact) + phase);
        }
    }
    return cmd;
}

Array2D<double> make_f(std::size_t h,
                       std::size_t w,
                       int shift_x,
                       int shift_y,
                       std::size_t nact) {
    Array2D<double> cmd(nact, nact, 0.0);
    const int center = static_cast<int>(nact / 2);
    const int top_row = center + static_cast<int>(h / 2) + shift_y;
    const int mid_row = center + shift_y;
    const int row0 = center - static_cast<int>(h / 2) + shift_y;

    const int col0 = center - static_cast<int>(w / 2) + shift_x + 1;
    const int right_col = center + static_cast<int>(w / 2) + shift_x + 1;

    for (int r = row0; r < top_row; ++r) {
        if (r >= 0 && r < static_cast<int>(nact) && col0 >= 0 && col0 < static_cast<int>(nact)) {
            cmd(r, col0) = 1.0;
        }
    }
    if (top_row >= 0 && top_row < static_cast<int>(nact)) {
        for (int c = col0; c < right_col; ++c) {
            if (c >= 0 && c < static_cast<int>(nact)) {
                cmd(top_row, c) = 1.0;
            }
        }
    }
    if (mid_row >= 0 && mid_row < static_cast<int>(nact)) {
        for (int c = col0; c < right_col; ++c) {
            if (c >= 0 && c < static_cast<int>(nact)) {
                cmd(mid_row, c) = 1.0;
            }
        }
    }
    return cmd;
}

Array2D<double> make_ring(double rad, std::size_t nact, double thresh) {
    Array2D<double> ring(nact, nact, 0.0);
    const double half = static_cast<double>(nact) / 2.0;
    for (std::size_t r = 0; r < nact; ++r) {
        for (std::size_t c = 0; c < nact; ++c) {
            const double x = static_cast<double>(c) - half + 0.5;
            const double y = static_cast<double>(r) - half + 0.5;
            const double rr = std::hypot(x, y);
            if (rr > rad - thresh && rr < rad + thresh) {
                ring(r, c) = 1.0;
            }
        }
    }
    return ring;
}

Array2D<double> make_cross_command(const std::vector<double>& xc,
                                   const std::vector<double>& yc,
                                   std::size_t nact) {
    Array2D<double> cross(nact, nact, 0.0);
    const double half = static_cast<double>(nact) / 2.0;
    for (std::size_t i = 0; i < xc.size(); ++i) {
        for (std::size_t r = 0; r < nact; ++r) {
            for (std::size_t c = 0; c < nact; ++c) {
                const double x = static_cast<double>(c) - half + 0.5;
                const double y = static_cast<double>(r) - half + 0.5;
                if (x >= xc[i] - 0.5 && x < xc[i] + 0.5) {
                    cross(r, c) = 1.0;
                }
                if (y >= yc[i] - 0.5 && y < yc[i] + 0.5) {
                    cross(r, c) = 1.0;
                }
            }
        }
    }
    return cross;
}

} // namespace lina
