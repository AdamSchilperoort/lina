#pragma once

#include "lina/array.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace lina {

Array2D<std::uint8_t> create_mask(std::size_t nact = 34);

Array2D<double> make_gaussian_inf_fun(double act_spacing = 300e-6,
                                      double sampling = 10.0,
                                      double coupling = 0.15,
                                      std::size_t nact = 4);

Array2D<double> create_hadamard_modes(const Array2D<std::uint8_t>& dm_mask);

Array2D<double> create_fourier_modes(const Array2D<std::uint8_t>& dm_mask,
                                     std::size_t npsf,
                                     double psf_pixelscale_lamD,
                                     double iwa,
                                     double owa,
                                     double rotation = 0.0,
                                     double fourier_sampling = 0.75,
                                     const char* which = "both");

Array2D<double> create_fourier_probes(const Array2D<std::uint8_t>& dm_mask,
                                      std::size_t npsf,
                                      double psf_pixelscale_lamD,
                                      double iwa,
                                      double owa,
                                      double rotation = 0.0,
                                      double fourier_sampling = 0.75,
                                      std::size_t nprobes = 2);

Array2D<double> make_fourier_command(std::size_t x_cpa = 10,
                                     std::size_t y_cpa = 10,
                                     std::size_t nact = 34,
                                     double phase = 0.0);

Array2D<double> make_f(std::size_t h = 10,
                       std::size_t w = 6,
                       int shift_x = -1,
                       int shift_y = 0,
                       std::size_t nact = 34);

Array2D<double> make_ring(double rad = 15.0,
                          std::size_t nact = 34,
                          double thresh = 0.5);

Array2D<double> make_cross_command(const std::vector<double>& xc,
                                   const std::vector<double>& yc,
                                   std::size_t nact = 34);

} // namespace lina
