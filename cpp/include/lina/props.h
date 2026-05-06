#pragma once

#include "lina/array.h"

#include <complex>

namespace lina {

Array2D<std::complex<double>> fft(const Array2D<std::complex<double>>& arr);
Array2D<std::complex<double>> ifft(const Array2D<std::complex<double>>& arr);
Array2D<std::complex<double>> fft_cpu(const Array2D<std::complex<double>>& arr);
Array2D<std::complex<double>> ifft_cpu(const Array2D<std::complex<double>>& arr);
Array2D<std::complex<double>> fft_gpu(const Array2D<std::complex<double>>& arr);
Array2D<std::complex<double>> ifft_gpu(const Array2D<std::complex<double>>& arr);

Array2D<std::complex<double>> ang_spec(const Array2D<std::complex<double>>& wavefront,
                                       double wavelength,
                                       double distance,
                                       double pixelscale);

Array2D<std::complex<double>> make_vortex_phase_mask(std::size_t npix,
                                                     int charge = 6,
                                                     const char* grid = "odd");

Array2D<std::complex<double>> mft_forward(
    const Array2D<std::complex<double>>& wavefront,
    std::size_t npix,
    std::size_t npsf,
    double psf_pixelscale_lamD,
    char convention = '-',
    const char* pp_centering = "odd",
    const char* fp_centering = "odd");

Array2D<std::complex<double>> mft_reverse(
    const Array2D<std::complex<double>>& fpwf,
    double psf_pixelscale_lamD,
    std::size_t npix,
    std::size_t N,
    char convention = '+',
    const char* pp_centering = "odd",
    const char* fp_centering = "odd");

// Fresnel transfer function for a defocus dz. Pure phase term in the
// pupil-plane spatial-frequency domain (multiplied with an unshifted
// pupil FFT). Matches lina.props.get_fresnel_TF.
//
//   df = 1 / (N * wavelength * fnum)
//   rp[i, j] = sqrt( ((i - (N-1)/2) * df)^2 + ((j - (N-1)/2) * df)^2 )
//   TF[i, j] = exp(-j * pi * dz * wavelength * rp[i, j]^2)
Array2D<std::complex<double>> get_fresnel_TF(double dz,
                                             std::size_t n,
                                             double wavelength,
                                             double fnum);

// ---------------------------------------------------------------------------
// GPU variants. These run on the CUDA device when the library was built
// with LINA_USE_CUDA=ON; otherwise they throw std::runtime_error so
// callers can fall back to the CPU equivalent. Numerical results match
// the CPU versions to within zgemm/cuFFT roundoff (~1e-12 for double).
// ---------------------------------------------------------------------------
Array2D<std::complex<double>> ang_spec_gpu(const Array2D<std::complex<double>>& wavefront,
                                           double wavelength,
                                           double distance,
                                           double pixelscale);

Array2D<std::complex<double>> make_vortex_phase_mask_gpu(std::size_t npix,
                                                         int charge = 6,
                                                         const char* grid = "odd");

Array2D<std::complex<double>> mft_forward_gpu(
    const Array2D<std::complex<double>>& wavefront,
    std::size_t npix,
    std::size_t npsf,
    double psf_pixelscale_lamD,
    char convention = '-',
    const char* pp_centering = "odd",
    const char* fp_centering = "odd");

Array2D<std::complex<double>> mft_reverse_gpu(
    const Array2D<std::complex<double>>& fpwf,
    double psf_pixelscale_lamD,
    std::size_t npix,
    std::size_t N,
    char convention = '+',
    const char* pp_centering = "odd",
    const char* fp_centering = "odd");

Array2D<std::complex<double>> get_fresnel_TF_gpu(double dz,
                                                 std::size_t n,
                                                 double wavelength,
                                                 double fnum);

} // namespace lina
