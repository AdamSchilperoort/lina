#include "lina/lina.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <complex>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

// ---------------------------------------------------------------------------
// Conversion helpers between numpy arrays and lina::Array2D<T>.
// ---------------------------------------------------------------------------

template <typename T>
lina::Array2D<T> array2d_from_numpy(
    const py::array_t<T, py::array::c_style | py::array::forcecast>& arr) {
    const auto info = arr.request();
    if (info.ndim != 2) {
        throw std::runtime_error("Expected 2D array");
    }
    const auto rows = static_cast<std::size_t>(info.shape[0]);
    const auto cols = static_cast<std::size_t>(info.shape[1]);
    lina::Array2D<T> out(rows, cols, T{});
    std::memcpy(out.data(), info.ptr, sizeof(T) * rows * cols);
    return out;
}

template <typename T>
lina::Array2D<T> modes_array2d_from_numpy(
    const py::array_t<T, py::array::c_style | py::array::forcecast>& arr) {
    const auto info = arr.request();
    if (info.ndim == 2) {
        return array2d_from_numpy<T>(arr);
    }
    if (info.ndim != 3) {
        throw std::runtime_error("Expected 2D (flattened) or 3D (cube) array");
    }
    const auto nmodes = static_cast<std::size_t>(info.shape[0]);
    const auto d1 = static_cast<std::size_t>(info.shape[1]);
    const auto d2 = static_cast<std::size_t>(info.shape[2]);
    lina::Array2D<T> out(nmodes, d1 * d2, T{});
    std::memcpy(out.data(), info.ptr, sizeof(T) * nmodes * d1 * d2);
    return out;
}

template <typename T>
py::array_t<T> numpy_from_array2d(const lina::Array2D<T>& arr) {
    py::array_t<T> out({arr.rows(), arr.cols()});
    std::memcpy(out.mutable_data(), arr.data(), sizeof(T) * arr.size());
    return out;
}

template <typename T>
std::vector<T> vector_from_numpy(
    const py::array_t<T, py::array::c_style | py::array::forcecast>& arr) {
    const auto info = arr.request();
    if (info.ndim != 1) {
        throw std::runtime_error("Expected 1D array");
    }
    std::vector<T> out(info.size);
    std::memcpy(out.data(), info.ptr, sizeof(T) * info.size);
    return out;
}

template <typename T>
py::array_t<T> numpy_from_vector(const std::vector<T>& v) {
    py::array_t<T> out({static_cast<py::ssize_t>(v.size())});
    if (!v.empty()) {
        std::memcpy(out.mutable_data(), v.data(), sizeof(T) * v.size());
    }
    return out;
}

// ---------------------------------------------------------------------------
// Wrappers
// ---------------------------------------------------------------------------

py::array_t<std::complex<double>> fft_cpu_wrapper(py::array_t<std::complex<double>> input) {
    auto arr = array2d_from_numpy<std::complex<double>>(input);
    return numpy_from_array2d(lina::fft_cpu(arr));
}

py::array_t<std::complex<double>> ifft_cpu_wrapper(py::array_t<std::complex<double>> input) {
    auto arr = array2d_from_numpy<std::complex<double>>(input);
    return numpy_from_array2d(lina::ifft_cpu(arr));
}

#ifdef LINA_USE_CUDA
py::array_t<std::complex<double>> fft_gpu_wrapper(py::array_t<std::complex<double>> input) {
    auto arr = array2d_from_numpy<std::complex<double>>(input);
    return numpy_from_array2d(lina::fft_gpu(arr));
}

py::array_t<std::complex<double>> ifft_gpu_wrapper(py::array_t<std::complex<double>> input) {
    auto arr = array2d_from_numpy<std::complex<double>>(input);
    return numpy_from_array2d(lina::ifft_gpu(arr));
}

py::array_t<std::complex<double>> ang_spec_gpu_wrapper(
    py::array_t<std::complex<double>> input,
    double wavelength, double distance, double pixelscale) {
    auto arr = array2d_from_numpy<std::complex<double>>(input);
    return numpy_from_array2d(lina::ang_spec_gpu(arr, wavelength, distance, pixelscale));
}

py::array_t<std::complex<double>> make_vortex_phase_mask_gpu_wrapper(
    std::size_t npix, int charge, const std::string& grid) {
    return numpy_from_array2d(lina::make_vortex_phase_mask_gpu(npix, charge, grid.c_str()));
}

py::array_t<std::complex<double>> get_fresnel_TF_gpu_wrapper(
    double dz, std::size_t n, double wavelength, double fnum) {
    return numpy_from_array2d(lina::get_fresnel_TF_gpu(dz, n, wavelength, fnum));
}

py::array_t<std::complex<double>> mft_forward_gpu_wrapper(
    py::array_t<std::complex<double>> input,
    double npix, std::size_t npsf, double psf_pixelscale_lamD,
    const std::string& convention,
    const std::string& pp_centering,
    const std::string& fp_centering) {
    auto arr = array2d_from_numpy<std::complex<double>>(input);
    return numpy_from_array2d(lina::mft_forward_gpu(
        arr, npix, npsf, psf_pixelscale_lamD,
        convention.empty() ? '-' : convention[0],
        pp_centering.c_str(), fp_centering.c_str()));
}

py::array_t<std::complex<double>> mft_reverse_gpu_wrapper(
    py::array_t<std::complex<double>> input,
    double psf_pixelscale_lamD, double npix, std::size_t N,
    const std::string& convention,
    const std::string& pp_centering,
    const std::string& fp_centering) {
    auto arr = array2d_from_numpy<std::complex<double>>(input);
    return numpy_from_array2d(lina::mft_reverse_gpu(
        arr, psf_pixelscale_lamD, npix, N,
        convention.empty() ? '+' : convention[0],
        pp_centering.c_str(), fp_centering.c_str()));
}
#endif

py::array_t<std::complex<double>> ang_spec_wrapper(py::array_t<std::complex<double>> input,
                                                   double wavelength,
                                                   double distance,
                                                   double pixelscale) {
    auto arr = array2d_from_numpy<std::complex<double>>(input);
    return numpy_from_array2d(lina::ang_spec(arr, wavelength, distance, pixelscale));
}

py::array_t<std::complex<double>> get_fresnel_TF_wrapper(double dz,
                                                        std::size_t n,
                                                        double wavelength,
                                                        double fnum) {
    return numpy_from_array2d(lina::get_fresnel_TF(dz, n, wavelength, fnum));
}

// ---------------------------------------------------------------------------
// FITS I/O. Mirrors the signature of lina.utils.save_fits / load_fits so the
// Python lina_cpp.utils wrappers can call this 1:1.
// ---------------------------------------------------------------------------

// Convert a Python dict {key: value} into the C++ FitsHeader vector. Values
// are stringified (str(v)) on the Python side -- here we just take strings.
lina::FitsHeader header_from_pydict(const py::object& obj) {
    lina::FitsHeader hdr;
    if (obj.is_none()) return hdr;
    py::dict d = py::cast<py::dict>(obj);
    for (auto item : d) {
        const std::string k = py::str(item.first);
        const std::string v = py::str(item.second);
        hdr.emplace_back(k, v);
    }
    return hdr;
}

void save_fits_wrapper(const std::string& fpath,
                       py::array data,
                       py::object header,
                       bool overwrite,
                       bool quiet) {
    lina::FitsHeader hdr = header_from_pydict(header);
    // Dispatch on the numpy dtype. We support the four most common dtypes
    // natively; everything else falls back to float64 via forcecast.
    const py::dtype dt = data.dtype();
    if (dt.is(py::dtype::of<double>())) {
        auto arr2d = array2d_from_numpy<double>(
            py::cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(data));
        lina::save_fits(fpath, arr2d, hdr, overwrite);
    } else if (dt.is(py::dtype::of<float>())) {
        auto arr2d = array2d_from_numpy<float>(
            py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(data));
        lina::save_fits(fpath, arr2d, hdr, overwrite);
    } else if (dt.is(py::dtype::of<std::int32_t>())) {
        auto arr2d = array2d_from_numpy<std::int32_t>(
            py::cast<py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>>(data));
        lina::save_fits(fpath, arr2d, hdr, overwrite);
    } else if (dt.is(py::dtype::of<std::uint8_t>())) {
        auto arr2d = array2d_from_numpy<std::uint8_t>(
            py::cast<py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>>(data));
        lina::save_fits(fpath, arr2d, hdr, overwrite);
    } else {
        // Cast unknown dtypes to float64.
        auto arr2d = array2d_from_numpy<double>(
            py::cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(data));
        lina::save_fits(fpath, arr2d, hdr, overwrite);
    }
    if (!quiet) {
        py::print("Saved data to:", fpath);
    }
}

py::object load_fits_wrapper(const std::string& fpath, bool with_header) {
    if (!with_header) {
        auto data = lina::load_fits_double(fpath);
        return numpy_from_array2d(data);
    }
    auto pair = lina::load_fits_with_header(fpath);
    py::dict hdr;
    for (const auto& kv : pair.second) {
        hdr[py::str(kv.first)] = py::str(kv.second);
    }
    return py::make_tuple(numpy_from_array2d(pair.first), hdr);
}

py::array_t<std::complex<double>> make_vortex_phase_mask_wrapper(std::size_t npix,
                                                                 int charge,
                                                                 const std::string& grid) {
    return numpy_from_array2d(lina::make_vortex_phase_mask(npix, charge, grid.c_str()));
}

py::array_t<std::complex<double>> mft_forward_wrapper(py::array_t<std::complex<double>> input,
                                                      double npix,
                                                      std::size_t npsf,
                                                      double psf_pixelscale_lamD,
                                                      const std::string& convention,
                                                      const std::string& pp_centering,
                                                      const std::string& fp_centering) {
    auto arr = array2d_from_numpy<std::complex<double>>(input);
    const char conv = convention.empty() ? '-' : convention[0];
    return numpy_from_array2d(lina::mft_forward(arr, npix, npsf, psf_pixelscale_lamD,
                                                conv, pp_centering.c_str(), fp_centering.c_str()));
}

py::array_t<std::complex<double>> mft_reverse_wrapper(py::array_t<std::complex<double>> input,
                                                      double psf_pixelscale_lamD,
                                                      double npix,
                                                      std::size_t N,
                                                      const std::string& convention,
                                                      const std::string& pp_centering,
                                                      const std::string& fp_centering) {
    auto arr = array2d_from_numpy<std::complex<double>>(input);
    const char conv = convention.empty() ? '+' : convention[0];
    return numpy_from_array2d(lina::mft_reverse(arr, psf_pixelscale_lamD, npix, N,
                                                conv, pp_centering.c_str(), fp_centering.c_str()));
}

double mean_wrapper(py::array_t<double> input) {
    auto arr = array2d_from_numpy<double>(input);
    return lina::mean(arr);
}

double mean_masked_wrapper(py::array_t<double> input,
                           py::array_t<std::uint8_t,
                                       py::array::c_style | py::array::forcecast> mask) {
    auto arr = array2d_from_numpy<double>(input);
    const auto info = mask.request();
    if (info.ndim != 1) {
        throw std::runtime_error("mask must be a flat 1D array");
    }
    std::vector<std::uint8_t> m(info.size);
    std::memcpy(m.data(), info.ptr, sizeof(std::uint8_t) * info.size);
    return lina::mean(arr, &m);
}

double rms_wrapper(py::array_t<double> input) {
    auto arr = array2d_from_numpy<double>(input);
    return lina::rms(arr);
}

py::tuple make_grid_wrapper(std::size_t npix, double pixelscale, bool half_shift) {
    auto pair = lina::make_grid(npix, pixelscale, half_shift);
    return py::make_tuple(numpy_from_array2d(pair.first),
                          numpy_from_array2d(pair.second));
}

py::array_t<double> pad_or_crop_wrapper(py::array_t<double> input, std::size_t npix) {
    auto arr = array2d_from_numpy<double>(input);
    return numpy_from_array2d(lina::pad_or_crop(arr, npix));
}

py::array_t<std::uint8_t> create_annular_mask_wrapper(std::size_t n,
                                                     double pixelscale,
                                                     double irad,
                                                     double orad,
                                                     std::optional<double> edge,
                                                     double x_shift,
                                                     double y_shift,
                                                     double rotation) {
    const double edge_val = edge.value_or(lina::kNoEdgeFilter);
    return numpy_from_array2d(lina::create_annular_mask(
        n, pixelscale, irad, orad, edge_val, x_shift, y_shift, rotation));
}

py::array_t<std::uint8_t> create_annular_focal_plane_mask_wrapper(
    std::size_t npsf,
    double psf_pixelscale,
    double irad,
    double orad,
    std::optional<double> edge,
    const std::string& centering,
    double rotation,
    double x_shift,
    double y_shift) {
    const double edge_val = edge.value_or(lina::kNoEdgeFilter);
    return numpy_from_array2d(lina::create_annular_focal_plane_mask(
        npsf, psf_pixelscale, irad, orad, edge_val, centering.c_str(),
        rotation, x_shift, y_shift));
}

py::array_t<std::uint8_t> create_mask_wrapper(std::size_t nact) {
    return numpy_from_array2d(lina::create_mask(nact));
}

py::array_t<double> make_gaussian_inf_fun_wrapper(double act_spacing,
                                                  double sampling,
                                                  double coupling,
                                                  std::size_t nact) {
    return numpy_from_array2d(lina::make_gaussian_inf_fun(
        act_spacing, sampling, coupling, nact));
}

py::array_t<double> create_hadamard_modes_wrapper(py::array_t<std::uint8_t> dm_mask) {
    auto m = array2d_from_numpy<std::uint8_t>(dm_mask);
    return numpy_from_array2d(lina::create_hadamard_modes(m));
}

py::array_t<double> create_fourier_modes_wrapper(py::array_t<std::uint8_t> dm_mask,
                                                 std::size_t npsf,
                                                 double psf_pixelscale_lamD,
                                                 double iwa,
                                                 double owa,
                                                 double rotation,
                                                 double fourier_sampling,
                                                 const std::string& which) {
    auto m = array2d_from_numpy<std::uint8_t>(dm_mask);
    return numpy_from_array2d(lina::create_fourier_modes(
        m, npsf, psf_pixelscale_lamD, iwa, owa, rotation,
        fourier_sampling, which.c_str()));
}

py::array_t<double> make_fourier_command_wrapper(std::size_t x_cpa,
                                                 std::size_t y_cpa,
                                                 std::size_t nact,
                                                 double phase) {
    return numpy_from_array2d(lina::make_fourier_command(x_cpa, y_cpa, nact, phase));
}

py::array_t<double> make_f_wrapper(std::size_t h, std::size_t w,
                                   int shift_x, int shift_y,
                                   std::size_t nact) {
    return numpy_from_array2d(lina::make_f(h, w, shift_x, shift_y, nact));
}

py::array_t<double> make_ring_wrapper(double rad, std::size_t nact, double thresh) {
    return numpy_from_array2d(lina::make_ring(rad, nact, thresh));
}

py::array_t<double> make_cross_command_wrapper(py::array_t<double> xc,
                                               py::array_t<double> yc,
                                               std::size_t nact) {
    auto x = vector_from_numpy<double>(xc);
    auto y = vector_from_numpy<double>(yc);
    return numpy_from_array2d(lina::make_cross_command(x, y, nact));
}

py::array_t<double> gemm_wrapper(py::array_t<double> a,
                                 py::array_t<double> b,
                                 bool transpose_a,
                                 bool transpose_b) {
    auto A = array2d_from_numpy<double>(a);
    auto B = array2d_from_numpy<double>(b);
    return numpy_from_array2d(lina::gemm(A, B, transpose_a, transpose_b));
}

py::array_t<double> gemv_wrapper(py::array_t<double> a,
                                 py::array_t<double> x,
                                 bool transpose_a) {
    auto A = array2d_from_numpy<double>(a);
    auto xv = vector_from_numpy<double>(x);
    return numpy_from_vector(lina::gemv(A, xv, transpose_a));
}

py::array_t<double> lstsq_wrapper(
    py::array_t<double, py::array::c_style | py::array::forcecast> modes,
    py::array_t<double, py::array::c_style | py::array::forcecast> data) {
    const auto m_info = modes.request();
    const auto d_info = data.request();
    if (m_info.ndim < 2) {
        throw std::runtime_error("modes must have at least 2 dimensions");
    }
    const std::size_t nmodes = static_cast<std::size_t>(m_info.shape[0]);
    std::size_t npoints = 1;
    for (int i = 1; i < m_info.ndim; ++i) {
        npoints *= static_cast<std::size_t>(m_info.shape[i]);
    }
    if (static_cast<std::size_t>(d_info.size) != npoints) {
        throw std::runtime_error("modes/data size mismatch in lstsq");
    }

    const double* m_ptr = static_cast<const double*>(m_info.ptr);
    const double* d_ptr = static_cast<const double*>(d_info.ptr);

    std::size_t nvalid = 0;
    for (std::size_t j = 0; j < npoints; ++j) {
        if (std::isfinite(d_ptr[j])) {
            ++nvalid;
        }
    }
    if (nvalid == 0) {
        return numpy_from_vector(std::vector<double>(nmodes, 0.0));
    }

    lina::Array2D<double> A(nvalid, nmodes, 0.0);
    std::vector<double> b(nvalid, 0.0);
    std::size_t row = 0;
    for (std::size_t j = 0; j < npoints; ++j) {
        if (!std::isfinite(d_ptr[j])) continue;
        b[row] = d_ptr[j];
        for (std::size_t k = 0; k < nmodes; ++k) {
            A(row, k) = m_ptr[k * npoints + j];
        }
        ++row;
    }

    const auto svd = lina::svd_thin(A);
    const std::size_t r = svd.s.size();
    const double smax = r ? *std::max_element(svd.s.begin(), svd.s.end()) : 0.0;
    const double cutoff = std::numeric_limits<double>::epsilon() *
                          static_cast<double>(std::max(A.rows(), A.cols())) *
                          smax;

    std::vector<double> y(r, 0.0);
    for (std::size_t i = 0; i < r; ++i) {
        const double si = svd.s[i];
        if (si <= cutoff) continue;
        double proj = 0.0;
        for (std::size_t j = 0; j < A.rows(); ++j) {
            proj += svd.u(j, i) * b[j];
        }
        y[i] = proj / si;
    }

    std::vector<double> x(nmodes, 0.0);
    for (std::size_t k = 0; k < nmodes; ++k) {
        double acc = 0.0;
        for (std::size_t i = 0; i < r; ++i) {
            acc += svd.vt(i, k) * y[i];
        }
        x[k] = acc;
    }
    return numpy_from_vector(x);
}

py::object tikhonov_inverse_wrapper(
    py::array_t<double, py::array::c_style | py::array::forcecast> a,
    double rcond,
    bool return_all) {
    auto A = array2d_from_numpy<double>(a);
    const auto svd = lina::svd_thin(A);
    const std::size_t m = A.rows();
    const std::size_t n = A.cols();
    const std::size_t r = svd.s.size();
    const double smax = r ? *std::max_element(svd.s.begin(), svd.s.end()) : 0.0;
    const double alpha = rcond * smax;
    const double alpha2 = alpha * alpha;

    // P = V diag(s/(s^2+alpha^2)) U^T via BLAS instead of an O(n^2 m) loop.
    lina::Array2D<double> M(r, m, 0.0);
    for (std::size_t i = 0; i < r; ++i) {
        const double si = svd.s[i];
        const double denom = si * si + alpha2;
        const double coeff = (denom > 0.0) ? (si / denom) : 0.0;
        if (coeff == 0.0) continue;
        for (std::size_t col = 0; col < m; ++col) {
            M(i, col) = coeff * svd.u(col, i);
        }
    }
    const auto P = lina::gemm(svd.vt, M, /*transpose_a=*/true, /*transpose_b=*/false);

    if (return_all) {
        return py::make_tuple(
            numpy_from_array2d(P),
            numpy_from_array2d(svd.u),
            numpy_from_vector(svd.s),
            numpy_from_array2d(svd.vt));
    }
    return numpy_from_array2d(P);
}

py::array_t<double> beta_reg_wrapper(
    py::array_t<double, py::array::c_style | py::array::forcecast> s,
    double beta) {
    const auto S = array2d_from_numpy<double>(s);
    auto sts = lina::gemm(S, S, true, false);  // S^T S
    double alpha2 = 0.0;
    for (std::size_t i = 0; i < sts.rows() && i < sts.cols(); ++i) {
        alpha2 = std::max(alpha2, sts(i, i));
    }
    const double reg = alpha2 * std::pow(10.0, beta);
    for (std::size_t i = 0; i < sts.rows() && i < sts.cols(); ++i) {
        sts(i, i) += reg;
    }

    const auto svd = lina::svd_thin(sts);
    const std::size_t n = sts.rows();
    const std::size_t r = svd.s.size();
    const double smax = r ? *std::max_element(svd.s.begin(), svd.s.end()) : 0.0;
    const double cutoff = std::numeric_limits<double>::epsilon() *
                          static_cast<double>(std::max(sts.rows(), sts.cols())) *
                          smax;

    // Pseudo-inverse inv = V diag(1/s) U^T, computed with BLAS rather than an
    // O(n^3) triple loop. Build M(i,col) = (1/s_i) * U(col,i) (r x n), then
    // inv = V @ M = (vt)^T @ M.
    lina::Array2D<double> M(r, n, 0.0);
    for (std::size_t i = 0; i < r; ++i) {
        const double si = svd.s[i];
        const double coeff = (si > cutoff) ? (1.0 / si) : 0.0;
        if (coeff == 0.0) continue;
        for (std::size_t col = 0; col < n; ++col) {
            M(i, col) = coeff * svd.u(col, i);
        }
    }
    const auto inv = lina::gemm(svd.vt, M, /*transpose_a=*/true, /*transpose_b=*/false);

    // control_matrix = inv(sts + reg*I) @ S^T
    const auto control = lina::gemm(inv, S, false, true);
    return numpy_from_array2d(control);
}

py::array_t<double> beta_reg_gpu_wrapper(
    py::array_t<double, py::array::c_style | py::array::forcecast> s,
    double beta) {
    const auto S = array2d_from_numpy<double>(s);
    return numpy_from_array2d(lina::beta_reg_gpu(S, beta));
}

py::tuple svd_wrapper(py::array_t<double> a) {
    auto A = array2d_from_numpy<double>(a);
    auto res = lina::svd(A);
    return py::make_tuple(numpy_from_array2d(res.u),
                          numpy_from_vector(res.s),
                          numpy_from_array2d(res.vt));
}

py::tuple svd_float_cpu_wrapper(py::array_t<float> input) {
    auto arr = array2d_from_numpy<float>(input);
    auto res = lina::svd_float_cpu(arr);
    return py::make_tuple(numpy_from_array2d(res.u),
                          numpy_from_vector(res.s),
                          numpy_from_array2d(res.vt));
}

#ifdef LINA_USE_CUDA
py::tuple svd_float_gpu_wrapper(py::array_t<float> input) {
    auto arr = array2d_from_numpy<float>(input);
    auto res = lina::svd_float_gpu(arr);
    return py::make_tuple(numpy_from_array2d(res.u),
                          numpy_from_vector(res.s),
                          numpy_from_array2d(res.vt));
}
#endif

py::array_t<double> normalize_coro_im_wrapper(py::array_t<double> raw,
                                              double exp_im, double gain_im, double atten_im,
                                              double exp_ref, double gain_ref, double atten_ref,
                                              double Imax,
                                              py::object dark_im) {
    auto R = array2d_from_numpy<double>(raw);
    lina::ImParams im{exp_im, gain_im, atten_im, Imax};
    lina::ImParams ref{exp_ref, gain_ref, atten_ref, Imax};
    if (dark_im.is_none()) {
        return numpy_from_array2d(lina::normalize_coro_im(R, im, ref, 0.0));
    }
    auto D = array2d_from_numpy<double>(
        dark_im.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>());
    return numpy_from_array2d(lina::normalize_coro_im(R, im, ref, D));
}

py::tuple compute_contrast_wrapper(py::array_t<double> ni,
                                   py::array_t<std::uint8_t> mask) {
    auto N = array2d_from_numpy<double>(ni);
    auto M = array2d_from_numpy<std::uint8_t>(mask);
    auto res = lina::compute_contrast(N, M);
    return py::make_tuple(res.contrast, res.n_mask, res.n_positive);
}

py::array_t<double> compute_hadamard_scale_factors_wrapper(py::array_t<double> had_modes,
                                                           double scale_exp,
                                                           double scale_thresh,
                                                           double iwa,
                                                           double owa,
                                                           std::size_t oversamp) {
    auto H = array2d_from_numpy<double>(had_modes);
    auto v = lina::compute_hadamard_scale_factors(
        H, scale_exp, scale_thresh, iwa, owa, oversamp);
    return numpy_from_vector(v);
}

// ---------------------------------------------------------------------------
// wfe.py wrappers
// ---------------------------------------------------------------------------

py::tuple noll_index_to_mn_wrapper(int j) {
    auto [m, n] = lina::wfe::noll_index_to_mn(j);
    return py::make_tuple(m, n);
}

py::tuple fringe_index_to_mn_wrapper(int j) {
    auto [m, n] = lina::wfe::fringe_index_to_mn(j);
    return py::make_tuple(m, n);
}

py::tuple wfe_generate_freqs_wrapper(double delt, double tmax) {
    auto fg = lina::wfe::generate_freqs(delt, tmax);
    return py::make_tuple(numpy_from_vector(fg.freqs), fg.delf,
                          numpy_from_vector(fg.times));
}

py::array_t<double> wfe_roll_psd_wrapper(py::array_t<double, py::array::c_style | py::array::forcecast> freqs,
                                         double beta, double f_roll, double alpha,
                                         bool normalized) {
    auto v = vector_from_numpy(freqs);
    auto out = lina::wfe::roll_psd(v, beta, f_roll, alpha, normalized);
    return numpy_from_vector(out);
}

py::tuple wfe_generate_time_series_wrapper(
    py::array_t<double, py::array::c_style | py::array::forcecast> psd,
    py::array_t<double, py::array::c_style | py::array::forcecast> freqs,
    double rms_target,
    std::uint64_t seed) {
    auto p = vector_from_numpy(psd);
    auto f = vector_from_numpy(freqs);
    auto ts = lina::wfe::generate_time_series(p, f, rms_target, seed);
    return py::make_tuple(numpy_from_vector(ts.values), numpy_from_vector(ts.times));
}

py::tuple wfe_compute_cumulative_psd_wrapper(
    py::array_t<double, py::array::c_style | py::array::forcecast> freqs,
    py::array_t<double, py::array::c_style | py::array::forcecast> psd) {
    auto f = vector_from_numpy(freqs);
    auto p = vector_from_numpy(psd);
    auto [cum, tail] = lina::wfe::compute_cumulative_psd(f, p);
    return py::make_tuple(numpy_from_vector(cum), numpy_from_vector(tail));
}

// ---------------------------------------------------------------------------
// llowfsc.py wrappers
// ---------------------------------------------------------------------------

py::tuple llowfsc_acquire_ref_wrapper(
    py::array_t<double, py::array::c_style | py::array::forcecast> camlo_ref_im,
    py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> wfs_mask,
    py::object dark_im,
    bool flux_norm) {
    auto cam = array2d_from_numpy<double>(camlo_ref_im);
    auto mask = array2d_from_numpy<std::uint8_t>(wfs_mask);
    lina::llowfsc::AcquireRefResult res;
    if (py::isinstance<py::array>(dark_im)) {
        auto d = py::cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(dark_im);
        auto darr = array2d_from_numpy<double>(d);
        res = lina::llowfsc::acquire_ref(cam, mask, darr, flux_norm);
    } else {
        const double scalar = py::cast<double>(dark_im);
        res = lina::llowfsc::acquire_ref(cam, mask, scalar, flux_norm);
    }
    return py::make_tuple(numpy_from_array2d(res.ref_image), res.flux_norm_coeff);
}

py::object llowfsc_reconstruct_wrapper(
    py::array_t<double, py::array::c_style | py::array::forcecast> camlo_im,
    py::array_t<double, py::array::c_style | py::array::forcecast> ref_im,
    py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> wfs_mask,
    py::array_t<double, py::array::c_style | py::array::forcecast> control_matrix,
    std::size_t mode_lo, std::size_t mode_hi,
    py::object dark_im, bool flux_norm, bool return_del_im) {

    auto cam = array2d_from_numpy<double>(camlo_im);
    auto ref = array2d_from_numpy<double>(ref_im);
    auto mask = array2d_from_numpy<std::uint8_t>(wfs_mask);
    auto C = array2d_from_numpy<double>(control_matrix);

    lina::Array2D<double> del_im;
    lina::Array2D<double>* del_im_ptr = return_del_im ? &del_im : nullptr;

    std::vector<double> coeff;
    if (py::isinstance<py::array>(dark_im)) {
        auto d = py::cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(dark_im);
        auto darr = array2d_from_numpy<double>(d);
        coeff = lina::llowfsc::reconstruct(cam, ref, mask, C, mode_lo, mode_hi,
                                            &darr, 0.0, flux_norm, del_im_ptr);
    } else {
        const double scalar = py::cast<double>(dark_im);
        coeff = lina::llowfsc::reconstruct(cam, ref, mask, C, mode_lo, mode_hi,
                                            nullptr, scalar, flux_norm, del_im_ptr);
    }

    if (return_del_im) {
        return py::make_tuple(numpy_from_vector(coeff), numpy_from_array2d(del_im));
    }
    return numpy_from_vector(coeff);
}

py::array_t<double> llowfsc_compute_zpo_wrapper(
    py::list dm_commands_masked,
    py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> wfs_mask,
    py::array_t<double, py::array::c_style | py::array::forcecast> response_matrix,
    py::array_t<double, py::array::c_style | py::array::forcecast> dm_modal_matrix) {
    std::vector<std::vector<double>> cmds;
    cmds.reserve(dm_commands_masked.size());
    for (auto h : dm_commands_masked) {
        auto a = py::cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(h);
        cmds.push_back(vector_from_numpy(a));
    }
    auto mask = array2d_from_numpy<std::uint8_t>(wfs_mask);
    auto R = array2d_from_numpy<double>(response_matrix);
    auto M = array2d_from_numpy<double>(dm_modal_matrix);
    return numpy_from_array2d(lina::llowfsc::compute_zpo(cmds, mask, R, M));
}

py::array_t<double> llowfsc_loop_step_wrapper(
    py::array_t<double, py::array::c_style | py::array::forcecast> camlo_im,
    py::array_t<double, py::array::c_style | py::array::forcecast> ref_plus_zpo,
    py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> wfs_mask,
    py::array_t<double, py::array::c_style | py::array::forcecast> control_matrix,
    py::array_t<double, py::array::c_style | py::array::forcecast> dm_modes_flat,
    std::size_t dm_rows, std::size_t dm_cols,
    py::array_t<double, py::array::c_style | py::array::forcecast> gains,
    std::size_t mode_lo, std::size_t mode_hi,
    py::array_t<double, py::array::c_style | py::array::forcecast> ffo,
    py::object dark_im, bool flux_norm) {

    auto cam = array2d_from_numpy<double>(camlo_im);
    auto ref = array2d_from_numpy<double>(ref_plus_zpo);
    auto mask = array2d_from_numpy<std::uint8_t>(wfs_mask);
    auto C = array2d_from_numpy<double>(control_matrix);
    auto Mflat = array2d_from_numpy<double>(dm_modes_flat);
    auto g = vector_from_numpy(gains);
    auto f = vector_from_numpy(ffo);

    lina::Array2D<double> result;
    if (py::isinstance<py::array>(dark_im)) {
        auto d = py::cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(dark_im);
        auto darr = array2d_from_numpy<double>(d);
        result = lina::llowfsc::loop_step(cam, ref, mask, C, Mflat,
                                           dm_rows, dm_cols, g, mode_lo, mode_hi,
                                           f, &darr, 0.0, flux_norm);
    } else {
        const double scalar = py::cast<double>(dark_im);
        result = lina::llowfsc::loop_step(cam, ref, mask, C, Mflat,
                                           dm_rows, dm_cols, g, mode_lo, mode_hi,
                                           f, nullptr, scalar, flux_norm);
    }
    return numpy_from_array2d(result);
}

py::array_t<std::complex<double>> control_model_forward_wrapper(
    lina::ControlModel& model,
    py::array_t<double, py::array::c_style | py::array::forcecast> actuators,
    std::optional<double> wavelength,
    bool use_vortex) {
    auto acts = vector_from_numpy(actuators);
    const double wl = wavelength.value_or(model.wavelength());
    return numpy_from_array2d(model.forward(acts, wl, use_vortex));
}

py::array_t<std::uint8_t> control_model_dm_mask_wrapper(
    const lina::ControlModel& model) {
    return numpy_from_array2d(model.dm_mask());
}

std::string control_model_device_wrapper(const lina::ControlModel& model) {
    return model.device();
}

void control_model_set_device_wrapper(
    lina::ControlModel& model,
    const std::string& device) {
    model.set_device(device);
}

void control_model_set_prefpm_amp_wrapper(
    lina::ControlModel& model,
    py::array_t<double, py::array::c_style | py::array::forcecast> amp) {
    model.set_prefpm_amp(array2d_from_numpy<double>(amp));
}

void control_model_set_prefpm_opd_wrapper(
    lina::ControlModel& model,
    py::array_t<double, py::array::c_style | py::array::forcecast> opd) {
    model.set_prefpm_opd(array2d_from_numpy<double>(opd));
}

void control_model_set_aperture_wrapper(
    lina::ControlModel& model,
    py::array_t<double, py::array::c_style | py::array::forcecast> aperture) {
    model.set_aperture(array2d_from_numpy<double>(aperture));
}

void control_model_set_lyotstop_wrapper(
    lina::ControlModel& model,
    py::array_t<double, py::array::c_style | py::array::forcecast> lyotstop) {
    model.set_lyotstop(array2d_from_numpy<double>(lyotstop));
}

void control_model_set_windowed_vortex_lres_wrapper(
    lina::ControlModel& model,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> m) {
    model.set_windowed_vortex_lres(array2d_from_numpy<std::complex<double>>(m));
}

void control_model_set_windowed_vortex_hres_wrapper(
    lina::ControlModel& model,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> m) {
    model.set_windowed_vortex_hres(array2d_from_numpy<std::complex<double>>(m));
}

void control_model_set_dm_model_wrapper(
    lina::ControlModel& model,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> inf_fun_fft,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> mx_dm,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> my_dm,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> mx_dm_back,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> my_dm_back) {
    model.set_dm_model(
        array2d_from_numpy<std::complex<double>>(inf_fun_fft),
        array2d_from_numpy<std::complex<double>>(mx_dm),
        array2d_from_numpy<std::complex<double>>(my_dm),
        array2d_from_numpy<std::complex<double>>(mx_dm_back),
        array2d_from_numpy<std::complex<double>>(my_dm_back));
}

py::tuple control_model_dm_val_and_grad_wrapper(
    lina::ControlModel& model,
    py::array_t<double, py::array::c_style | py::array::forcecast> del_acts,
    py::array_t<double, py::array::c_style | py::array::forcecast> opd) {
    const auto acts = vector_from_numpy<double>(del_acts);
    const auto opd_arr = array2d_from_numpy<double>(opd);
    std::vector<double> grad;
    const double J = lina::dm_val_and_grad(model, acts, opd_arr, grad);
    return py::make_tuple(J, numpy_from_vector(grad));
}

py::array_t<double> control_model_solve_flat_wrapper(
    lina::ControlModel& model,
    py::array_t<double, py::array::c_style | py::array::forcecast> opd,
    double tol,
    int max_iter) {
    const auto opd_arr = array2d_from_numpy<double>(opd);
    const auto x = lina::solve_flat_command(model, opd_arr, tol, max_iter);
    return numpy_from_vector(x);
}

py::tuple control_model_val_and_grad_wrapper(
    py::array_t<double, py::array::c_style | py::array::forcecast> del_acts,
    lina::ControlModel& model,
    py::array_t<double, py::array::c_style | py::array::forcecast> current_acts,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> e_ab,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> e_fp_nom,
    py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> control_mask,
    std::optional<double> wavelength,
    double r_cond) {
    auto del = vector_from_numpy(del_acts);
    auto cur = vector_from_numpy(current_acts);
    auto eab = array2d_from_numpy<std::complex<double>>(e_ab);
    auto efp = array2d_from_numpy<std::complex<double>>(e_fp_nom);
    auto mask = array2d_from_numpy<std::uint8_t>(control_mask);
    const double wl = wavelength.value_or(model.wavelength());

    std::vector<double> grad;
    const double J = lina::val_and_grad(
        del, model, cur, eab, efp, mask, wl, r_cond, grad);
    return py::make_tuple(J, numpy_from_vector(grad));
}

py::tuple control_model_iefc_calibrate_wrapper(
    lina::ControlModel& model,
    py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> control_mask,
    double probe_amplitude,
    py::array_t<double, py::array::c_style | py::array::forcecast> probe_modes,
    double calibration_amplitude,
    py::array_t<double, py::array::c_style | py::array::forcecast> calibration_modes,
    py::object scale_factors,
    py::object initial_command,
    bool use_vortex,
    double imax_ref,
    py::object progress_cb) {
    auto mask = array2d_from_numpy<std::uint8_t>(control_mask);
    auto probes = modes_array2d_from_numpy<double>(probe_modes);
    auto modes = modes_array2d_from_numpy<double>(calibration_modes);

    std::vector<double> scales;
    if (!scale_factors.is_none()) {
        scales = vector_from_numpy<double>(
            py::cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(
                scale_factors));
    }

    std::optional<lina::Array2D<double>> init_cmd = std::nullopt;
    if (!initial_command.is_none()) {
        init_cmd = array2d_from_numpy<double>(
            py::cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(
                initial_command));
    }

    std::function<void(std::size_t, std::size_t)> progress;
    if (!progress_cb.is_none()) {
        progress = [progress_cb](std::size_t done, std::size_t total) {
            // The C++ loop holds the GIL, so calling back into Python is safe.
            progress_cb(done, total);
        };
    }

    const auto res = lina::calibrate_control_model(
        model,
        mask,
        probe_amplitude,
        probes,
        calibration_amplitude,
        modes,
        scales,
        init_cmd,
        use_vortex,
        imax_ref,
        progress);

    const auto nmodes = static_cast<py::ssize_t>(res.response_cube.size());
    const auto nprobes = static_cast<py::ssize_t>(res.nprobes);
    const auto ncamsci = static_cast<py::ssize_t>(res.ncamsci);
    py::array_t<double> cube({nmodes, nprobes, ncamsci, ncamsci});
    auto cube_mut = cube.mutable_unchecked<4>();
    for (py::ssize_t m = 0; m < nmodes; ++m) {
        const auto& mode_resp = res.response_cube[static_cast<std::size_t>(m)];
        for (py::ssize_t p = 0; p < nprobes; ++p) {
            for (py::ssize_t r = 0; r < ncamsci; ++r) {
                for (py::ssize_t c = 0; c < ncamsci; ++c) {
                    cube_mut(m, p, r, c) =
                        mode_resp(static_cast<std::size_t>(p),
                                  static_cast<std::size_t>(r * ncamsci + c));
                }
            }
        }
    }

    return py::make_tuple(
        numpy_from_array2d(res.response_matrix),
        cube);
}

} // namespace

// LINA_PYBIND_MODULE_NAME is set by CMake. It defaults to `lina_cpp` for the
// standalone build that the parity tests use; the lina_cpp Python package
// passes `_core` so the extension lives at `lina_cpp/_core.so` and the
// package __init__.py can re-export from it without a self-name collision.
#ifndef LINA_PYBIND_MODULE_NAME
#define LINA_PYBIND_MODULE_NAME lina_cpp
#endif

PYBIND11_MODULE(LINA_PYBIND_MODULE_NAME, m) {
    m.doc() = "Lina C++ bindings (pybind11) — math primitives for parity testing";

    // Build stamp so a smoke test can confirm exactly which compiled
    // .so was loaded. Bump LINA_BUILD_TAG whenever a C++ source change
    // needs to be observable from Python.
    m.attr("__build_tag__") = "llowfsc-safe-dot-row-noinline-2026.05.17";
    m.attr("__build_date__") = __DATE__ " " __TIME__;

    // FFT
    m.def("fft_cpu", &fft_cpu_wrapper, py::arg("array"),
          "2D FFT (CPU/FFTW backend) with centered shifts");
    m.def("ifft_cpu", &ifft_cpu_wrapper, py::arg("array"),
          "2D inverse FFT (CPU/FFTW backend) with centered shifts");
#ifdef LINA_USE_CUDA
    m.def("fft_gpu", &fft_gpu_wrapper, py::arg("array"),
          "2D FFT (GPU/cuFFT backend) with centered shifts");
    m.def("ifft_gpu", &ifft_gpu_wrapper, py::arg("array"),
          "2D inverse FFT (GPU/cuFFT backend) with centered shifts");
    m.def("ang_spec_gpu", &ang_spec_gpu_wrapper,
          py::arg("wavefront"), py::arg("wavelength"),
          py::arg("distance"), py::arg("pixelscale"),
          "Angular-spectrum propagation on the GPU.");
    m.def("make_vortex_phase_mask_gpu", &make_vortex_phase_mask_gpu_wrapper,
          py::arg("npix"), py::arg("charge") = 6, py::arg("grid") = "odd",
          "Vortex phase mask on the GPU.");
    m.def("get_fresnel_TF_gpu", &get_fresnel_TF_gpu_wrapper,
          py::arg("dz"), py::arg("n"), py::arg("wavelength"), py::arg("fnum"),
          "Fresnel defocus transfer function on the GPU.");
    m.def("mft_forward_gpu", &mft_forward_gpu_wrapper,
          py::arg("wavefront"), py::arg("npix"), py::arg("npsf"),
          py::arg("psf_pixelscale_lamD"),
          py::arg("convention") = "-",
          py::arg("pp_centering") = "odd",
          py::arg("fp_centering") = "odd",
          "Pupil -> focal MFT on the GPU (cuBLAS zgemm).");
    m.def("mft_reverse_gpu", &mft_reverse_gpu_wrapper,
          py::arg("fpwf"), py::arg("psf_pixelscale_lamD"),
          py::arg("npix"), py::arg("N"),
          py::arg("convention") = "+",
          py::arg("pp_centering") = "odd",
          py::arg("fp_centering") = "odd",
          "Focal -> pupil MFT on the GPU (cuBLAS zgemm).");
#endif

    // Propagation
    m.def("ang_spec", &ang_spec_wrapper,
          py::arg("wavefront"), py::arg("wavelength"), py::arg("distance"),
          py::arg("pixelscale"),
          "Angular spectrum propagation");
    m.def("get_fresnel_TF", &get_fresnel_TF_wrapper,
          py::arg("dz"), py::arg("n"), py::arg("wavelength"), py::arg("fnum"),
          "Fresnel defocus transfer function for a given dz, array size, "
          "wavelength, and beam f/#. Matches lina.props.get_fresnel_TF.");

    // FITS I/O
    m.def("save_fits", &save_fits_wrapper,
          py::arg("fpath"), py::arg("data"),
          py::arg("header") = py::none(),
          py::arg("ow") = true,
          py::arg("quiet") = false,
          "Save a 2D numpy array as the primary HDU of a FITS file. "
          "Mirrors lina.utils.save_fits.");
    m.def("load_fits", &load_fits_wrapper,
          py::arg("fpath"), py::arg("header") = false,
          "Load a 2D primary HDU as a numpy float64 array. If header=True, "
          "returns (data, header_dict). Mirrors lina.utils.load_fits.");
    m.def("fits_available", &lina::fits_available,
          "True if this build linked against cfitsio.");

    // wfe.py: math primitives for temporal-PSD synthesis and Zernike indexing.
    m.def("noll_index_to_mn", &noll_index_to_mn_wrapper, py::arg("j"));
    m.def("mn_to_noll_index", &lina::wfe::mn_to_noll_index, py::arg("m"), py::arg("n"));
    m.def("fringe_index_to_mn", &fringe_index_to_mn_wrapper, py::arg("j"));
    m.def("mn_to_fringe_index", &lina::wfe::mn_to_fringe_index, py::arg("m"), py::arg("n"));
    m.def("wfe_generate_freqs", &wfe_generate_freqs_wrapper,
          py::arg("delt"), py::arg("tmax"),
          "Returns (freqs, delf, times) for a given temporal grid.");
    m.def("wfe_roll_psd", &wfe_roll_psd_wrapper,
          py::arg("freqs"), py::arg("beta"), py::arg("f_roll"), py::arg("alpha"),
          py::arg("normalized") = true,
          "Knee PSD on a frequency grid.");
    m.def("wfe_generate_time_series", &wfe_generate_time_series_wrapper,
          py::arg("psd"), py::arg("freqs"),
          py::arg("rms_target") = 0.0, py::arg("seed") = 123ull,
          "Synthesize a real-valued time series from a one-sided PSD.");
    m.def("wfe_compute_cumulative_psd", &wfe_compute_cumulative_psd_wrapper,
          py::arg("freqs"), py::arg("psd"));

    // llowfsc.py: math kernels (acquire_ref, reconstruct, compute_zpo, loop_step).
    m.def("llowfsc_acquire_ref", &llowfsc_acquire_ref_wrapper,
          py::arg("camlo_ref_im"), py::arg("wfs_mask"),
          py::arg("dark_im") = 0.0, py::arg("flux_norm") = true,
          "Returns (ref_image, flux_norm_coeff). flux_norm_coeff is 0 when flux_norm=False.");
    m.def("llowfsc_reconstruct", &llowfsc_reconstruct_wrapper,
          py::arg("camlo_im"), py::arg("ref_im"), py::arg("wfs_mask"),
          py::arg("control_matrix"),
          py::arg("mode_lo"), py::arg("mode_hi"),
          py::arg("dark_im") = 0.0, py::arg("flux_norm") = true,
          py::arg("return_del_im") = false,
          "Project a dark-subtracted, flux-normalised, ref-subtracted image onto the "
          "[mode_lo, mode_hi) rows of `control_matrix`. Returns coeff or (coeff, del_im).");
    m.def("llowfsc_compute_zpo", &llowfsc_compute_zpo_wrapper,
          py::arg("dm_commands_masked"), py::arg("wfs_mask"),
          py::arg("response_matrix"), py::arg("dm_modal_matrix"),
          "Sum-project a list of masked DM commands through (R . M . cmd) into a 2D ZPO image.");
    m.def("llowfsc_loop_step", &llowfsc_loop_step_wrapper,
          py::arg("camlo_im"), py::arg("ref_plus_zpo"), py::arg("wfs_mask"),
          py::arg("control_matrix"), py::arg("dm_modes_flat"),
          py::arg("dm_rows"), py::arg("dm_cols"), py::arg("gains"),
          py::arg("mode_lo"), py::arg("mode_hi"), py::arg("ffo"),
          py::arg("dark_im") = 0.0, py::arg("flux_norm") = true,
          "One full math iteration: reconstruct -> -ffo -> -gain*coeff -> sum into "
          "delta DM command. Returns (dm_rows x dm_cols) Array2D.");
    m.def("make_vortex_phase_mask", &make_vortex_phase_mask_wrapper,
          py::arg("npix"), py::arg("charge") = 6, py::arg("grid") = "odd",
          "Vortex phase mask");
    m.def("mft_forward", &mft_forward_wrapper,
          py::arg("wavefront"), py::arg("npix"), py::arg("npsf"),
          py::arg("psf_pixelscale_lamD"),
          py::arg("convention") = "-",
          py::arg("pp_centering") = "odd",
          py::arg("fp_centering") = "odd",
          "Forward Matrix Fourier Transform");
    m.def("mft_reverse", &mft_reverse_wrapper,
          py::arg("fpwf"), py::arg("psf_pixelscale_lamD"),
          py::arg("npix"), py::arg("N"),
          py::arg("convention") = "+",
          py::arg("pp_centering") = "odd",
          py::arg("fp_centering") = "odd",
          "Reverse Matrix Fourier Transform");

    // Utils
    m.def("mean", &mean_wrapper, py::arg("array"));
    m.def("mean_masked", &mean_masked_wrapper, py::arg("array"), py::arg("mask_flat"));
    m.def("rms", &rms_wrapper, py::arg("array"));
    m.def("make_grid", &make_grid_wrapper,
          py::arg("npix"), py::arg("pixelscale") = 1.0, py::arg("half_shift") = false);
    m.def("pad_or_crop", &pad_or_crop_wrapper,
          py::arg("array"), py::arg("npix"));
    m.def("create_annular_mask", &create_annular_mask_wrapper,
          py::arg("n"), py::arg("pixelscale"), py::arg("irad"), py::arg("orad"),
          py::arg("edge") = py::none(), py::arg("x_shift") = 0.0, py::arg("y_shift") = 0.0,
          py::arg("rotation") = 0.0,
          "edge=None disables the half-plane filter; pass a numeric value "
          "(including 0.0) to filter at xr > edge.");
    m.def("create_annular_focal_plane_mask", &create_annular_focal_plane_mask_wrapper,
          py::arg("npsf"), py::arg("psf_pixelscale"), py::arg("irad"), py::arg("orad"),
          py::arg("edge") = py::none(), py::arg("centering") = "odd",
          py::arg("rotation") = 0.0, py::arg("x_shift") = 0.0, py::arg("y_shift") = 0.0,
          "edge=None disables the half-plane filter; pass a numeric value "
          "(including 0.0) to filter at xr > edge.");

    // DM
    m.def("create_mask", &create_mask_wrapper, py::arg("nact"));
    m.def("make_gaussian_inf_fun", &make_gaussian_inf_fun_wrapper,
          py::arg("act_spacing") = 300e-6,
          py::arg("sampling") = 10.0,
          py::arg("coupling") = 0.15,
          py::arg("nact") = 4);
    m.def("create_hadamard_modes", &create_hadamard_modes_wrapper, py::arg("dm_mask"));
    m.def("create_fourier_modes", &create_fourier_modes_wrapper,
          py::arg("dm_mask"), py::arg("npsf"), py::arg("psf_pixelscale_lamD"),
          py::arg("iwa"), py::arg("owa"),
          py::arg("rotation") = 0.0,
          py::arg("fourier_sampling") = 0.75,
          py::arg("which") = "both");
    m.def("make_fourier_command", &make_fourier_command_wrapper,
          py::arg("x_cpa") = 10, py::arg("y_cpa") = 10,
          py::arg("nact") = 34, py::arg("phase") = 0.0);
    m.def("make_f", &make_f_wrapper,
          py::arg("h") = 10, py::arg("w") = 6,
          py::arg("shift_x") = -1, py::arg("shift_y") = 0,
          py::arg("nact") = 34);
    m.def("make_ring", &make_ring_wrapper,
          py::arg("rad") = 15.0, py::arg("nact") = 34,
          py::arg("thresh") = 0.5);
    m.def("make_cross_command", &make_cross_command_wrapper,
          py::arg("xc"), py::arg("yc"), py::arg("nact") = 34);

    // Linear algebra
    m.def("gemm", &gemm_wrapper,
          py::arg("a"), py::arg("b"),
          py::arg("transpose_a") = false, py::arg("transpose_b") = false);
    m.def("gemv", &gemv_wrapper,
          py::arg("a"), py::arg("x"), py::arg("transpose_a") = false);
    m.def("lstsq", &lstsq_wrapper,
          py::arg("modes"), py::arg("data"),
          "Least-squares projection used by lina.utils.lstsq.");
    m.def("tikhonov_inverse", &tikhonov_inverse_wrapper,
          py::arg("a"), py::arg("rcond") = 1e-15,
          py::arg("return_all") = false,
          "SVD-based pseudo-inverse with Tikhonov regularization.");
    m.def("beta_reg", &beta_reg_wrapper,
          py::arg("S"), py::arg("beta") = -1.0,
          "Compute beta-regularized control matrix (CPU).");
#ifdef LINA_USE_CUDA
    m.def("beta_reg_gpu", &beta_reg_gpu_wrapper,
          py::arg("S"), py::arg("beta") = -1.0,
          "Compute beta-regularized control matrix natively on the GPU "
          "(cuBLAS GEMM + cuSOLVER Cholesky solve).");
#endif
    m.def("svd", &svd_wrapper, py::arg("a"),
          "SVD (double, full matrices). Returns (U, s, Vt).");
    m.def("svd_float_cpu", &svd_float_cpu_wrapper, py::arg("a"),
          "SVD (float, CPU/LAPACKE)");
#ifdef LINA_USE_CUDA
    m.def("svd_float_gpu", &svd_float_gpu_wrapper, py::arg("a"),
          "SVD (float, GPU/cuSOLVER)");
#endif
    m.def("set_num_threads", &lina::set_num_threads, py::arg("nthreads"),
          "Set CPU math thread count (OpenBLAS / OpenMP when enabled).");
    m.def("get_num_threads", &lina::get_num_threads,
          "Get current CPU math thread count.");

    // Coronagraph utilities
    m.def("normalize_coro_im", &normalize_coro_im_wrapper,
          py::arg("raw_im"),
          py::arg("exp_time_im"), py::arg("gain_im"), py::arg("atten_im"),
          py::arg("exp_time_ref"), py::arg("gain_ref"), py::arg("atten_ref"),
          py::arg("Imax"),
          py::arg("dark_im") = py::none());
    m.def("compute_contrast", &compute_contrast_wrapper,
          py::arg("ni_im"), py::arg("mask"),
          "Returns (contrast, n_mask, n_positive)");

    // iEFC helpers
    m.def("compute_hadamard_scale_factors", &compute_hadamard_scale_factors_wrapper,
          py::arg("had_modes"), py::arg("scale_exp") = 1.0 / 6.0,
          py::arg("scale_thresh") = 4.0, py::arg("iwa") = 2.5,
          py::arg("owa") = 13.0, py::arg("oversamp") = 4);

    // C++ control model surface (optical simulation backend).
    py::class_<lina::ControlModel>(m, "ControlModelCpp")
        .def(py::init<
                 double, std::optional<double>, std::size_t, std::size_t,
                 std::size_t, double, double, double, double, double, double,
                 std::optional<double>, double, std::size_t, std::size_t,
                 double, double>(),
             py::arg("wavelength_c") = 630e-9,
             py::arg("wavelength") = std::nullopt,
             py::arg("npix") = 500,
             py::arg("ndef") = 502,
             py::arg("n_vortex_lres") = 2048,
             py::arg("vortex_win_diam") = 30.0,
             py::arg("vortex_hres_sampling") = 0.025,
             py::arg("vortex_dot_mask_diam_lamDc") = 0.5,
             py::arg("dm_beam_diam") = 9.3e-3,
             py::arg("lyot_pupil_diam") = 9.1e-3,
             py::arg("lyot_stop_diam") = 8.6e-3,
             py::arg("exit_pupil_prop_dist") = std::nullopt,
             py::arg("camsci_pxscl_lamDc") = 0.2,
             py::arg("ncamsci") = 256,
             py::arg("nact") = 34,
             py::arg("act_spacing") = 300e-6,
             py::arg("act_coupling") = 0.15)
        .def("nacts", &lina::ControlModel::nacts)
        .def("wavelength", &lina::ControlModel::wavelength)
        .def("ndef", &lina::ControlModel::ndef)
        .def("device", &control_model_device_wrapper)
        .def("set_device", &control_model_set_device_wrapper,
             py::arg("device"),
             "Set ControlModel backend: 'cpu' or 'gpu'.")
        .def("dm_mask", &control_model_dm_mask_wrapper,
             "Boolean DM mask as uint8 array.")
        .def("set_prefpm_amp", &control_model_set_prefpm_amp_wrapper,
             py::arg("amp"),
             "Set pre-FPM amplitude map (shape ndef x ndef).")
        .def("set_prefpm_opd", &control_model_set_prefpm_opd_wrapper,
             py::arg("opd"),
             "Set pre-FPM OPD map (shape ndef x ndef).")
        .def("set_aperture", &control_model_set_aperture_wrapper,
             py::arg("aperture"),
             "Override entrance-pupil aperture (shape ndef x ndef).")
        .def("set_lyotstop", &control_model_set_lyotstop_wrapper,
             py::arg("lyotstop"),
             "Override Lyot stop mask (shape ndef x ndef).")
        .def("set_windowed_vortex_lres", &control_model_set_windowed_vortex_lres_wrapper,
             py::arg("m"),
             "Override windowed low-res vortex FPM (complex, N_vortex_lres^2).")
        .def("set_windowed_vortex_hres", &control_model_set_windowed_vortex_hres_wrapper,
             py::arg("m"),
             "Override windowed high-res vortex FPM (complex, N_vortex_hres^2).")
        .def("set_dm_model", &control_model_set_dm_model_wrapper,
             py::arg("inf_fun_fft"), py::arg("mx_dm"), py::arg("my_dm"),
             py::arg("mx_dm_back"), py::arg("my_dm_back"),
             "Override DM influence-function FFT and MFT matrices (complex).")
        .def("forward", &control_model_forward_wrapper,
             py::arg("actuators"),
             py::arg("wavelength") = std::nullopt,
             py::arg("use_vortex") = true,
             "Forward optical simulation (returns focal-plane complex field).");

    m.def("control_model_val_and_grad", &control_model_val_and_grad_wrapper,
          py::arg("del_acts"), py::arg("model"), py::arg("current_acts"),
          py::arg("e_ab"), py::arg("e_fp_nom"), py::arg("control_mask"),
          py::arg("wavelength") = std::nullopt, py::arg("r_cond") = 1e-3,
          "C++ val_and_grad for the EFC objective. Returns (J, grad).");
    m.def("control_model_dm_val_and_grad", &control_model_dm_val_and_grad_wrapper,
          py::arg("model"), py::arg("del_acts"), py::arg("opd"),
          "Native DM-surface fit objective+gradient. Returns (J, grad).");
    m.def("control_model_solve_flat", &control_model_solve_flat_wrapper,
          py::arg("model"), py::arg("opd"),
          py::arg("tol") = 1e-4, py::arg("max_iter") = 0,
          "Solve the flat-DM actuator vector via native L-BFGS. Returns del_acts.");
    m.def("control_model_iefc_calibrate", &control_model_iefc_calibrate_wrapper,
          py::arg("model"),
          py::arg("control_mask"),
          py::arg("probe_amplitude"),
          py::arg("probe_modes"),
          py::arg("calibration_amplitude"),
          py::arg("calibration_modes"),
          py::arg("scale_factors") = py::none(),
          py::arg("initial_command") = py::none(),
          py::arg("use_vortex") = true,
          py::arg("imax_ref") = 1.0,
          py::arg("progress_cb") = py::none(),
          "Native iEFC calibration for ControlModel. Returns "
          "(response_matrix, response_cube).");
}
