#include "lina/lina.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <complex>
#include <cstring>
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
    std::size_t npix, std::size_t npsf, double psf_pixelscale_lamD,
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
    double psf_pixelscale_lamD, std::size_t npix, std::size_t N,
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
                                                      std::size_t npix,
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
                                                      std::size_t npix,
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
    m.def("svd", &svd_wrapper, py::arg("a"),
          "SVD (double, full matrices). Returns (U, s, Vt).");
    m.def("svd_float_cpu", &svd_float_cpu_wrapper, py::arg("a"),
          "SVD (float, CPU/LAPACKE)");
#ifdef LINA_USE_CUDA
    m.def("svd_float_gpu", &svd_float_gpu_wrapper, py::arg("a"),
          "SVD (float, GPU/cuSOLVER)");
#endif

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
}
