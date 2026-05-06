#include "lina/llowfsc.h"

#include "lina/linalg.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace lina::llowfsc {

namespace {

// Common shape check used everywhere in this file.
void require_same_shape(const char* who,
                        std::size_t a_rows, std::size_t a_cols,
                        std::size_t b_rows, std::size_t b_cols) {
    if (a_rows != b_rows || a_cols != b_cols) {
        throw std::invalid_argument(
            std::string(who) + ": shape mismatch ("
            + std::to_string(a_rows) + "x" + std::to_string(a_cols) + " vs "
            + std::to_string(b_rows) + "x" + std::to_string(b_cols) + ")");
    }
}

// Sum of `arr` restricted to entries where mask != 0.
double sum_masked(const Array2D<double>& arr,
                  const Array2D<std::uint8_t>& mask) {
    double s = 0.0;
    const std::size_t n = arr.size();
    const double* p = arr.data();
    const std::uint8_t* m = mask.data();
    for (std::size_t i = 0; i < n; ++i) {
        if (m[i]) s += p[i];
    }
    return s;
}

// Count of mask entries that are non-zero.
std::size_t count_mask(const Array2D<std::uint8_t>& mask) {
    std::size_t c = 0;
    const std::size_t n = mask.size();
    const std::uint8_t* m = mask.data();
    for (std::size_t i = 0; i < n; ++i) {
        if (m[i]) ++c;
    }
    return c;
}

// Extract `arr[mask]` into a flat std::vector<double>. The order
// matches NumPy's row-major iteration (the same as lina.llowfsc).
std::vector<double> extract_masked(const Array2D<double>& arr,
                                   const Array2D<std::uint8_t>& mask) {
    std::vector<double> out;
    out.reserve(count_mask(mask));
    const std::size_t n = arr.size();
    const double* p = arr.data();
    const std::uint8_t* m = mask.data();
    for (std::size_t i = 0; i < n; ++i) {
        if (m[i]) out.push_back(p[i]);
    }
    return out;
}

// Scatter `vals` into `out[mask]` (out has the same shape as mask).
// Entries where mask == 0 are left untouched.
void scatter_masked(Array2D<double>& out,
                    const Array2D<std::uint8_t>& mask,
                    const std::vector<double>& vals) {
    const std::size_t n = mask.size();
    const std::uint8_t* m = mask.data();
    double* p = out.data();
    std::size_t k = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (m[i]) {
            if (k >= vals.size())
                throw std::invalid_argument("scatter_masked: vals shorter than mask count");
            p[i] = vals[k++];
        }
    }
    if (k != vals.size())
        throw std::invalid_argument("scatter_masked: vals longer than mask count");
}

} // namespace

// ---------------------------------------------------------------------------
// acquire_ref kernels
// ---------------------------------------------------------------------------

AcquireRefResult acquire_ref(const Array2D<double>& camlo_ref_im,
                             const Array2D<std::uint8_t>& wfs_mask,
                             const Array2D<double>& dark_im,
                             bool flux_norm) {
    require_same_shape("acquire_ref",
                       camlo_ref_im.rows(), camlo_ref_im.cols(),
                       wfs_mask.rows(), wfs_mask.cols());
    require_same_shape("acquire_ref",
                       camlo_ref_im.rows(), camlo_ref_im.cols(),
                       dark_im.rows(), dark_im.cols());

    AcquireRefResult res;
    res.ref_image = Array2D<double>(camlo_ref_im.rows(), camlo_ref_im.cols(), 0.0);
    const std::size_t n = camlo_ref_im.size();
    for (std::size_t i = 0; i < n; ++i) {
        const double v = (camlo_ref_im.data()[i] - dark_im.data()[i]) * (wfs_mask.data()[i] ? 1.0 : 0.0);
        res.ref_image.data()[i] = v;
    }

    res.flux_norm_coeff = 0.0;
    if (flux_norm) {
        const double s = sum_masked(res.ref_image, wfs_mask);
        if (s == 0.0) {
            throw std::runtime_error("acquire_ref: masked sum is zero, cannot flux-normalize");
        }
        for (std::size_t i = 0; i < n; ++i) {
            res.ref_image.data()[i] /= s;
        }
        res.flux_norm_coeff = s;
    }
    return res;
}

AcquireRefResult acquire_ref(const Array2D<double>& camlo_ref_im,
                             const Array2D<std::uint8_t>& wfs_mask,
                             double dark_scalar,
                             bool flux_norm) {
    Array2D<double> dark(camlo_ref_im.rows(), camlo_ref_im.cols(), dark_scalar);
    return acquire_ref(camlo_ref_im, wfs_mask, dark, flux_norm);
}

// ---------------------------------------------------------------------------
// reconstruct kernel
// ---------------------------------------------------------------------------

std::vector<double> reconstruct(const Array2D<double>& camlo_im,
                                const Array2D<double>& ref_im,
                                const Array2D<std::uint8_t>& wfs_mask,
                                const Array2D<double>& control_matrix,
                                std::size_t mode_lo,
                                std::size_t mode_hi,
                                const Array2D<double>* dark_im,
                                double dark_scalar,
                                bool flux_norm,
                                Array2D<double>* del_im_out) {
    require_same_shape("reconstruct(camlo,ref)",
                       camlo_im.rows(), camlo_im.cols(),
                       ref_im.rows(), ref_im.cols());
    require_same_shape("reconstruct(camlo,mask)",
                       camlo_im.rows(), camlo_im.cols(),
                       wfs_mask.rows(), wfs_mask.cols());

    const std::size_t Nmask = count_mask(wfs_mask);
    if (control_matrix.cols() != Nmask) {
        throw std::invalid_argument(
            "reconstruct: control_matrix has " + std::to_string(control_matrix.cols())
            + " columns but wfs_mask has " + std::to_string(Nmask) + " active pixels");
    }
    if (mode_hi > control_matrix.rows() || mode_lo > mode_hi) {
        throw std::invalid_argument(
            "reconstruct: mode range [" + std::to_string(mode_lo)
            + "," + std::to_string(mode_hi) + ") out of bounds for "
            + std::to_string(control_matrix.rows()) + " modes");
    }

    // 1) dark-subtract (in-place into a working buffer)
    Array2D<double> work(camlo_im.rows(), camlo_im.cols(), 0.0);
    const std::size_t n = camlo_im.size();
    if (dark_im) {
        require_same_shape("reconstruct(dark)",
                           camlo_im.rows(), camlo_im.cols(),
                           dark_im->rows(), dark_im->cols());
        for (std::size_t i = 0; i < n; ++i) {
            work.data()[i] = camlo_im.data()[i] - dark_im->data()[i];
        }
    } else {
        for (std::size_t i = 0; i < n; ++i) {
            work.data()[i] = camlo_im.data()[i] - dark_scalar;
        }
    }

    // 2) optional flux normalization by the masked sum of the dark-subtracted image
    if (flux_norm) {
        const double s = sum_masked(work, wfs_mask);
        if (s != 0.0) {
            for (std::size_t i = 0; i < n; ++i) work.data()[i] /= s;
        }
    }

    // 3) del_im = camlo_im_flux_norm - ref_im
    for (std::size_t i = 0; i < n; ++i) {
        work.data()[i] -= ref_im.data()[i];
    }
    if (del_im_out) {
        *del_im_out = work; // copy out before continuing
    }

    // 4) Extract masked entries and gemv against the [mode_lo, mode_hi) slice.
    std::vector<double> del_masked = extract_masked(work, wfs_mask);
    std::vector<double> coeff(mode_hi - mode_lo, 0.0);

    // The control matrix is row-major (Nmodes, Nmask). We do
    //     coeff[k] = sum_j  control_matrix[mode_lo+k][j] * del_masked[j]
    // which is gemv with stride = Nmask.
    const double* C = control_matrix.data();
    const std::size_t Ncol = control_matrix.cols();
    for (std::size_t k = 0; k < coeff.size(); ++k) {
        const double* row = C + (mode_lo + k) * Ncol;
        double s = 0.0;
        for (std::size_t j = 0; j < Ncol; ++j) {
            s += row[j] * del_masked[j];
        }
        coeff[k] = s;
    }
    return coeff;
}

// ---------------------------------------------------------------------------
// compute_zpo kernel
// ---------------------------------------------------------------------------

Array2D<double> compute_zpo(const std::vector<std::vector<double>>& dm_commands_masked,
                            const Array2D<std::uint8_t>& wfs_mask,
                            const Array2D<double>& response_matrix,
                            const Array2D<double>& dm_modal_matrix) {
    const std::size_t Nmask = count_mask(wfs_mask);
    if (response_matrix.rows() != Nmask) {
        throw std::invalid_argument(
            "compute_zpo: response_matrix has " + std::to_string(response_matrix.rows())
            + " rows but wfs_mask has " + std::to_string(Nmask) + " active pixels");
    }
    const std::size_t Nmodes = response_matrix.cols();
    if (dm_modal_matrix.rows() != Nmodes) {
        throw std::invalid_argument(
            "compute_zpo: dm_modal_matrix.rows (" + std::to_string(dm_modal_matrix.rows())
            + ") != response_matrix.cols (" + std::to_string(Nmodes) + ")");
    }
    const std::size_t Ndm = dm_modal_matrix.cols();

    std::vector<double> zpo_masked(Nmask, 0.0);
    std::vector<double> tmp_modes(Nmodes, 0.0);
    std::vector<double> tmp_pixels(Nmask, 0.0);

    for (const auto& dm_cmd : dm_commands_masked) {
        if (dm_cmd.size() != Ndm) {
            throw std::invalid_argument(
                "compute_zpo: dm_command size " + std::to_string(dm_cmd.size())
                + " != dm_modal_matrix.cols " + std::to_string(Ndm));
        }

        // tmp_modes = dm_modal_matrix . dm_cmd   (Nmodes x Ndm) * (Ndm) -> (Nmodes)
        const double* M = dm_modal_matrix.data();
        for (std::size_t i = 0; i < Nmodes; ++i) {
            const double* row = M + i * Ndm;
            double s = 0.0;
            for (std::size_t j = 0; j < Ndm; ++j) s += row[j] * dm_cmd[j];
            tmp_modes[i] = s;
        }

        // tmp_pixels = response_matrix . tmp_modes  (Nmask x Nmodes) * (Nmodes) -> (Nmask)
        const double* R = response_matrix.data();
        for (std::size_t i = 0; i < Nmask; ++i) {
            const double* row = R + i * Nmodes;
            double s = 0.0;
            for (std::size_t j = 0; j < Nmodes; ++j) s += row[j] * tmp_modes[j];
            tmp_pixels[i] = s;
        }

        // zpo_masked += tmp_pixels
        for (std::size_t i = 0; i < Nmask; ++i) zpo_masked[i] += tmp_pixels[i];
    }

    Array2D<double> zpo(wfs_mask.rows(), wfs_mask.cols(), 0.0);
    scatter_masked(zpo, wfs_mask, zpo_masked);
    return zpo;
}

// ---------------------------------------------------------------------------
// loop_step kernel: full per-iteration math
// ---------------------------------------------------------------------------

Array2D<double> loop_step(const Array2D<double>& camlo_im,
                          const Array2D<double>& ref_plus_zpo,
                          const Array2D<std::uint8_t>& wfs_mask,
                          const Array2D<double>& control_matrix,
                          const Array2D<double>& dm_modes_flat,
                          std::size_t dm_rows,
                          std::size_t dm_cols,
                          const std::vector<double>& gains,
                          std::size_t mode_lo,
                          std::size_t mode_hi,
                          const std::vector<double>& ffo,
                          const Array2D<double>* dark_im,
                          double dark_scalar,
                          bool flux_norm) {
    const std::size_t Nmodes_active = mode_hi - mode_lo;
    if (gains.size() < Nmodes_active) {
        throw std::invalid_argument("loop_step: gains shorter than active mode range");
    }
    if (ffo.size() != Nmodes_active && !ffo.empty()) {
        throw std::invalid_argument("loop_step: ffo length must match (mode_hi - mode_lo) or be empty");
    }
    if (dm_modes_flat.cols() != dm_rows * dm_cols) {
        throw std::invalid_argument("loop_step: dm_modes_flat.cols != dm_rows*dm_cols");
    }
    if (dm_modes_flat.rows() < mode_hi) {
        throw std::invalid_argument("loop_step: dm_modes_flat has fewer rows than mode_hi");
    }

    // 1) reconstruct
    std::vector<double> coeff = reconstruct(
        camlo_im, ref_plus_zpo, wfs_mask, control_matrix,
        mode_lo, mode_hi, dark_im, dark_scalar, flux_norm);

    // 2) coeff -= ffo (when provided)
    if (!ffo.empty()) {
        for (std::size_t i = 0; i < coeff.size(); ++i) coeff[i] -= ffo[i];
    }

    // 3) modal_coeff = - gains * coeff (elementwise; gains is indexed
    //    relative to mode_lo so gains[0] applies to mode mode_lo.)
    std::vector<double> modal_coeff(coeff.size());
    for (std::size_t i = 0; i < coeff.size(); ++i) {
        modal_coeff[i] = -gains[i] * coeff[i];
    }

    // 4) del_dm_command = sum_i modal_coeff[i] * dm_modes_flat[mode_lo+i, :]
    Array2D<double> del_dm(dm_rows, dm_cols, 0.0);
    const std::size_t Ndm = dm_rows * dm_cols;
    const double* MFlat = dm_modes_flat.data();
    double* dst = del_dm.data();
    for (std::size_t i = 0; i < modal_coeff.size(); ++i) {
        const double w = modal_coeff[i];
        if (w == 0.0) continue;
        const double* row = MFlat + (mode_lo + i) * Ndm;
        for (std::size_t k = 0; k < Ndm; ++k) dst[k] += w * row[k];
    }
    return del_dm;
}

} // namespace lina::llowfsc
