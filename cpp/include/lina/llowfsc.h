#pragma once

// LLOWFS math kernels.
//
// This is the C++ port of the math hot path inside lina/llowfsc.py.
// The high-level run() loop -- which orchestrates camera reads / DM
// writes / per-iteration logging -- stays in Python (it is the same
// for both backends; what we want here is fast inner-loop math).
//
// The functions below operate on row-major Array2D buffers and 1D
// std::vectors so they can be called either:
//   1. from a future native C++ MagAOX llowfsc app, or
//   2. from Python via the lina_cpp pybind extension (used by
//      lina_cpp.llowfsc).
//
// Numerical equivalence with lina.llowfsc is verified by
// lina/tests/test_per_method_parity.py.

#include "lina/array.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace lina::llowfsc {

// ---------------------------------------------------------------------------
// acquire_ref math kernel.
//
// camlo_ref_im - dark, masked to wfs_mask, optionally flux-normalized
// so the mean count rate inside the mask is 1.
//
// Inputs:
//   camlo_ref_im (rows x cols)  -- raw camera frame
//   wfs_mask     (rows x cols)  -- 0/1 mask of the WFS region
//   dark_im      (rows x cols)  -- dark frame, OR a single scalar dark via the overload
// Outputs:
//   ref_image    (rows x cols)  -- dark-subtracted, masked, optionally normalized
//   flux_norm_coeff             -- sum of the masked region (the divisor); 0 if not flux-normalized
// ---------------------------------------------------------------------------
struct AcquireRefResult {
    Array2D<double> ref_image;
    double flux_norm_coeff;  // 0.0 means "no flux normalization was applied"
};

AcquireRefResult acquire_ref(const Array2D<double>& camlo_ref_im,
                             const Array2D<std::uint8_t>& wfs_mask,
                             const Array2D<double>& dark_im,
                             bool flux_norm = true);

AcquireRefResult acquire_ref(const Array2D<double>& camlo_ref_im,
                             const Array2D<std::uint8_t>& wfs_mask,
                             double dark_scalar = 0.0,
                             bool flux_norm = true);

// ---------------------------------------------------------------------------
// reconstruct math kernel.
//
//   camlo_im_dark_sub   = camlo_im - dark_im
//   camlo_im_flux_norm  = camlo_im_dark_sub / sum(camlo_im_dark_sub[wfs_mask])
//                         (skipped when flux_norm == false)
//   del_im              = camlo_im_flux_norm - ref_im
//   coeff               = control_matrix[mode_lo:mode_hi].dot(del_im[wfs_mask])
//
// `control_matrix` is shape (Nmodes, Nmask) row-major, matching what
// lina.llowfsc.calibrate_dm_modes returns. We always slice by mode
// indices [mode_lo, mode_hi) along the leading axis so the caller can
// run only the modes they care about that iteration.
//
// `del_im_out` (when non-null) receives camlo_im_flux_norm - ref_im,
// matching the `return_del_im=True` Python branch.
// ---------------------------------------------------------------------------
std::vector<double> reconstruct(const Array2D<double>& camlo_im,
                                const Array2D<double>& ref_im,
                                const Array2D<std::uint8_t>& wfs_mask,
                                const Array2D<double>& control_matrix,
                                std::size_t mode_lo,
                                std::size_t mode_hi,
                                const Array2D<double>* dark_im = nullptr,
                                double dark_scalar = 0.0,
                                bool flux_norm = true,
                                Array2D<double>* del_im_out = nullptr);

// ---------------------------------------------------------------------------
// compute_zpo math kernel.
//
// For each of `n_streams` DM commands, project through dm_modal_matrix
// and response_matrix, accumulating the result in a 2D image at the
// pixels selected by wfs_mask.
//
//   zpo[wfs_mask] = sum_i  R . (M . dm_command_i[dm_mask])
//
// where R is response_matrix (Nmask x Nmodes), M is dm_modal_matrix
// (Nmodes x Ndm), and each dm_command_i is the *masked* DM command
// already extracted into a length-Ndm vector. Returns the rows x cols
// image with the projected ZPO. Caller is responsible for writing it
// to a shmim if desired.
// ---------------------------------------------------------------------------
Array2D<double> compute_zpo(const std::vector<std::vector<double>>& dm_commands_masked,
                            const Array2D<std::uint8_t>& wfs_mask,
                            const Array2D<double>& response_matrix,
                            const Array2D<double>& dm_modal_matrix);

// ---------------------------------------------------------------------------
// loop_step kernel: one iteration of the closed-loop math, returning
// the delta DM command to be added to the current DM. Combines:
//   - reconstruct (above)
//   - subtract feed-forward offset (ffo)
//   - apply per-mode gains (modal_coeff = - gains * recon_coeff)
//   - synthesize delta DM command: sum_i modal_coeff[i] * dm_modes[i]
//
// dm_modes has shape (Nmodes, dm_rows, dm_cols) flattened to row-major
// (Nmodes, dm_rows*dm_cols).
//
// Returns a (dm_rows x dm_cols) Array2D containing the delta DM
// command. The Python caller then adds it to the current DM and
// writes via set_dm_fun.
// ---------------------------------------------------------------------------
Array2D<double> loop_step(const Array2D<double>& camlo_im,
                          const Array2D<double>& ref_plus_zpo,
                          const Array2D<std::uint8_t>& wfs_mask,
                          const Array2D<double>& control_matrix,
                          const Array2D<double>& dm_modes_flat, // (Nmodes, dm_rows*dm_cols)
                          std::size_t dm_rows,
                          std::size_t dm_cols,
                          const std::vector<double>& gains,     // length Nmodes (or [mode_lo, mode_hi))
                          std::size_t mode_lo,
                          std::size_t mode_hi,
                          const std::vector<double>& ffo,       // length (mode_hi - mode_lo)
                          const Array2D<double>* dark_im = nullptr,
                          double dark_scalar = 0.0,
                          bool flux_norm = true);

} // namespace lina::llowfsc
