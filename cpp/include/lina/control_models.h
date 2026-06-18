#pragma once

#include "lina/array.h"
#include "lina/efc.h"

#include <complex>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace lina {

// Opaque GPU-resident state (device buffers + plans). Defined in
// control_models_cuda.cu; only instantiated when LINA_USE_CUDA is on.
struct ControlModelGpuState;

class ControlModel : public EfcModel {
public:
    struct ForwardResult {
        Array2D<std::complex<double>> e_fp;
        Array2D<std::complex<double>> e_ep;
        Array2D<std::complex<double>> dm_phasor;
    };

    ControlModel(double wavelength_c = 630e-9,
                 std::optional<double> wavelength = std::nullopt,
                 std::size_t npix = 500,
                 std::size_t ndef = 502,
                 std::size_t n_vortex_lres = 2048,
                 double vortex_win_diam = 30.0,
                 double vortex_hres_sampling = 0.025,
                 double vortex_dot_mask_diam_lamDc = 0.5,
                 double dm_beam_diam = 9.3e-3,
                 double lyot_pupil_diam = 9.1e-3,
                 double lyot_stop_diam = 8.6e-3,
                 std::optional<double> exit_pupil_prop_dist = std::nullopt,
                 double camsci_pxscl_lamDc = 0.2,
                 std::size_t ncamsci = 256,
                 std::size_t nact = 34,
                 double act_spacing = 300e-6,
                 double act_coupling = 0.15);

    std::size_t nacts() const override { return nacts_; }
    double wavelength() const override { return wavelength_; }
    std::size_t ndef() const { return ndef_; }
    std::string device() const { return use_gpu_ ? "gpu" : "cpu"; }
    void set_device(const std::string& device);

    Array2D<std::complex<double>> forward(const std::vector<double>& actuators,
                                          double wavelength,
                                          bool use_vortex) override;

    const Array2D<std::uint8_t>& dm_mask() const { return dm_mask_; }
    void set_prefpm_amp(const Array2D<double>& amp);
    void set_prefpm_opd(const Array2D<double>& opd);
    // Override the entrance-pupil aperture / Lyot stop with externally supplied
    // masks (e.g. poppy's anti-aliased transmission), so results match the
    // pure-Python reference exactly instead of a hard-edged C++ circle.
    void set_aperture(const Array2D<double>& aperture);
    void set_lyotstop(const Array2D<double>& lyotstop);
    // Override the precomputed windowed vortex focal-plane masks (low-res FFT
    // branch and high-res MFT branch), so the coronagraph path matches the
    // pure-Python reference (which uses scipy Tukey windows + poppy dot mask).
    void set_windowed_vortex_lres(const Array2D<std::complex<double>>& m);
    void set_windowed_vortex_hres(const Array2D<std::complex<double>>& m);
    // Override the DM influence-function FFT and the DM MFT matrices with the
    // pure-Python reference's exact arrays, so the DM-surface model (and its
    // adjoint used in dm_val_and_grad) matches to floating-point precision.
    void set_dm_model(const Array2D<std::complex<double>>& inf_fun_fft,
                      const Array2D<std::complex<double>>& mx_dm,
                      const Array2D<std::complex<double>>& my_dm,
                      const Array2D<std::complex<double>>& mx_dm_back,
                      const Array2D<std::complex<double>>& my_dm_back);

private:
    double wavelength_c_;
    double wavelength_;
    double dm_beam_diam_;
    double lyot_pupil_diam_;
    double lyot_stop_diam_;
    double lyot_ratio_;
    std::optional<double> exit_pupil_prop_dist_;
    double camsci_pxscl_lamDc_;
    double camsci_pxscl_lamD_;
    std::size_t npix_;
    std::size_t ndef_;
    double def_oversample_;
    std::size_t ncamsci_;
    double exit_pupil_pxscl_;

    std::size_t nact_;
    std::size_t nacts_;
    double act_spacing_;
    double dm_pxscl_;
    double inf_sampling_;

    Array2D<double> aperture_;
    Array2D<double> lyotstop_;
    Array2D<double> prefpm_amp_;
    Array2D<double> prefpm_opd_;

    Array2D<std::uint8_t> dm_mask_;

    Array2D<double> inf_fun_;
    std::size_t nsurf_;
    Array2D<std::complex<double>> inf_fun_fft_;
    Array2D<std::complex<double>> mx_dm_;
    Array2D<std::complex<double>> my_dm_;
    Array2D<std::complex<double>> mx_dm_back_;
    Array2D<std::complex<double>> my_dm_back_;

    std::size_t n_vortex_lres_;
    double vortex_win_diam_;
    double hres_sampling_;
    double vortex_dot_mask_diam_lamDc_;
    double vortex_dot_mask_diam_lamD_;
    double oversample_vortex_;
    double lres_sampling_;
    std::size_t lres_win_size_;
    std::size_t n_vortex_hres_;
    std::size_t hres_win_size_;

    Array2D<double> lres_window_;
    Array2D<double> hres_window_;
    Array2D<double> hres_dot_mask_;
    Array2D<std::complex<double>> vortex_lres_;
    Array2D<std::complex<double>> vortex_hres_;
    Array2D<std::complex<double>> windowed_vortex_lres_;
    Array2D<std::complex<double>> windowed_vortex_hres_;

    double camsci_rotation_ = 0.0;

    Array2D<std::complex<double>> matmul(const Array2D<std::complex<double>>& a,
                                         const Array2D<std::complex<double>>& b) const;
    Array2D<std::complex<double>> fft_backend(const Array2D<std::complex<double>>& arr) const;
    Array2D<std::complex<double>> ifft_backend(const Array2D<std::complex<double>>& arr) const;
    Array2D<std::complex<double>> ang_spec_backend(const Array2D<std::complex<double>>& wavefront,
                                                   double wavelength,
                                                   double distance,
                                                   double pixelscale) const;
    Array2D<std::complex<double>> mft_forward_backend(
        const Array2D<std::complex<double>>& wavefront,
        double npix,
        std::size_t npsf,
        double psf_pixelscale_lamD,
        char convention,
        const char* pp_centering,
        const char* fp_centering) const;
    Array2D<std::complex<double>> mft_reverse_backend(
        const Array2D<std::complex<double>>& fpwf,
        double psf_pixelscale_lamD,
        double npix,
        std::size_t N,
        char convention,
        const char* pp_centering,
        const char* fp_centering) const;

    ForwardResult forward_internal(const std::vector<double>& actuators,
                                   double wavelength,
                                   bool use_vortex) const;

    // Lazily-built device state for the GPU-resident forward path. Holds
    // device copies of the constant arrays and reusable scratch/plan
    // caches so a steady-state loop keeps all intermediates on the GPU.
    mutable std::shared_ptr<ControlModelGpuState> gpu_state_;

    friend Array2D<std::complex<double>> control_model_forward_gpu(
        const ControlModel& model,
        const std::vector<double>& actuators,
        double wavelength,
        bool use_vortex);

    friend double dm_val_and_grad(const ControlModel& model,
                                  const std::vector<double>& del_acts,
                                  const Array2D<double>& opd,
                                  std::vector<double>& grad_out);

    friend double val_and_grad(const std::vector<double>& del_acts,
                               const ControlModel& model,
                               const std::vector<double>& current_acts,
                               const Array2D<std::complex<double>>& e_ab,
                               const Array2D<std::complex<double>>& e_fp_nom,
                               const Array2D<std::uint8_t>& control_mask,
                               double wavelength,
                               double r_cond,
                               std::vector<double>& grad_out);

    friend double val_and_grad_bb(const std::vector<double>& del_acts,
                                  const ControlModel& model,
                                  const std::vector<double>& actuators,
                                  const std::vector<Array2D<std::complex<double>>>& e_abs,
                                  const Array2D<std::uint8_t>& control_mask,
                                  const std::vector<double>& waves,
                                  double r_cond,
                                  std::vector<double>& grad_out);

    bool use_gpu_ = false;
    // When the DM model is supplied externally (set_dm_model), set_device must
    // not recompute inf_fun_fft_ from the C++ influence function.
    bool dm_model_external_ = false;
};

// GPU-resident forward (device-only intermediates). Defined in
// control_models_cuda.cu; only referenced when LINA_USE_CUDA is on.
Array2D<std::complex<double>> control_model_forward_gpu(
    const ControlModel& model,
    const std::vector<double>& actuators,
    double wavelength,
    bool use_vortex);

// DM-surface fit objective + gradient (matches Python lina dm_val_and_grad):
// minimizes the residual of (OPD + 2 * DM_surface) over the beam aperture.
// `opd` is ndef x ndef; returns J and writes the gradient w.r.t. the masked
// actuators into grad_out (length nacts).
double dm_val_and_grad(const ControlModel& model,
                       const std::vector<double>& del_acts,
                       const Array2D<double>& opd,
                       std::vector<double>& grad_out);

// Solve for the flat-DM actuator vector that minimizes the residual OPD,
// using L-BFGS (libLBFGS). Returns the optimal del_acts (length nacts).
std::vector<double> solve_flat_command(const ControlModel& model,
                                       const Array2D<double>& opd,
                                       double tol = 1e-4,
                                       int max_iter = 0);

double val_and_grad(const std::vector<double>& del_acts,
                    const ControlModel& model,
                    const std::vector<double>& current_acts,
                    const Array2D<std::complex<double>>& e_ab,
                    const Array2D<std::complex<double>>& e_fp_nom,
                    const Array2D<std::uint8_t>& control_mask,
                    double wavelength,
                    double r_cond,
                    std::vector<double>& grad_out);

double val_and_grad_bb(const std::vector<double>& del_acts,
                       const ControlModel& model,
                       const std::vector<double>& actuators,
                       const std::vector<Array2D<std::complex<double>>>& e_abs,
                       const Array2D<std::uint8_t>& control_mask,
                       const std::vector<double>& waves,
                       double r_cond,
                       std::vector<double>& grad_out);

} // namespace lina
