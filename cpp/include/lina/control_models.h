#pragma once

#include "lina/array.h"
#include "lina/efc.h"

#include <complex>
#include <optional>
#include <vector>

namespace lina {

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

    Array2D<std::complex<double>> forward(const std::vector<double>& actuators,
                                          double wavelength,
                                          bool use_vortex) override;

    const Array2D<std::uint8_t>& dm_mask() const { return dm_mask_; }

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

    ForwardResult forward_internal(const std::vector<double>& actuators,
                                   double wavelength,
                                   bool use_vortex) const;

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
};

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
