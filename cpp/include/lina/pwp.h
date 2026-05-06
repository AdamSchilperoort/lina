#pragma once

#include "lina/array.h"
#include "lina/coro_utils.h"
#include "lina/efc.h"
#include "lina/stream.h"

#include <optional>
#include <vector>

namespace lina {

class PwpSolver : public PwpEstimator {
public:
    PwpSolver(const Array2D<double>& probes,
              const Array2D<std::uint8_t>& control_mask,
              const Array2D<std::uint8_t>& dm_mask,
              std::size_t nframes,
              double probe_amp,
              double reg_cond = 1e-3,
              double gain = 1.0,
              double dm_scale = 1e-6);

    void set_model(EfcModel* model, double wavelength);
    void set_jacobian(const Array2D<double>& jacobian);
    void set_fp_shift(double x_shift, double y_shift);
    void set_e_fp_nom(const Array2D<std::complex<double>>& e_fp_nom);

    PwpResult run(Stream2D& camsci,
                  Stream2D& dm,
                  const ImParams& im_params,
                  const ImParams& ref_params) override;

    PwpResult run_with_jacobian(Stream2D& camsci,
                                Stream2D& dm,
                                const ImParams& im_params,
                                const ImParams& ref_params) override;

private:
    Array2D<double> probes_;
    Array2D<std::uint8_t> control_mask_;
    Array2D<std::uint8_t> dm_mask_;
    std::size_t nframes_ = 1;
    double probe_amp_;
    double reg_cond_;
    double gain_;
    double dm_scale_;

    std::optional<std::pair<double, double>> fp_shift_;
    std::optional<Array2D<double>> jacobian_;
    std::optional<Array2D<std::complex<double>>> e_fp_nom_;
    EfcModel* model_ = nullptr;
    double wavelength_ = 0.0;
};

} // namespace lina
