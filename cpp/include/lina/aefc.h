#pragma once

#include "lina/array.h"
#include "lina/coro_utils.h"
#include "lina/efc.h"
#include "lina/stream.h"

#include <functional>
#include <vector>

namespace lina {

struct AefcData {
    std::vector<Array2D<double>> raw_images;
    std::vector<Array2D<double>> dark_images;
    std::vector<Array2D<double>> ni_images;
    std::vector<double> contrasts;
    std::vector<Array2D<std::complex<double>>> efields;
    std::vector<Array2D<double>> commands;
    std::vector<Array2D<double>> del_commands;
    std::vector<double> bfgs_tols;
    std::vector<double> reg_conds;
};

struct AefcVars {
    Array2D<std::complex<double>> e_ab;
    std::vector<double> current_acts;
    Array2D<std::complex<double>> e_fp_nom;
    Array2D<std::uint8_t> control_mask;
    double wavelength = 0.0;
    double r_cond = 1e-2;
};

using AefcObjective = std::function<void(const std::vector<double>& x,
                                         double& fval,
                                         std::vector<double>& grad,
                                         const EfcModel& model,
                                         const AefcVars& vars,
                                         bool verbose)>;

class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual std::vector<double> minimize(const std::vector<double>& x0,
                                         double tol,
                                         const AefcObjective& objective,
                                         const EfcModel& model,
                                         const AefcVars& vars,
                                         bool verbose) = 0;
};

class GradientDescentOptimizer : public Optimizer {
public:
    GradientDescentOptimizer(std::size_t max_iters = 200, double step = 1e-2);
    std::vector<double> minimize(const std::vector<double>& x0,
                                 double tol,
                                 const AefcObjective& objective,
                                 const EfcModel& model,
                                 const AefcVars& vars,
                                 bool verbose) override;

private:
    std::size_t max_iters_;
    double step_;
};

#ifdef LINA_USE_LBFGS
class LbfgsOptimizer : public Optimizer {
public:
    LbfgsOptimizer();
    std::vector<double> minimize(const std::vector<double>& x0,
                                 double tol,
                                 const AefcObjective& objective,
                                 const EfcModel& model,
                                 const AefcVars& vars,
                                 bool verbose) override;
};
#endif

void run(AefcData& efc_data,
         Stream2D& camsci,
         Stream2D& dm,
         const ImParams& im_params,
         const ImParams& ref_params,
         std::size_t ncamsci,
         const Array2D<double>& dark_im,
         EfcModel& model,
         const AefcObjective& objective,
         const Array2D<std::uint8_t>& control_mask,
         const Array2D<std::uint8_t>& dm_mask,
         PwpEstimator& pwp,
         Optimizer& optimizer,
         std::size_t nitr = 3,
         double reg_cond = 1e-2,
         double bfgs_tol = 1e-3,
         double gain = 1.0,
         double leakage = 0.0,
         double delay_s = 0.05,
         double dm_scale = 1e-6,
         bool verbose = false);

} // namespace lina
