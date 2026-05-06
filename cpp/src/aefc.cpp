#include "lina/aefc.h"

#include "lina/coro_utils.h"

#include <chrono>
#include <cmath>
#include <thread>

namespace lina {

GradientDescentOptimizer::GradientDescentOptimizer(std::size_t max_iters, double step)
    : max_iters_(max_iters), step_(step) {}

std::vector<double> GradientDescentOptimizer::minimize(
    const std::vector<double>& x0,
    double tol,
    const AefcObjective& objective,
    const EfcModel& model,
    const AefcVars& vars,
    bool verbose) {
    std::vector<double> x = x0;
    std::vector<double> grad(x0.size(), 0.0);
    double fval = 0.0;

    for (std::size_t iter = 0; iter < max_iters_; ++iter) {
        objective(x, fval, grad, model, vars, verbose);
        double max_grad = 0.0;
        for (double g : grad) {
            max_grad = std::max(max_grad, std::abs(g));
        }
        if (max_grad < tol) {
            break;
        }
        for (std::size_t i = 0; i < x.size(); ++i) {
            x[i] -= step_ * grad[i];
        }
    }
    return x;
}

#ifdef LINA_USE_LBFGS
#include <lbfgs.h>

namespace {

struct LbfgsContext {
    const AefcObjective* objective = nullptr;
    const EfcModel* model = nullptr;
    const AefcVars* vars = nullptr;
    bool verbose = false;
};

lbfgsfloatval_t evaluate_cb(void* instance,
                            const lbfgsfloatval_t* x,
                            lbfgsfloatval_t* g,
                            const int n,
                            const lbfgsfloatval_t /*step*/) {
    auto* ctx = static_cast<LbfgsContext*>(instance);
    std::vector<double> xvec(n, 0.0);
    for (int i = 0; i < n; ++i) {
        xvec[i] = x[i];
    }
    std::vector<double> grad(n, 0.0);
    double fval = 0.0;
    (*ctx->objective)(xvec, fval, grad, *ctx->model, *ctx->vars, ctx->verbose);
    for (int i = 0; i < n; ++i) {
        g[i] = grad[i];
    }
    return fval;
}

int progress_cb(void* /*instance*/,
                const lbfgsfloatval_t* /*x*/,
                const lbfgsfloatval_t* /*g*/,
                const lbfgsfloatval_t /*fx*/,
                const lbfgsfloatval_t /*xnorm*/,
                const lbfgsfloatval_t /*gnorm*/,
                const lbfgsfloatval_t /*step*/,
                int /*n*/,
                int /*k*/,
                int /*ls*/) {
    return 0;
}

} // namespace

LbfgsOptimizer::LbfgsOptimizer() = default;

std::vector<double> LbfgsOptimizer::minimize(const std::vector<double>& x0,
                                             double tol,
                                             const AefcObjective& objective,
                                             const EfcModel& model,
                                             const AefcVars& vars,
                                             bool verbose) {
    const int n = static_cast<int>(x0.size());
    std::vector<lbfgsfloatval_t> x(n, 0.0);
    for (int i = 0; i < n; ++i) {
        x[i] = x0[i];
    }

    lbfgs_parameter_t param;
    lbfgs_parameter_init(&param);
    param.epsilon = tol;

    LbfgsContext ctx;
    ctx.objective = &objective;
    ctx.model = &model;
    ctx.vars = &vars;
    ctx.verbose = verbose;

    lbfgsfloatval_t fx = 0.0;
    const int ret = lbfgs(n, x.data(), &fx, evaluate_cb, progress_cb, &ctx, &param);
    if (ret < 0) {
        throw std::runtime_error("libLBFGS failed");
    }

    std::vector<double> out(n, 0.0);
    for (int i = 0; i < n; ++i) {
        out[i] = x[i];
    }
    return out;
}
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
         std::size_t nitr,
         double reg_cond,
         double bfgs_tol,
         double gain,
         double leakage,
         double delay_s,
         double dm_scale,
         bool verbose) {
    Array2D<double> del_command(dm.rows(), dm.cols(), 0.0);

    for (std::size_t i = 0; i < nitr; ++i) {
        Array2D<double> current_command = dm.grab_latest();
        for (std::size_t j = 0; j < current_command.size(); ++j) {
            current_command.data()[j] *= dm_scale;
        }

        std::vector<double> current_acts;
        current_acts.reserve(dm_mask.size());
        for (std::size_t idx = 0; idx < dm_mask.size(); ++idx) {
            if (dm_mask.data()[idx]) {
                current_acts.push_back(current_command.data()[idx]);
            }
        }

        const auto e_fp_nom = model.forward(current_acts, model.wavelength(), true);
        const PwpResult pwp_result = pwp.run(camsci, dm, im_params, ref_params);

        AefcVars vars;
        vars.e_ab = pwp_result.field;
        vars.current_acts = current_acts;
        vars.e_fp_nom = e_fp_nom;
        vars.control_mask = control_mask;
        vars.wavelength = model.wavelength();
        vars.r_cond = reg_cond;

        std::vector<double> x0(model.nacts(), 0.0);
        const std::vector<double> del_acts = optimizer.minimize(
            x0, bfgs_tol, objective, model, vars, verbose);

        del_command.fill(0.0);

        std::size_t act_idx = 0;
        for (std::size_t idx = 0; idx < dm_mask.size(); ++idx) {
            if (dm_mask.data()[idx]) {
                del_command.data()[idx] = gain * del_acts[act_idx++];
            }
        }

        Array2D<double> total_command(dm.rows(), dm.cols(), 0.0);
        for (std::size_t j = 0; j < total_command.size(); ++j) {
            total_command.data()[j] = (1.0 - leakage) * current_command.data()[j] +
                                      del_command.data()[j];
            total_command.data()[j] /= dm_scale;
        }
        dm.write(total_command);
        std::this_thread::sleep_for(std::chrono::duration<double>(delay_s));

        const Array2D<double> coro_im = camsci.grab_mean(ncamsci);
        const Array2D<double> coro_im_ni = normalize_coro_im(coro_im, im_params, ref_params, dark_im);
        const ContrastResult contrast = compute_contrast(coro_im_ni, control_mask);

        efc_data.raw_images.push_back(coro_im);
        efc_data.dark_images.push_back(dark_im);
        efc_data.ni_images.push_back(coro_im_ni);
        efc_data.contrasts.push_back(contrast.contrast);
        efc_data.efields.push_back(pwp_result.field);
        efc_data.commands.push_back(total_command);
        efc_data.del_commands.push_back(del_command);
        efc_data.bfgs_tols.push_back(bfgs_tol);
        efc_data.reg_conds.push_back(reg_cond);
    }
}

} // namespace lina
