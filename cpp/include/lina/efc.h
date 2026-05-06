#pragma once

#include "lina/array.h"
#include "lina/coro_utils.h"
#include "lina/stream.h"

#include <complex>
#include <cstddef>
#include <vector>

namespace lina {

struct PwpResult {
    Array2D<std::complex<double>> field;
    std::vector<std::complex<double>> field_vec;
};

class PwpEstimator {
public:
    virtual ~PwpEstimator() = default;
    virtual PwpResult run(Stream2D& camsci,
                          Stream2D& dm,
                          const ImParams& im_params,
                          const ImParams& ref_params) = 0;
    virtual PwpResult run_with_jacobian(Stream2D& camsci,
                                        Stream2D& dm,
                                        const ImParams& im_params,
                                        const ImParams& ref_params) = 0;
};

class EfcModel {
public:
    virtual ~EfcModel() = default;
    virtual std::size_t nacts() const = 0;
    virtual double wavelength() const = 0;
    virtual Array2D<std::complex<double>> forward(const std::vector<double>& actuators,
                                                  double wavelength,
                                                  bool use_vortex) = 0;
};

struct EfcData {
    std::vector<Array2D<double>> images;
    std::vector<double> contrasts;
    std::vector<Array2D<std::complex<double>>> efields;
    std::vector<Array2D<double>> commands;
    std::vector<Array2D<double>> del_commands;
};

Array2D<double> compute_jacobian(EfcModel& model,
                                 const Array2D<std::uint8_t>& control_mask,
                                 double amp = 1e-9,
                                 const std::vector<double>* current_acts = nullptr);

std::pair<Array2D<double>, std::vector<Array2D<double>>> compute_jacobian_bb(
    std::vector<EfcModel*>& models,
    const Array2D<std::uint8_t>& control_mask,
    double amp = 1e-9,
    const std::vector<double>* current_acts = nullptr);

void run(EfcData& efc_data,
         Stream2D& camsci,
         Stream2D& dm,
         const ImParams& im_params,
         const ImParams& ref_params,
         std::size_t nframes,
         const Array2D<double>& dark_im,
         const Array2D<std::uint8_t>& control_mask,
         const Array2D<std::uint8_t>& dm_mask,
         const Array2D<double>& control_matrix,
         PwpEstimator& pwp,
         std::size_t nitr = 3,
         double gain = 1.0,
         double leakage = 0.0,
         double delay_s = 0.05,
         double dm_scale = 1e-6);

void run_bb(EfcData& efc_data,
            Stream2D& camsci,
            Stream2D& dm,
            const ImParams& im_params,
            const ImParams& ref_params,
            std::size_t nframes,
            const Array2D<double>& dark_im,
            const Array2D<std::uint8_t>& control_mask,
            const Array2D<std::uint8_t>& dm_mask,
            const Array2D<double>& control_matrix,
            PwpEstimator& pwp,
            std::size_t nwaves,
            std::size_t nitr = 3,
            double gain = 1.0,
            double leakage = 0.0,
            double dm_delay_s = 0.05,
            double filter_delay_s = 1.0,
            double dm_scale = 1e-6);

} // namespace lina
