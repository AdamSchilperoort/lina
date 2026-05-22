#pragma once

#include "lina/array.h"

#include <complex>
#include <vector>

namespace lina {

Array2D<double> gemm(const Array2D<double>& a,
                     const Array2D<double>& b,
                     bool transpose_a = false,
                     bool transpose_b = false);

std::vector<double> gemv(const Array2D<double>& a,
                         const std::vector<double>& x,
                         bool transpose_a = false);

struct SvdResult {
    Array2D<double> u;
    std::vector<double> s;
    Array2D<double> vt;
};

struct SvdResultF {
    Array2D<float> u;
    std::vector<float> s;
    Array2D<float> vt;
};

struct SvdCalibrationPoint {
    std::size_t m;
    std::size_t n;
    double gesvd_ms;
    double gesvdj_ms;
};

struct SvdCalibrationResult {
    std::size_t threshold_elems;
    int sm;
    std::vector<SvdCalibrationPoint> points;
};

SvdResult svd(const Array2D<double>& a);
SvdResultF svd_float(const Array2D<float>& a);
SvdResultF svd_float_cpu(const Array2D<float>& a);
SvdResultF svd_float_gpu(const Array2D<float>& a);

void set_num_threads(int nthreads);
int get_num_threads();

double benchmark_svd_float_gpu_e2e_ms(const Array2D<float>& a, int iters);
double benchmark_svd_float_gpu_kernel_ms(const Array2D<float>& a, int iters);
double benchmark_svd_float_gpu_xfer_ms(const Array2D<float>& a, int iters);
std::size_t svd_float_gpu_threshold_elems();
SvdCalibrationResult calibrate_svd_float_gpu_threshold();

} // namespace lina
