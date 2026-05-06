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

SvdResult svd(const Array2D<double>& a);
SvdResultF svd_float(const Array2D<float>& a);
SvdResultF svd_float_cpu(const Array2D<float>& a);
SvdResultF svd_float_gpu(const Array2D<float>& a);

} // namespace lina
