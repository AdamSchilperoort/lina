#include "lina/linalg.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

#ifdef LINA_USE_OPENBLAS
#include <cblas.h>
extern "C" {
int openblas_get_num_threads(void);
void openblas_set_num_threads(int);
}
#endif

#ifdef LINA_USE_LAPACKE
#include <lapacke.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef LINA_USE_EIGEN_SVD
#include <Eigen/Dense>
#endif

#ifdef LINA_USE_CUDA
namespace lina {
SvdResultF svd_float_cuda(const Array2D<float>& a);
}
#endif

namespace lina {

Array2D<double> gemm(const Array2D<double>& a,
                     const Array2D<double>& b,
                     bool transpose_a,
                     bool transpose_b) {
    const std::size_t a_rows = transpose_a ? a.cols() : a.rows();
    const std::size_t a_cols = transpose_a ? a.rows() : a.cols();
    const std::size_t b_rows = transpose_b ? b.cols() : b.rows();
    const std::size_t b_cols = transpose_b ? b.rows() : b.cols();

    if (a_cols != b_rows) {
        throw std::invalid_argument("gemm dimension mismatch");
    }

    Array2D<double> out(a_rows, b_cols, 0.0);

#ifdef LINA_USE_OPENBLAS
    const CBLAS_TRANSPOSE ta = transpose_a ? CblasTrans : CblasNoTrans;
    const CBLAS_TRANSPOSE tb = transpose_b ? CblasTrans : CblasNoTrans;
    const int m = static_cast<int>(a_rows);
    const int n = static_cast<int>(b_cols);
    const int k = static_cast<int>(a_cols);
    const int lda = static_cast<int>(a.cols());
    const int ldb = static_cast<int>(b.cols());
    const int ldc = static_cast<int>(out.cols());

    cblas_dgemm(CblasRowMajor, ta, tb, m, n, k,
                1.0, a.data(), lda, b.data(), ldb, 0.0, out.data(), ldc);
#else
    for (std::size_t i = 0; i < a_rows; ++i) {
        for (std::size_t j = 0; j < b_cols; ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < a_cols; ++k) {
                const double av = transpose_a ? a(k, i) : a(i, k);
                const double bv = transpose_b ? b(j, k) : b(k, j);
                sum += av * bv;
            }
            out(i, j) = sum;
        }
    }
#endif

    return out;
}

// Internal: hand-rolled gemv via the safe Array2D accessor. Used as a
// fallback when CBLAS is disabled or as a runtime escape hatch on
// platforms where the system BLAS produces wrong dgemv results
// (observed: some NVIDIA-Jetson aarch64 OpenBLAS builds).
static std::vector<double> gemv_naive(const Array2D<double>& a,
                                      const std::vector<double>& x,
                                      bool transpose_a) {
    const std::size_t rows = transpose_a ? a.cols() : a.rows();
    const std::size_t cols = transpose_a ? a.rows() : a.cols();
    std::vector<double> out(rows, 0.0);
    for (std::size_t i = 0; i < rows; ++i) {
        double sum = 0.0;
        for (std::size_t j = 0; j < cols; ++j) {
            const double av = transpose_a ? a(j, i) : a(i, j);
            sum += av * x[j];
        }
        out[i] = sum;
    }
    return out;
}

std::vector<double> gemv(const Array2D<double>& a,
                         const std::vector<double>& x,
                         bool transpose_a) {
    const std::size_t cols = transpose_a ? a.rows() : a.cols();
    if (cols != x.size()) {
        throw std::invalid_argument("gemv dimension mismatch");
    }

#ifdef LINA_USE_OPENBLAS
    // Runtime opt-out for platforms whose CBLAS dgemv is buggy. Set
    // LINA_GEMV_NAIVE=1 in the environment to force the hand-rolled
    // loop and bypass the system BLAS.
    static const bool force_naive = [] {
        const char* env = std::getenv("LINA_GEMV_NAIVE");
        return env && env[0] && env[0] != '0';
    }();
    if (force_naive) {
        return gemv_naive(a, x, transpose_a);
    }

    const std::size_t rows = transpose_a ? a.cols() : a.rows();
    std::vector<double> out(rows, 0.0);
    const CBLAS_TRANSPOSE ta = transpose_a ? CblasTrans : CblasNoTrans;
    const int m = static_cast<int>(a.rows());
    const int n = static_cast<int>(a.cols());
    const int lda = static_cast<int>(a.cols());
    cblas_dgemv(CblasRowMajor, ta, m, n, 1.0, a.data(), lda,
                x.data(), 1, 0.0, out.data(), 1);
    return out;
#else
    return gemv_naive(a, x, transpose_a);
#endif
}

SvdResult svd(const Array2D<double>& a) {
    const std::size_t m = a.rows();
    const std::size_t n = a.cols();
    Array2D<double> u(m, m, 0.0);
    Array2D<double> vt(n, n, 0.0);
    std::vector<double> s(std::min(m, n), 0.0);

#if defined(LINA_FORCE_LAPACKE) && defined(LINA_USE_LAPACKE)
    std::vector<double> a_copy(a.data(), a.data() + a.size());
    const lapack_int m_i = static_cast<lapack_int>(m);
    const lapack_int n_i = static_cast<lapack_int>(n);
    const lapack_int lda = static_cast<lapack_int>(n);
    const lapack_int ldu = static_cast<lapack_int>(m);
    const lapack_int ldvt = static_cast<lapack_int>(n);

    const lapack_int info = LAPACKE_dgesdd(
        LAPACK_ROW_MAJOR, 'A', m_i, n_i,
        a_copy.data(), lda, s.data(), u.data(), ldu, vt.data(), ldvt);
    if (info != 0) {
        throw std::runtime_error("LAPACKE_dgesdd failed");
    }
#elif defined(LINA_USE_EIGEN_SVD)
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat(
        a.data(), static_cast<int>(m), static_cast<int>(n));
    Eigen::JacobiSVD<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> svd(
        mat, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::MatrixXd u_e = svd.matrixU();
    Eigen::MatrixXd v_e = svd.matrixV();
    Eigen::VectorXd s_e = svd.singularValues();

    for (std::size_t i = 0; i < s.size(); ++i) {
        s[i] = s_e(static_cast<int>(i));
    }
    for (std::size_t r = 0; r < m; ++r) {
        for (std::size_t c = 0; c < m; ++c) {
            u(r, c) = u_e(static_cast<int>(r), static_cast<int>(c));
        }
    }
    for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t c = 0; c < n; ++c) {
            vt(r, c) = v_e(static_cast<int>(c), static_cast<int>(r));
        }
    }
#elif defined(LINA_USE_LAPACKE)
    std::vector<double> a_copy(a.data(), a.data() + a.size());
    const lapack_int m_i = static_cast<lapack_int>(m);
    const lapack_int n_i = static_cast<lapack_int>(n);
    const lapack_int lda = static_cast<lapack_int>(n);
    const lapack_int ldu = static_cast<lapack_int>(m);
    const lapack_int ldvt = static_cast<lapack_int>(n);

    const lapack_int info = LAPACKE_dgesdd(
        LAPACK_ROW_MAJOR, 'A', m_i, n_i,
        a_copy.data(), lda, s.data(), u.data(), ldu, vt.data(), ldvt);
    if (info != 0) {
        throw std::runtime_error("LAPACKE_dgesdd failed");
    }
#else
    throw std::runtime_error("SVD unavailable: build with LAPACKE or Eigen");
#endif

    return {u, s, vt};
}

SvdResultF svd_float_cpu(const Array2D<float>& a) {
    const std::size_t m = a.rows();
    const std::size_t n = a.cols();
    Array2D<float> u(m, m, 0.0f);
    Array2D<float> vt(n, n, 0.0f);
    std::vector<float> s(std::min(m, n), 0.0f);

#ifdef LINA_USE_LAPACKE
    std::vector<float> a_copy(a.data(), a.data() + a.size());
    const lapack_int m_i = static_cast<lapack_int>(m);
    const lapack_int n_i = static_cast<lapack_int>(n);
    const lapack_int lda = static_cast<lapack_int>(n);
    const lapack_int ldu = static_cast<lapack_int>(m);
    const lapack_int ldvt = static_cast<lapack_int>(n);
    const lapack_int info = LAPACKE_sgesdd(
        LAPACK_ROW_MAJOR, 'A', m_i, n_i,
        a_copy.data(), lda, s.data(), u.data(), ldu, vt.data(), ldvt);
    if (info != 0) {
        throw std::runtime_error("LAPACKE_sgesdd failed");
    }
#elif defined(LINA_USE_EIGEN_SVD)
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat(
        a.data(), static_cast<int>(m), static_cast<int>(n));
    Eigen::JacobiSVD<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> svd(
        mat, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::MatrixXf u_e = svd.matrixU();
    Eigen::MatrixXf v_e = svd.matrixV();
    Eigen::VectorXf s_e = svd.singularValues();

    for (std::size_t i = 0; i < s.size(); ++i) {
        s[i] = s_e(static_cast<int>(i));
    }
    for (std::size_t r = 0; r < m; ++r) {
        for (std::size_t c = 0; c < m; ++c) {
            u(r, c) = u_e(static_cast<int>(r), static_cast<int>(c));
        }
    }
    for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t c = 0; c < n; ++c) {
            vt(r, c) = v_e(static_cast<int>(c), static_cast<int>(r));
        }
    }
#else
    throw std::runtime_error("Float SVD unavailable: build with LAPACKE or Eigen");
#endif

    return {u, s, vt};
}

SvdResultF svd_float_gpu(const Array2D<float>& a) {
#ifdef LINA_USE_CUDA
    return svd_float_cuda(a);
#else
    throw std::runtime_error("CUDA SVD unavailable: build with LINA_USE_CUDA=ON");
#endif
}

SvdResultF svd_float(const Array2D<float>& a) {
#ifdef LINA_USE_CUDA
    return svd_float_cuda(a);
#else
    return svd_float_cpu(a);
#endif
}

void set_num_threads(int nthreads) {
    if (nthreads < 1) nthreads = 1;
#ifdef LINA_USE_OPENBLAS
    openblas_set_num_threads(nthreads);
#endif
#ifdef _OPENMP
    omp_set_num_threads(nthreads);
#endif
}

int get_num_threads() {
#ifdef LINA_USE_OPENBLAS
    return openblas_get_num_threads();
#else
    return 1;
#endif
}

} // namespace lina
