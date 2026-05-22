#include "lina/lina.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>

namespace {

std::vector<double> load_doubles(const std::string& path, std::size_t count) {
    std::vector<double> data(count, 0.0);
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open data file: " + path);
    }
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(count * sizeof(double)));
    if (!in) {
        throw std::runtime_error("Failed to read double data from: " + path);
    }
    return data;
}

std::vector<float> load_floats(const std::string& path, std::size_t count) {
    std::vector<float> data(count, 0.0f);
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open data file: " + path);
    }
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(count * sizeof(float)));
    if (!in) {
        throw std::runtime_error("Failed to read float data from: " + path);
    }
    return data;
}

auto build_fft_input(std::size_t n, const std::string& data_path) {
    lina::Array2D<std::complex<double>> input(n, n, {0.0, 0.0});
    if (!data_path.empty()) {
        auto vals = load_doubles(data_path, n * n);
        for (std::size_t i = 0; i < input.size(); ++i) {
            input.data()[i] = vals[i];
        }
    } else {
        for (std::size_t i = 0; i < input.size(); ++i) {
            input.data()[i] = static_cast<double>(i);
        }
    }
    return input;
}

void print_fft() {
    auto input = build_fft_input(4, "");
    auto out = lina::fft_cpu(input);
    std::cout << "FFT 4 4\n";
    for (std::size_t i = 0; i < out.size(); ++i) {
        std::cout << out.data()[i].real() << " " << out.data()[i].imag() << "\n";
    }
}

auto compute_svd() {
    lina::Array2D<double> mat(3, 2, 0.0);
    double vals[] = {3.0, 1.0, 0.0, -1.0, 2.0, 4.0};
    for (std::size_t i = 0; i < mat.size(); ++i) {
        mat.data()[i] = vals[i];
    }
    return lina::svd(mat);
}

auto compute_svd_float() {
    lina::Array2D<float> mat(3, 2, 0.0f);
    float vals[] = {3.0f, 1.0f, 0.0f, -1.0f, 2.0f, 4.0f};
    for (std::size_t i = 0; i < mat.size(); ++i) {
        mat.data()[i] = vals[i];
    }
    return lina::svd_float(mat);
}

auto build_svd_input(std::size_t m, std::size_t n, const std::string& data_path) {
    lina::Array2D<float> mat(m, n, 0.0f);
    if (!data_path.empty()) {
        auto vals = load_floats(data_path, m * n);
        for (std::size_t i = 0; i < mat.size(); ++i) {
            mat.data()[i] = vals[i];
        }
    } else {
        for (std::size_t i = 0; i < mat.size(); ++i) {
            mat.data()[i] = static_cast<float>(i % 1024) / 1024.0f;
        }
    }
    return mat;
}

void print_svd() {
    try {
        auto res = compute_svd();
        std::cout << "SVD " << res.s.size() << "\n";
        for (double v : res.s) {
            std::cout << v << "\n";
        }
    } catch (const std::exception&) {
        std::cout << "SVD_UNAVAILABLE\n";
        std::exit(2);
    }
}

struct DummyDmStream : public lina::Stream2D {
    lina::Array2D<double> current;
    explicit DummyDmStream(std::size_t n) : current(n, n, 0.0) {}

    lina::Array2D<double> grab_latest() override { return current; }
    lina::Array2D<double> grab_mean(std::size_t) override { return current; }
    void write(const lina::Array2D<double>& data) override { current = data; }
    std::size_t rows() const override { return current.rows(); }
    std::size_t cols() const override { return current.cols(); }
};

struct DummyCamStream : public lina::Stream2D {
    DummyDmStream* dm = nullptr;
    lina::Array2D<double> base;
    double scale = 1.0;

    DummyCamStream(std::size_t n, DummyDmStream* dm_stream, double scale_val)
        : dm(dm_stream), base(n, n, 1.0), scale(scale_val) {}

    lina::Array2D<double> grab_latest() override { return grab_mean(1); }
    lina::Array2D<double> grab_mean(std::size_t) override {
        double sum = 0.0;
        for (std::size_t i = 0; i < dm->current.size(); ++i) {
            sum += dm->current.data()[i];
        }
        lina::Array2D<double> out = base;
        for (std::size_t i = 0; i < out.size(); ++i) {
            out.data()[i] += scale * sum;
        }
        return out;
    }
    void write(const lina::Array2D<double>&) override {}
    std::size_t rows() const override { return base.rows(); }
    std::size_t cols() const override { return base.cols(); }
};

struct DummyPwp : public lina::PwpEstimator {
    lina::Array2D<std::uint8_t> control_mask;
    std::vector<std::size_t> mask_idx;
    std::vector<std::complex<double>> field_vec;

    DummyPwp() : control_mask(4, 4, 0) {
        control_mask(1, 1) = 1;
        control_mask(1, 2) = 1;
        for (std::size_t i = 0; i < control_mask.size(); ++i) {
            if (control_mask.data()[i]) {
                mask_idx.push_back(i);
            }
        }
        field_vec = { {1.0, 2.0}, {-0.5, 0.25} };
    }

    lina::PwpResult run(lina::Stream2D&,
                        lina::Stream2D&,
                        const lina::ImParams&,
                        const lina::ImParams&) override {
        lina::Array2D<std::complex<double>> field(4, 4, {0.0, 0.0});
        for (std::size_t i = 0; i < mask_idx.size(); ++i) {
            field.data()[mask_idx[i]] = field_vec[i];
        }
        return {field, field_vec};
    }

    lina::PwpResult run_with_jacobian(lina::Stream2D& camsci,
                                      lina::Stream2D& dm,
                                      const lina::ImParams& im_params,
                                      const lina::ImParams& ref_params) override {
        return run(camsci, dm, im_params, ref_params);
    }
};

auto compute_efc() {
    DummyDmStream dm(2);
    DummyCamStream camsci(4, &dm, 0.1);
    DummyPwp pwp;

    lina::ImParams im_params{1.0, 0.0, 0.0, 1.0};
    lina::ImParams ref_params{1.0, 0.0, 0.0, 1.0};
    lina::Array2D<double> dark_im(4, 4, 0.0);

    lina::Array2D<std::uint8_t> control_mask(4, 4, 0);
    control_mask(1, 1) = 1;
    control_mask(1, 2) = 1;

    lina::Array2D<std::uint8_t> dm_mask(2, 2, 1);

    lina::Array2D<double> control_matrix(4, 4, 0.0);
    for (std::size_t i = 0; i < 4; ++i) {
        control_matrix(i, i) = 1.0;
    }

    lina::EfcData data;
    lina::run(data, camsci, dm, im_params, ref_params, 1, dark_im,
              control_mask, dm_mask, control_matrix, pwp, 1, 1.0, 0.0, 0.0, 1e-6);
    return data;
}

void print_efc() {
    auto data = compute_efc();
    std::cout << "EFC_CONTRAST " << data.contrasts[0] << "\n";
    std::cout << "EFC_COMMAND 2 2\n";
    for (std::size_t i = 0; i < data.commands[0].size(); ++i) {
        std::cout << data.commands[0].data()[i] << "\n";
    }
}

auto compute_iefc() {
    DummyDmStream dm(2);
    DummyCamStream camsci(4, &dm, 0.1);

    lina::ImParams im_params{1.0, 0.0, 0.0, 1.0};
    lina::ImParams ref_params{1.0, 0.0, 0.0, 1.0};
    lina::Array2D<double> dark_im(4, 4, 0.0);

    lina::Array2D<std::uint8_t> control_mask(4, 4, 0);
    control_mask(1, 1) = 1;
    control_mask(1, 2) = 1;

    lina::Array2D<double> probe_modes(2, 4, 0.0);
    probe_modes(0, 0) = 1.0;
    probe_modes(1, 1) = -1.0;

    lina::Array2D<double> calib_modes(4, 4, 0.0);
    for (std::size_t i = 0; i < 4; ++i) {
        calib_modes(i, i) = 1.0;
    }

    lina::Array2D<double> control_matrix(4, 4, 0.0);
    for (std::size_t i = 0; i < 4; ++i) {
        control_matrix(i, i) = 1.0;
    }

    lina::IefcData data;
    lina::run(data, camsci, 1, dm, im_params, ref_params, dark_im,
              control_matrix, 0.5, probe_modes, calib_modes, control_mask,
              0.0, 1, 0.75, 0.0, 1e-6);
    return data;
}

void print_iefc() {
    auto data = compute_iefc();
    std::cout << "IEFC_CONTRAST " << data.contrasts[0] << "\n";
    std::cout << "IEFC_COMMAND 2 2\n";
    for (std::size_t i = 0; i < data.commands[0].size(); ++i) {
        std::cout << data.commands[0].data()[i] << "\n";
    }
}

void print_shmim() {
    lina::ShmimStream stream;
    stream.create("lina_test_shmim", 2, 2);
    lina::Array2D<double> data(2, 2, 0.0);
    data(0, 0) = 1.0;
    data(0, 1) = 2.0;
    data(1, 0) = 3.0;
    data(1, 1) = 4.0;
    stream.write(data);
    auto out = stream.grab_latest();
    std::cout << "SHMIM 2 2\n";
    for (std::size_t i = 0; i < out.size(); ++i) {
        std::cout << out.data()[i] << "\n";
    }
    stream.close();
}

void print_vortex() {
    auto mask = lina::make_vortex_phase_mask(4);
    std::cout << "VORTEX 4 4\n";
    for (std::size_t i = 0; i < mask.size(); ++i) {
        std::cout << mask.data()[i].real() << " " << mask.data()[i].imag() << "\n";
    }
}

void print_annular() {
    auto mask = lina::create_annular_mask(4, 1.0, 0.5, 2.0, -1e9, 0.0, 0.0, 0.0);
    std::cout << "ANNULAR 4 4\n";
    for (std::size_t i = 0; i < mask.size(); ++i) {
        std::cout << static_cast<int>(mask.data()[i]) << "\n";
    }
}

void print_dm_mask() {
    auto mask = lina::create_mask(4);
    std::cout << "DM_MASK 4 4\n";
    for (std::size_t i = 0; i < mask.size(); ++i) {
        std::cout << static_cast<int>(mask.data()[i]) << "\n";
    }
}

template <typename Fn>
void print_bench(const std::string& name, int iters, Fn fn) {
    using clock = std::chrono::high_resolution_clock;
    const auto start = clock::now();
    for (int i = 0; i < iters; ++i) {
        fn();
    }
    const auto end = clock::now();
    const double ms = std::chrono::duration<double, std::milli>(end - start).count();
    const double avg_ms = ms / std::max(1, iters);
    std::cout << name << " " << iters << " " << avg_ms << "\n";
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: lina_runner <fft|svd|efc|iefc|shmim|bench_svd_autocal>\n";
        return 1;
    }
    const std::string cmd = argv[1];
    if (cmd == "fft") {
        print_fft();
    } else if (cmd == "svd") {
        print_svd();
    } else if (cmd == "efc") {
        print_efc();
    } else if (cmd == "iefc") {
        print_iefc();
    } else if (cmd == "shmim") {
        print_shmim();
    } else if (cmd == "vortex") {
        print_vortex();
    } else if (cmd == "annular") {
        print_annular();
    } else if (cmd == "dm_mask") {
        print_dm_mask();
    } else if (cmd == "bench_fft") {
        const char* n_env = std::getenv("LINA_BENCH_FFT_N");
        const char* it_env = std::getenv("LINA_BENCH_FFT_ITERS");
        const char* data_env = std::getenv("LINA_BENCH_DATA_PATH");
        const std::size_t n = n_env ? static_cast<std::size_t>(std::stoul(n_env)) : 4;
        const int iters = it_env ? std::stoi(it_env) : 200;
        const std::string data_path = data_env ? data_env : "";
        const auto input = build_fft_input(n, data_path);
        print_bench("BENCH_FFT " + std::to_string(n) + " " + std::to_string(iters), iters,
                    [&input] { lina::fft_cpu(input); });
    } else if (cmd == "bench_fft_cpu") {
        const char* n_env = std::getenv("LINA_BENCH_FFT_N");
        const char* it_env = std::getenv("LINA_BENCH_FFT_ITERS");
        const char* data_env = std::getenv("LINA_BENCH_DATA_PATH");
        const std::size_t n = n_env ? static_cast<std::size_t>(std::stoul(n_env)) : 4;
        const int iters = it_env ? std::stoi(it_env) : 200;
        const std::string data_path = data_env ? data_env : "";
        const auto input = build_fft_input(n, data_path);
        lina::fft_cpu(input);
        lina::fft_cpu(input);
        print_bench("BENCH_FFT_CPU " + std::to_string(n) + " " + std::to_string(iters), iters,
                    [&input] { lina::fft_cpu(input); });
    } else if (cmd == "bench_fft_gpu") {
#ifdef LINA_USE_CUDA
        const char* n_env = std::getenv("LINA_BENCH_FFT_N");
        const char* it_env = std::getenv("LINA_BENCH_FFT_ITERS");
        const char* data_env = std::getenv("LINA_BENCH_DATA_PATH");
        const std::size_t n = n_env ? static_cast<std::size_t>(std::stoul(n_env)) : 4;
        const int iters = it_env ? std::stoi(it_env) : 200;
        const std::string data_path = data_env ? data_env : "";
        const auto input = build_fft_input(n, data_path);
        lina::fft_gpu(input);
        lina::fft_gpu(input);
        print_bench("BENCH_FFT_GPU " + std::to_string(n) + " " + std::to_string(iters), iters,
                    [&input] { lina::fft_gpu(input); });
#else
        std::cerr << "bench_fft_gpu requires LINA_USE_CUDA=ON\n";
        return 1;
#endif
    } else if (cmd == "bench_fft_gpu_kernel") {
#ifdef LINA_USE_CUDA
        const char* n_env = std::getenv("LINA_BENCH_FFT_N");
        const char* it_env = std::getenv("LINA_BENCH_FFT_ITERS");
        const char* data_env = std::getenv("LINA_BENCH_DATA_PATH");
        const std::size_t n = n_env ? static_cast<std::size_t>(std::stoul(n_env)) : 4;
        const int iters = it_env ? std::stoi(it_env) : 200;
        const std::string data_path = data_env ? data_env : "";
        const auto input = build_fft_input(n, data_path);
        const double avg_ms = lina::benchmark_fft_gpu_kernel_ms(input, iters);
        std::cout << "BENCH_FFT_GPU_KERNEL " << n << " " << iters << " " << avg_ms << "\n";
#else
        std::cerr << "bench_fft_gpu_kernel requires LINA_USE_CUDA=ON\n";
        return 1;
#endif
    } else if (cmd == "bench_fft_gpu_xfer") {
#ifdef LINA_USE_CUDA
        const char* n_env = std::getenv("LINA_BENCH_FFT_N");
        const char* it_env = std::getenv("LINA_BENCH_FFT_ITERS");
        const char* data_env = std::getenv("LINA_BENCH_DATA_PATH");
        const std::size_t n = n_env ? static_cast<std::size_t>(std::stoul(n_env)) : 4;
        const int iters = it_env ? std::stoi(it_env) : 200;
        const std::string data_path = data_env ? data_env : "";
        const auto input = build_fft_input(n, data_path);
        const double avg_ms = lina::benchmark_fft_gpu_xfer_ms(input, iters);
        std::cout << "BENCH_FFT_GPU_XFER " << n << " " << iters << " " << avg_ms << "\n";
#else
        std::cerr << "bench_fft_gpu_xfer requires LINA_USE_CUDA=ON\n";
        return 1;
#endif
    } else if (cmd == "bench_fft_gpu_e2e") {
#ifdef LINA_USE_CUDA
        const char* n_env = std::getenv("LINA_BENCH_FFT_N");
        const char* it_env = std::getenv("LINA_BENCH_FFT_ITERS");
        const char* data_env = std::getenv("LINA_BENCH_DATA_PATH");
        const std::size_t n = n_env ? static_cast<std::size_t>(std::stoul(n_env)) : 4;
        const int iters = it_env ? std::stoi(it_env) : 200;
        const std::string data_path = data_env ? data_env : "";
        const auto input = build_fft_input(n, data_path);
        const double avg_ms = lina::benchmark_fft_gpu_e2e_ms(input, iters);
        std::cout << "BENCH_FFT_GPU_E2E " << n << " " << iters << " " << avg_ms << "\n";
#else
        std::cerr << "bench_fft_gpu_e2e requires LINA_USE_CUDA=ON\n";
        return 1;
#endif
    } else if (cmd == "bench_svd") {
        print_bench("BENCH_SVD", 200, [] { compute_svd_float(); });
    } else if (cmd == "bench_svd_size") {
        const char* m_env = std::getenv("LINA_BENCH_SVD_M");
        const char* n_env = std::getenv("LINA_BENCH_SVD_N");
        const char* it_env = std::getenv("LINA_BENCH_SVD_ITERS");
        const char* data_env = std::getenv("LINA_BENCH_DATA_PATH");
        const std::size_t m = m_env ? static_cast<std::size_t>(std::stoul(m_env)) : 5000;
        const std::size_t n = n_env ? static_cast<std::size_t>(std::stoul(n_env)) : 2000;
        const int iters = it_env ? std::stoi(it_env) : 1;
        const std::string data_path = data_env ? data_env : "";
        const auto input = build_svd_input(m, n, data_path);
        lina::svd_float(input);
        lina::svd_float(input);
        print_bench("BENCH_SVD_SIZE " + std::to_string(m) + " " + std::to_string(n) + " " +
                        std::to_string(iters),
                    iters, [&input] { lina::svd_float(input); });
    } else if (cmd == "bench_svd_size_cpu") {
        const char* m_env = std::getenv("LINA_BENCH_SVD_M");
        const char* n_env = std::getenv("LINA_BENCH_SVD_N");
        const char* it_env = std::getenv("LINA_BENCH_SVD_ITERS");
        const char* data_env = std::getenv("LINA_BENCH_DATA_PATH");
        const std::size_t m = m_env ? static_cast<std::size_t>(std::stoul(m_env)) : 5000;
        const std::size_t n = n_env ? static_cast<std::size_t>(std::stoul(n_env)) : 2000;
        const int iters = it_env ? std::stoi(it_env) : 1;
        const std::string data_path = data_env ? data_env : "";
        const auto input = build_svd_input(m, n, data_path);
        lina::svd_float_cpu(input);
        lina::svd_float_cpu(input);
        print_bench("BENCH_SVD_SIZE_CPU " + std::to_string(m) + " " + std::to_string(n) + " " +
                        std::to_string(iters),
                    iters, [&input] { lina::svd_float_cpu(input); });
    } else if (cmd == "bench_svd_size_gpu") {
#ifdef LINA_USE_CUDA
        const char* m_env = std::getenv("LINA_BENCH_SVD_M");
        const char* n_env = std::getenv("LINA_BENCH_SVD_N");
        const char* it_env = std::getenv("LINA_BENCH_SVD_ITERS");
        const char* data_env = std::getenv("LINA_BENCH_DATA_PATH");
        const std::size_t m = m_env ? static_cast<std::size_t>(std::stoul(m_env)) : 5000;
        const std::size_t n = n_env ? static_cast<std::size_t>(std::stoul(n_env)) : 2000;
        const int iters = it_env ? std::stoi(it_env) : 1;
        const std::string data_path = data_env ? data_env : "";
        const auto input = build_svd_input(m, n, data_path);
        lina::svd_float_gpu(input);
        lina::svd_float_gpu(input);
        print_bench("BENCH_SVD_SIZE_GPU " + std::to_string(m) + " " + std::to_string(n) + " " +
                        std::to_string(iters),
                    iters, [&input] { lina::svd_float_gpu(input); });
#else
        std::cerr << "bench_svd_size_gpu requires LINA_USE_CUDA=ON\n";
        return 1;
#endif
    } else if (cmd == "bench_svd_size_gpu_kernel") {
#ifdef LINA_USE_CUDA
        const char* m_env = std::getenv("LINA_BENCH_SVD_M");
        const char* n_env = std::getenv("LINA_BENCH_SVD_N");
        const char* it_env = std::getenv("LINA_BENCH_SVD_ITERS");
        const char* data_env = std::getenv("LINA_BENCH_DATA_PATH");
        const std::size_t m = m_env ? static_cast<std::size_t>(std::stoul(m_env)) : 5000;
        const std::size_t n = n_env ? static_cast<std::size_t>(std::stoul(n_env)) : 2000;
        const int iters = it_env ? std::stoi(it_env) : 1;
        const std::string data_path = data_env ? data_env : "";
        const auto input = build_svd_input(m, n, data_path);
        const double avg_ms = lina::benchmark_svd_float_gpu_kernel_ms(input, iters);
        std::cout << "BENCH_SVD_SIZE_GPU_KERNEL " << m << " " << n << " " << iters << " " << avg_ms << "\n";
#else
        std::cerr << "bench_svd_size_gpu_kernel requires LINA_USE_CUDA=ON\n";
        return 1;
#endif
    } else if (cmd == "bench_svd_size_gpu_xfer") {
#ifdef LINA_USE_CUDA
        const char* m_env = std::getenv("LINA_BENCH_SVD_M");
        const char* n_env = std::getenv("LINA_BENCH_SVD_N");
        const char* it_env = std::getenv("LINA_BENCH_SVD_ITERS");
        const char* data_env = std::getenv("LINA_BENCH_DATA_PATH");
        const std::size_t m = m_env ? static_cast<std::size_t>(std::stoul(m_env)) : 5000;
        const std::size_t n = n_env ? static_cast<std::size_t>(std::stoul(n_env)) : 2000;
        const int iters = it_env ? std::stoi(it_env) : 1;
        const std::string data_path = data_env ? data_env : "";
        const auto input = build_svd_input(m, n, data_path);
        const double avg_ms = lina::benchmark_svd_float_gpu_xfer_ms(input, iters);
        std::cout << "BENCH_SVD_SIZE_GPU_XFER " << m << " " << n << " " << iters << " " << avg_ms << "\n";
#else
        std::cerr << "bench_svd_size_gpu_xfer requires LINA_USE_CUDA=ON\n";
        return 1;
#endif
    } else if (cmd == "bench_svd_size_gpu_e2e") {
#ifdef LINA_USE_CUDA
        const char* m_env = std::getenv("LINA_BENCH_SVD_M");
        const char* n_env = std::getenv("LINA_BENCH_SVD_N");
        const char* it_env = std::getenv("LINA_BENCH_SVD_ITERS");
        const char* data_env = std::getenv("LINA_BENCH_DATA_PATH");
        const std::size_t m = m_env ? static_cast<std::size_t>(std::stoul(m_env)) : 5000;
        const std::size_t n = n_env ? static_cast<std::size_t>(std::stoul(n_env)) : 2000;
        const int iters = it_env ? std::stoi(it_env) : 1;
        const std::string data_path = data_env ? data_env : "";
        const auto input = build_svd_input(m, n, data_path);
        const double avg_ms = lina::benchmark_svd_float_gpu_e2e_ms(input, iters);
        std::cout << "BENCH_SVD_SIZE_GPU_E2E " << m << " " << n << " " << iters << " " << avg_ms << "\n";
#else
        std::cerr << "bench_svd_size_gpu_e2e requires LINA_USE_CUDA=ON\n";
        return 1;
#endif
    } else if (cmd == "bench_svd_autocal") {
#ifdef LINA_USE_CUDA
        const auto cal = lina::calibrate_svd_float_gpu_threshold();
        std::cout << "BENCH_SVD_AUTOCAL_THRESHOLD " << cal.threshold_elems
                  << " SM_" << cal.sm << "\n";
        for (const auto& p : cal.points) {
            std::cout << "BENCH_SVD_AUTOCAL_POINT "
                      << p.m << " " << p.n << " "
                      << p.gesvd_ms << " " << p.gesvdj_ms << "\n";
        }
        std::cout << "BENCH_SVD_AUTOCAL_ACTIVE_THRESHOLD "
                  << lina::svd_float_gpu_threshold_elems() << "\n";
#else
        std::cerr << "bench_svd_autocal requires LINA_USE_CUDA=ON\n";
        return 1;
#endif
    } else if (cmd == "bench_svd_large") {
        const char* m_env = std::getenv("LINA_BENCH_SVD_M");
        const char* n_env = std::getenv("LINA_BENCH_SVD_N");
        const std::size_t m = m_env ? static_cast<std::size_t>(std::stoul(m_env)) : 5000;
        const std::size_t n = n_env ? static_cast<std::size_t>(std::stoul(n_env)) : 2000;
        const auto input = build_svd_input(m, n, "");
        print_bench("BENCH_SVD_LARGE", 1, [&input] { lina::svd_float(input); });
    } else if (cmd == "bench_efc") {
        print_bench("BENCH_EFC", 50, [] { compute_efc(); });
    } else if (cmd == "bench_iefc") {
        print_bench("BENCH_IEFC", 50, [] { compute_iefc(); });
    } else {
        std::cerr << "Unknown command\n";
        return 1;
    }
    return 0;
}
