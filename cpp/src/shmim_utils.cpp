#include "lina/shmim_utils.h"

#include <cstring>
#include <stdexcept>
#include <vector>

#ifdef LINA_USE_IMAGESTREAMIO
#include <ImageStreamIO.h>
#include <ImageStruct.h>
#endif

namespace lina {

struct ShmimStream::Impl {
#ifdef LINA_USE_IMAGESTREAMIO
    IMAGE image{};
    bool open = false;
#endif
    std::size_t rows = 0;
    std::size_t cols = 0;
};

ShmimStream::ShmimStream() : impl_(new Impl()) {}

ShmimStream::ShmimStream(const std::string& name) : ShmimStream() {
    open(name);
}

ShmimStream::~ShmimStream() {
    close();
    delete impl_;
}

void ShmimStream::open(const std::string& name) {
#ifndef LINA_USE_IMAGESTREAMIO
    throw std::runtime_error("ImageStreamIO not enabled");
#else
    if (impl_->open) {
        close();
    }
    if (ImageStreamIO_openIm(&impl_->image, name.c_str()) != IMAGESTREAMIO_SUCCESS) {
        throw std::runtime_error("ImageStreamIO_openIm failed");
    }
    impl_->open = true;
    impl_->rows = impl_->image.md->size[0];
    impl_->cols = impl_->image.md->size[1];
#endif
}

void ShmimStream::create(const std::string& name,
                         std::size_t rows,
                         std::size_t cols,
                         int datatype,
                         std::size_t cbsize) {
#ifndef LINA_USE_IMAGESTREAMIO
    throw std::runtime_error("ImageStreamIO not enabled");
#else
    if (impl_->open) {
        close();
    }
    uint32_t sizes[2] = {static_cast<uint32_t>(rows), static_cast<uint32_t>(cols)};
    if (ImageStreamIO_createIm(&impl_->image, name.c_str(), 2, sizes,
                               static_cast<uint8_t>(datatype), 1, 8,
                               static_cast<int>(cbsize)) != IMAGESTREAMIO_SUCCESS) {
        throw std::runtime_error("ImageStreamIO_createIm failed");
    }
    impl_->open = true;
    impl_->rows = rows;
    impl_->cols = cols;
#endif
}

void ShmimStream::close() {
#ifdef LINA_USE_IMAGESTREAMIO
    if (impl_->open) {
        ImageStreamIO_closeIm(&impl_->image);
        impl_->open = false;
    }
#endif
}

std::size_t ShmimStream::rows() const {
    return impl_->rows;
}

std::size_t ShmimStream::cols() const {
    return impl_->cols;
}

Array2D<double> ShmimStream::grab_latest() {
#ifndef LINA_USE_IMAGESTREAMIO
    throw std::runtime_error("ImageStreamIO not enabled");
#else
    if (!impl_->open) {
        throw std::runtime_error("ShmimStream not open");
    }
    void* buffer = nullptr;
    if (ImageStreamIO_readLastWroteBuffer(&impl_->image, &buffer) != IMAGESTREAMIO_SUCCESS) {
        throw std::runtime_error("ImageStreamIO_readLastWroteBuffer failed");
    }

    const std::size_t n = impl_->rows * impl_->cols;
    Array2D<double> out(impl_->rows, impl_->cols, 0.0);
    if (impl_->image.md->datatype == _DATATYPE_FLOAT) {
        const float* src = static_cast<const float*>(buffer);
        for (std::size_t i = 0; i < n; ++i) {
            out.data()[i] = static_cast<double>(src[i]);
        }
    } else if (impl_->image.md->datatype == _DATATYPE_DOUBLE) {
        const double* src = static_cast<const double*>(buffer);
        std::memcpy(out.data(), src, sizeof(double) * n);
    } else {
        throw std::runtime_error("Unsupported shmim datatype");
    }
    return out;
#endif
}

Array2D<double> ShmimStream::grab_mean(std::size_t nframes) {
#ifndef LINA_USE_IMAGESTREAMIO
    throw std::runtime_error("ImageStreamIO not enabled");
#else
    if (!impl_->open) {
        throw std::runtime_error("ShmimStream not open");
    }
    if (nframes == 0) {
        throw std::invalid_argument("nframes must be > 0");
    }

    const std::size_t rows = impl_->rows;
    const std::size_t cols = impl_->cols;
    const std::size_t n = rows * cols;
    Array2D<double> mean(rows, cols, 0.0);

    const uint64_t n_slices = ImageStreamIO_nbSlices(&impl_->image);
    const uint64_t last_index = ImageStreamIO_readLastWroteIndex(&impl_->image);
    const std::size_t frames = std::min<std::size_t>(nframes, n_slices);

    for (std::size_t i = 0; i < frames; ++i) {
        const uint64_t idx = (last_index + n_slices - i) % n_slices;
        void* buffer = nullptr;
        if (ImageStreamIO_readBufferAt(&impl_->image, static_cast<unsigned int>(idx), &buffer)
            != IMAGESTREAMIO_SUCCESS) {
            throw std::runtime_error("ImageStreamIO_readBufferAt failed");
        }
        if (impl_->image.md->datatype == _DATATYPE_FLOAT) {
            const float* src = static_cast<const float*>(buffer);
            for (std::size_t j = 0; j < n; ++j) {
                mean.data()[j] += static_cast<double>(src[j]);
            }
        } else if (impl_->image.md->datatype == _DATATYPE_DOUBLE) {
            const double* src = static_cast<const double*>(buffer);
            for (std::size_t j = 0; j < n; ++j) {
                mean.data()[j] += src[j];
            }
        } else {
            throw std::runtime_error("Unsupported shmim datatype");
        }
    }

    const double inv = 1.0 / static_cast<double>(frames);
    for (std::size_t j = 0; j < n; ++j) {
        mean.data()[j] *= inv;
    }
    return mean;
#endif
}

void ShmimStream::write(const Array2D<double>& data) {
    write_scaled(data, 1.0);
}

void ShmimStream::write_scaled(const Array2D<double>& data, double scale) {
#ifndef LINA_USE_IMAGESTREAMIO
    throw std::runtime_error("ImageStreamIO not enabled");
#else
    if (!impl_->open) {
        throw std::runtime_error("ShmimStream not open");
    }
    if (data.rows() != impl_->rows || data.cols() != impl_->cols) {
        throw std::invalid_argument("write size mismatch");
    }
    void* buffer = nullptr;
    if (ImageStreamIO_writeBuffer(&impl_->image, &buffer) != IMAGESTREAMIO_SUCCESS) {
        throw std::runtime_error("ImageStreamIO_writeBuffer failed");
    }
    const std::size_t n = impl_->rows * impl_->cols;
    if (impl_->image.md->datatype == _DATATYPE_FLOAT) {
        float* dst = static_cast<float*>(buffer);
        for (std::size_t i = 0; i < n; ++i) {
            dst[i] = static_cast<float>(data.data()[i] * scale);
        }
    } else if (impl_->image.md->datatype == _DATATYPE_DOUBLE) {
        double* dst = static_cast<double*>(buffer);
        for (std::size_t i = 0; i < n; ++i) {
            dst[i] = data.data()[i] * scale;
        }
    } else {
        throw std::runtime_error("Unsupported shmim datatype");
    }
    ImageStreamIO_sempost(&impl_->image, -1);
#endif
}

void ShmimStream::zero() {
    Array2D<double> zeros(rows(), cols(), 0.0);
    write(zeros);
}

} // namespace lina
