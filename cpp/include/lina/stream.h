#pragma once

#include "lina/array.h"

#include <cstddef>

namespace lina {

class Stream2D {
public:
    virtual ~Stream2D() = default;

    virtual Array2D<double> grab_latest() = 0;
    virtual Array2D<double> grab_mean(std::size_t nframes) = 0;
    virtual void write(const Array2D<double>& data) = 0;
    virtual std::size_t rows() const = 0;
    virtual std::size_t cols() const = 0;
};

} // namespace lina
