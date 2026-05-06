#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace lina {

template <typename T>
class Array2D {
public:
    Array2D() : rows_(0), cols_(0) {}
    Array2D(std::size_t rows, std::size_t cols, const T& value = T())
        : rows_(rows), cols_(cols), data_(rows * cols, value) {}

    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }
    std::size_t size() const { return data_.size(); }

    const T* data() const { return data_.data(); }
    T* data() { return data_.data(); }

    const T& operator()(std::size_t r, std::size_t c) const {
        return data_.at(r * cols_ + c);
    }
    T& operator()(std::size_t r, std::size_t c) {
        return data_.at(r * cols_ + c);
    }

    void fill(const T& value) {
        std::fill(data_.begin(), data_.end(), value);
    }

private:
    std::size_t rows_;
    std::size_t cols_;
    std::vector<T> data_;
};

} // namespace lina
