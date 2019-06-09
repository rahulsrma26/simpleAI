#pragma once

#include "snn/math/tensor.hpp"
#include <cmath>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace snn {
namespace math {

template <class T>
tensor<T> exp(const tensor<T>& t) {
    tensor<T> r(t.get_shape());
    for (size_t i = 0; i < t.size(); i++)
        r[i] = std::exp(t[i]);
    return r;
}

template <class T>
tensor<T> tanh(const tensor<T>& t) {
    tensor<T> r(t.get_shape());
    for (size_t i = 0; i < t.size(); i++)
        r[i] = std::tanh(t[i]);
    return r;
}

template <class T>
tensor<T> log(const tensor<T>& t) {
    tensor<T> r(t.get_shape());
    for (size_t i = 0; i < t.size(); i++)
        r[i] = std::log(t[i]);
    return r;
}

template <class T>
tensor<T> log(const tensor<T>& t, T epsilon) {
    tensor<T> r(t.get_shape());
    for (size_t i = 0; i < t.size(); i++)
        r[i] = std::log(t[i] + epsilon);
    return r;
}

template <class T>
tensor<T> matmul(const tensor<T>& a, const tensor<T>& b) {
    const auto a_dim = a.get_shape();
    if (a_dim.size() != 2)
        throw std::runtime_error("Invalid shape. First argument is not a 2D matrix.");

    const auto b_dim = b.get_shape();
    if (b_dim.size() != 2)
        throw std::runtime_error("Invalid shape. Second argument is not a 2D matrix.");

    if (a_dim[1] != b_dim[0])
        throw std::runtime_error("Invalid shape for matrix multiplication.");

    const size_t n = a_dim[0];
    const size_t p = b_dim[0];
    const size_t m = b_dim[1];

    tensor<T> c({(shapeType)(n), (shapeType)(m)});
#pragma omp parallel for
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < m; j++) {
            T sum{0};
            for (size_t k = 0; k < p; k++)
                sum += a[i * p + k] * b[k * m + j];
            c[i * m + j] = sum;
        }
    return c;
}

template <class T>
tensor<T> transpose(const tensor<T>& a) {
    const auto a_dim = a.get_shape();
    if (a_dim.size() != 2)
        throw std::runtime_error("Invalid shape. First argument is not a 2D matrix.");

    const size_t r = a_dim[1];
    const size_t c = a_dim[0];

    tensor<T> b({(shapeType)(r), (shapeType)(c)});
    for (size_t i = 0; i < r; i++)
        for (size_t j = 0; j < c; j++)
            b[i * c + j] = a[j * r + i];
    return b;
}

template <class T>
tensor<T> matmulT(const tensor<T>& a, const tensor<T>& bt) {
    const auto a_dim = a.get_shape();
    if (a_dim.size() != 2)
        throw std::runtime_error("Invalid shape. First argument is not a 2D matrix.");

    const auto bt_dim = bt.get_shape();
    if (bt_dim.size() != 2)
        throw std::runtime_error("Invalid shape. Second argument is not a 2D matrix.");

    if (a_dim[1] != bt_dim[1])
        throw std::runtime_error("Invalid shape for matrix multiplication.");

    const size_t n = a_dim[0];
    const size_t p = a_dim[1];
    const size_t m = bt_dim[0];

    tensor<T> c({(shapeType)(n), (shapeType)(m)});
#pragma omp parallel for
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < m; j++) {
            T sum{0};
            for (size_t k = 0; k < p; k++)
                sum += a[i * p + k] * bt[j * p + k];
            c[i * m + j] = sum;
        }
    return c;
}

} // namespace math
} // namespace snn
