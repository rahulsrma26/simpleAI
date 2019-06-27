#pragma once

#include "snn/math/tensor.hpp"
#include "snn/nntypes.hpp"
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
tensor<T> log(const tensor<T>& t, T epsilon = T(0)) {
    tensor<T> r(t.get_shape());
    for (size_t i = 0; i < t.size(); i++)
        r[i] = std::log(t[i] + epsilon);
    return r;
}

template <class T>
tensor<T> matmul(const tensor<T>& a, const tensor<T>& b, const T bias = T(0)) {
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
            T sum{bias};
            for (size_t k = 0; k < p; k++)
                sum += a[i * p + k] * b[k * m + j];
            c[i * m + j] = sum;
        }
    return c;
}

template <class T>
tensor<T> matmulT_b(const tensor<T>& a, const tensor<T>& bt, const tensor<T>& bias) {
    const auto a_dim = a.get_shape();
    if (a_dim.size() != 2)
        throw std::runtime_error("Invalid shape. First argument is not a 2D matrix.");

    const auto bt_dim = bt.get_shape();
    if (bt_dim.size() != 2)
        throw std::runtime_error("Invalid shape. Second argument is not a 2D matrix.");

    if (a_dim[1] != bt_dim[1])
        throw std::runtime_error("Invalid shape for matrix multiplication.");

    const auto bias_dim = bias.get_shape();
    if (bias_dim.size() != 1)
        throw std::runtime_error("Invalid shape for bias. Bias is not a vector.");

    if (bias_dim[0] != bt_dim[0])
        throw std::runtime_error("Invalid shape for bias addition.");

    const size_t n = a_dim[0];
    const size_t p = a_dim[1];
    const size_t m = bt_dim[0];

    tensor<T> c({(shapeType)(n), (shapeType)(m)});
#pragma omp parallel for if (n >= OPENMP_LARGE_THRESHOLD)
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < m; j++) {
            T sum{bias[j]};
            for (size_t k = 0; k < p; k++)
                sum += a[i * p + k] * bt[j * p + k];
            c[i * m + j] = sum;
        }
    return c;
}

typedef std::pair<shapeType, shapeType> shapeBound;
typedef std::vector<shapeBound> shapeBounds;

template <typename T>
tensor<T> pad(const tensor<T>& t, const shapeBounds& padding) {
    const auto dim = t.get_shape();
    const int n = dim.size();
    if (padding.size() != (size_t)n)
        throw std::runtime_error("Padding elements should be same as tensor dimensions.");
    shape newShape(dim);
    for (int i = 0; i < n; i++)
        newShape[i] += padding[i].first + padding[i].second;

    tensor<T> padded(newShape);
#pragma omp parallel for if (n >= OPENMP_MINI_THRESHOLD)
    for (size_t i = 0; i < t.size(); i++) {
        int k1 = 0;
        for (int j = n - 1, k = i, s = 1; j >= 0; j--) {
            k1 += s * ((k % dim[j]) + padding[j].first);
            k /= dim[j];
            s *= newShape[j];
        }
        padded[k1] = t[i];
    }
    return padded;
}

template <typename T>
T l2_norm(const tensor<T>& a, const tensor<T>& b) {
    if (a.size() != b.size())
        throw std::runtime_error("Size should be same for l2_norm.");
    T r(0);
    for (size_t i = 0; i < a.size(); i++)
        r += (a[i] - b[i]) * (a[i] - b[i]);
    return r;
}

template <typename T>
tensor<T> batch_select(const tensor<T>& t, size_t batch, size_t index) {
    auto dim = t.get_shape();
    const size_t data_points = dim[0];
    const size_t data_size = t.size() / data_points;
    const size_t start_idx = batch * index;
    const size_t end_idx = std::min(start_idx + batch, data_points);
    dim[0] = end_idx - start_idx;
    tensor<T> r(dim);
    const auto n = r.size();
    for (size_t i = 0, j = start_idx * data_size; i < n; i++, j++)
        r[i] = t[j];
    return r;
}

} // namespace math
} // namespace snn
