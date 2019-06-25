#pragma once

#include <vector>
#include <algorithm>
#include <functional>
#include <iostream>
#include <type_traits>
#include <stdexcept>
#include <string>

namespace snn {

constexpr size_t LOW_DIM_PRINT_LIMIT = 12;
constexpr size_t LOW_DIM_PRINT_ELLIPSES = 4;
static_assert(
    LOW_DIM_PRINT_LIMIT > LOW_DIM_PRINT_ELLIPSES * 2,
    "Full element limit should be more than twice of ellipses element limit to prevent overflow.");

constexpr size_t HIGH_DIM_PRINT_LIMIT = 5;
constexpr size_t HIGH_DIM_PRINT_ELLIPSES = 2;
static_assert(
    HIGH_DIM_PRINT_LIMIT > HIGH_DIM_PRINT_ELLIPSES * 2,
    "Full element limit should be more than twice of ellipses element limit to prevent overflow.");

constexpr char PRINT_BEGIN = '[';
constexpr char PRINT_END = ']';

template <class T>
size_t vector_get_size(const std::vector<T>& s) {
    size_t m = 1;
    for (const auto& x : s)
        m *= x;
    return m;
}

template <class T>
std::string vector_to_string(const std::vector<T>& t) {
    std::stringstream ss;
    if (t.size())
        ss << t[0];
    else
        ss << "EMPTY";
    for (size_t i = 1; i < t.size(); i++)
        ss << 'x' << t[i];
    return ss.str();
}

template <class T>
void vector_to_stream(std::ostream& os, const std::vector<T>& s) {
    uint32_t ds = s.size();
    os.write(reinterpret_cast<const char*>(&ds), sizeof(uint32_t));
    os.write(reinterpret_cast<const char*>(s.data()), sizeof(T) * ds);
}

template <class T>
std::vector<T> vector_from_stream(std::istream& is) {
    uint32_t ds;
    is.read(reinterpret_cast<char*>(&ds), sizeof(uint32_t));
    std::vector<T> s(ds);
    is.read(reinterpret_cast<char*>(s.data()), sizeof(T) * ds);
    return s;
}

typedef int shapeType;
typedef std::vector<shapeType> shape;

template <class T>
class tensor {
    static_assert(std::is_fundamental<T>::value, "Only fundamental type is supported");

    shape dim_m{1};
    std::vector<T> data_m;

    template <class Ch, class Tr>
    void print(std::basic_ostream<Ch, Tr>&, const shapeType, const size_t) const;

public:
    tensor(const shape&);
    tensor(shape&&);
    tensor(const tensor<T>&);
    tensor(tensor<T>&&) noexcept;

    tensor(const shape&, const std::vector<T>&);
    tensor(const shape&, std::vector<T>&&);

    tensor<T>& operator=(const tensor<T>&);
    tensor<T>& operator=(tensor<T>&&) noexcept;

    bool empty() const noexcept;
    size_t size() const noexcept;
    const shape& get_shape() const;
    std::vector<T>& data();
    const std::vector<T>& data() const;

    void fill(const T& val);
    void fill(const std::vector<T>&);
    void fill(const tensor<T>&);

    tensor<T>& reshape(const shape&);
    tensor<T>& squeeze(const shape& axis = {});
    tensor<T>& expand_dims(const size_t);

    template <typename Function>
    void apply(Function generator);

    template <typename Function>
    void initialize(Function generator);

    void transpose(const std::vector<size_t>& permutation);

    T& operator[](const shape&);
    const T& operator[](const shape&) const;

    T& operator[](const size_t);
    const T& operator[](const size_t) const;

    tensor<T>& operator+=(const T);
    tensor<T>& operator+=(const tensor<T>&);
    tensor<T>& operator-=(const T);
    tensor<T>& operator-=(const tensor<T>&);
    tensor<T>& operator*=(const T);
    tensor<T>& operator*=(const tensor<T>&);
    tensor<T>& operator/=(const T);
    tensor<T>& operator/=(const tensor<T>&);

    typename std::vector<T>::iterator begin() noexcept;
    typename std::vector<T>::iterator end() noexcept;
    typename std::vector<T>::const_iterator begin() const noexcept;
    typename std::vector<T>::const_iterator end() const noexcept;
    typename std::vector<T>::const_iterator cbegin() const noexcept;
    typename std::vector<T>::const_iterator cend() const noexcept;
    typename std::vector<T>::reverse_iterator rbegin() noexcept;
    typename std::vector<T>::reverse_iterator rend() noexcept;
    typename std::vector<T>::const_reverse_iterator rbegin() const noexcept;
    typename std::vector<T>::const_reverse_iterator rend() const noexcept;
    typename std::vector<T>::const_reverse_iterator crbegin() const noexcept;
    typename std::vector<T>::const_reverse_iterator crend() const noexcept;

    template <class U>
    friend tensor<U> operator+(const tensor<U>&, const U);
    template <class U>
    friend tensor<U> operator+(const U, const tensor<U>&);
    template <class U>
    friend tensor<U> operator-(const tensor<U>&, const U);
    template <class U>
    friend tensor<U> operator-(const U, const tensor<U>&);
    template <class U>
    friend tensor<U> operator*(const tensor<U>&, const U);
    template <class U>
    friend tensor<U> operator*(const U, const tensor<U>&);
    template <class U>
    friend tensor<U> operator/(const tensor<U>&, const U);
    template <class U>
    friend tensor<U> operator/(const U, const tensor<U>&);

    template <class U>
    friend tensor<U> operator+(const tensor<U>&, const tensor<U>&);
    template <class U>
    friend tensor<U> operator-(const tensor<U>&, const tensor<U>&);
    template <class U>
    friend tensor<U> operator*(const tensor<U>&, const tensor<U>&);
    template <class U>
    friend tensor<U> operator/(const tensor<U>&, const tensor<U>&);

    template <class U>
    tensor<U> cast();
    void to_stream(std::ostream& os) const;
    static tensor<T> from_stream(std::istream& is);

    template <class Ch, class Tr, class U>
    friend std::basic_ostream<Ch, Tr>& operator<<(std::basic_ostream<Ch, Tr>& os,
                                                  const tensor<U>& t);
};

template <class Ch, class Tr>
std::basic_ostream<Ch, Tr>& operator<<(std::basic_ostream<Ch, Tr>&, const shape&);

// -------------------------------------------------------------------------------

template <class T>
tensor<T>::tensor(const shape& dim) : dim_m(dim) {
    auto n = vector_get_size(dim_m);
    if (!n)
        throw std::runtime_error("Invalid shape. Shape can not be zero.");
    data_m.resize(n);
}

template <class T>
tensor<T>::tensor(shape&& dim) : dim_m(std::move(dim)) {
    auto n = vector_get_size(dim_m);
    if (!n)
        throw std::runtime_error("Invalid shape. Shape can not be zero.");
    data_m.resize(n);
}

template <typename T>
tensor<T>::tensor(const tensor<T>& other) : dim_m(other.dim_m), data_m(other.data_m) {}

template <typename T>
tensor<T>::tensor(tensor<T>&& other) noexcept
    : dim_m(std::move(other.dim_m)), data_m(std::move(other.data_m)) {}

template <typename T>
tensor<T>::tensor(const shape& dim, const std::vector<T>& data) : dim_m(dim) {
    auto n = vector_get_size(dim_m);
    if (!n)
        throw std::runtime_error("Invalid shape. Shape can not be zero.");
    if (n != data.size())
        throw std::runtime_error("Invalid shape. Shape and size are not matching.");
    data_m = data;
}

template <typename T>
tensor<T>::tensor(const shape& dim, std::vector<T>&& data) : dim_m(dim), data_m(std::move(data)) {
    auto n = vector_get_size(dim_m);
    if (!n)
        throw std::runtime_error("Invalid shape. Shape can not be zero.");
    if (n != data_m.size())
        throw std::runtime_error("Invalid shape. Shape and size are not matching.");
}

template <typename T>
template <class U>
tensor<U> tensor<T>::cast() {
    std::vector<U> casted(data_m.begin(), data_m.end());
    return tensor<U>(dim_m, std::move(casted));
}

// -------------------------------------------------------------------------------

template <typename T>
tensor<T>& tensor<T>::operator=(const tensor<T>& other) {
    dim_m = other.dim_m;
    data_m = other.data_m;
    return *this;
}

template <typename T>
tensor<T>& tensor<T>::operator=(tensor<T>&& other) noexcept {
    dim_m = std::move(other.dim_m);
    data_m = std::move(other.data_m);
    return *this;
}

// -------------------------------------------------------------------------------

template <typename T>
std::vector<T>& tensor<T>::data() {
    return data_m;
}

template <typename T>
const std::vector<T>& tensor<T>::data() const {
    return data_m;
}

template <typename T>
bool tensor<T>::empty() const noexcept {
    return data_m.empty();
}

template <typename T>
size_t tensor<T>::size() const noexcept {
    return data_m.size();
}

template <typename T>
const shape& tensor<T>::get_shape() const {
    return dim_m;
}

// -------------------------------------------------------------------------------

template <typename T>
tensor<T>& tensor<T>::reshape(const shape& s) {
    size_t n = 1, z = 0;
    for (const auto& x : s) {
        if (x)
            n *= x;
        else
            z++;
    }

    if (z > 1)
        throw std::runtime_error("Can't reshape as shape contains more than one zero.");

    if (z) {
        const size_t m = data_m.size() / n;
        if (m * n != data_m.size())
            throw std::runtime_error("Invalid shape after filling missing.");
        dim_m = s;
        for (auto& x : dim_m) {
            if (!x) {
                x = m;
                break;
            }
        }
        return *this;
    }

    if (n != data_m.size())
        throw std::runtime_error("Invalid reshape shape. Total size not matching.");
    dim_m = s;
    return *this;
}

template <typename T>
tensor<T>& tensor<T>::squeeze(const shape& axis) {
    shape new_dim;
    for (size_t i = 0; i < dim_m.size(); i++)
        if (dim_m[i] != 1 || (axis.size() && std::find(axis.begin(), axis.end(), i) == axis.end()))
            new_dim.push_back(dim_m[i]);
    dim_m = std::move(new_dim);
    if (dim_m.size() == 0 || vector_get_size(dim_m) == 0)
        dim_m = {1};
    return *this;
}

template <typename T>
tensor<T>& tensor<T>::expand_dims(const size_t axis) {
    dim_m.insert(dim_m.begin() + axis, 1);
    return *this;
}

// -------------------------------------------------------------------------------

template <class T>
void permute(std::vector<T>& data, const std::vector<size_t>& permutation) {
    auto copy = data;
    for (size_t i = 0; i < copy.size(); i++)
        data[i] = copy[permutation[i]];
}

template <typename T>
void tensor<T>::transpose(const std::vector<size_t>& permutation) {
    if (dim_m.size() < 2)
        return;
    if (dim_m.size() != permutation.size())
        throw std::runtime_error("Can't transpose. Permutation size (" +
                                 std::to_string(permutation.size()) + ") != Shape size (" +
                                 std::to_string(dim_m.size()) + ")");
    const auto n = dim_m.size();
    std::vector<size_t> inc(n, 1);
    for (size_t i = n - 1; i; i--)
        inc[i - 1] *= inc[i] * dim_m[i];

    permute(inc, permutation);
    permute(dim_m, permutation);

    auto data = data_m;
    shape idx(dim_m.size(), 0);
    for (size_t i = 0, j = 0; i < data.size(); i++) {
        data_m[i] = data[j];
        idx[permutation[n - 1]]++;
        j += inc[n - 1];
        for (size_t k = n - 1; k && idx[permutation[k]] >= dim_m[k];) {
            idx[permutation[k]] = 0;
            j -= inc[k] * dim_m[k];
            k--;
            idx[permutation[k]]++;
            j += inc[k];
        }
    }
}

// -------------------------------------------------------------------------------

template <typename T>
void tensor<T>::fill(const T& val) {
    std::fill(data_m.begin(), data_m.end(), val);
}

template <typename T>
void tensor<T>::fill(const std::vector<T>& other) {
    const auto n = data_m.size();
    const auto m = other.size();
    const auto k = n / m;
    if (n < m || k * m != n)
        throw std::runtime_error("Can't broadcast. Size is not a multiple.");
    for (size_t i = 0; i < k; i++)
        std::copy(other.cbegin(), other.cend(), data_m.begin() + i * m);
}

template <typename T>
void tensor<T>::fill(const tensor<T>& other) {
    fill(other.data_m);
}

// -------------------------------------------------------------------------------

template <typename T>
template <typename Function>
void tensor<T>::apply(Function generator) {
    for (auto& element : data_m)
        element = generator(element);
}

template <typename T>
template <typename Function>
void tensor<T>::initialize(Function generator) {
    for (auto& element : data_m)
        element = generator();
}

// -------------------------------------------------------------------------------

template <typename T>
T& tensor<T>::operator[](const size_t idx) {
    return data_m[idx];
}

template <typename T>
const T& tensor<T>::operator[](const size_t idx) const {
    return data_m[idx];
}

template <typename T>
T& tensor<T>::operator[](const shape& idx) {
    size_t index = idx.back();
    for (size_t i = idx.size() - 1; i; i--)
        index = index * dim_m[i] + idx[i - 1];
    return data_m[index];
}

template <typename T>
const T& tensor<T>::operator[](const shape& idx) const {
    size_t index = idx.back();
    for (size_t i = idx.size() - 1; i; i--)
        index = index * dim_m[i] + idx[i - 1];
    return data_m[index];
}

// -------------------------------------------------------------------------------

template <typename T>
tensor<T>& tensor<T>::operator+=(const T v) {
    for (auto& x : data_m)
        x += v;
    return *this;
}

template <typename T>
tensor<T>& tensor<T>::operator-=(const T v) {
    for (auto& x : data_m)
        x -= v;
    return *this;
}

template <typename T>
tensor<T>& tensor<T>::operator*=(const T v) {
    for (auto& x : data_m)
        x *= v;
    return *this;
}

template <typename T>
tensor<T>& tensor<T>::operator/=(const T v) {
    for (auto& x : data_m)
        x /= v;
    return *this;
}

template <typename T>
tensor<T>& tensor<T>::operator+=(const tensor<T>& other) {
    const auto n = data_m.size();
    const auto m = other.size();
    const auto k = n / m;
    if (n < m || k * m != n)
        throw std::runtime_error("Can't broadcast. Size is not a multiple.");
    for (size_t l = 0; l < k; l++)
        for (size_t i = 0, j = l * m; i < m; i++, j++)
            data_m[j] += other.data_m[i];
    return *this;
}

template <typename T>
tensor<T>& tensor<T>::operator-=(const tensor<T>& other) {
    const auto n = data_m.size();
    const auto m = other.size();
    const auto k = n / m;
    if (n < m || k * m != n)
        throw std::runtime_error("Can't broadcast. Size is not a multiple.");
    for (size_t l = 0; l < k; l++)
        for (size_t i = 0, j = l * m; i < m; i++, j++)
            data_m[j] -= other.data_m[i];
    return *this;
}

template <typename T>
tensor<T>& tensor<T>::operator*=(const tensor<T>& other) {
    const auto n = data_m.size();
    const auto m = other.size();
    const auto k = n / m;
    if (n < m || k * m != n)
        throw std::runtime_error("Can't broadcast. Size is not a multiple.");
    for (size_t l = 0; l < k; l++)
        for (size_t i = 0, j = l * m; i < m; i++, j++)
            data_m[j] *= other.data_m[i];
    return *this;
}

template <typename T>
tensor<T>& tensor<T>::operator/=(const tensor<T>& other) {
    const auto n = data_m.size();
    const auto m = other.size();
    const auto k = n / m;
    if (n < m || k * m != n)
        throw std::runtime_error("Can't broadcast. Size is not a multiple.");
    for (size_t l = 0; l < k; l++)
        for (size_t i = 0, j = l * m; i < m; i++, j++)
            data_m[j] /= other.data_m[i];
    return *this;
}

template <typename U>
tensor<U> operator+(const tensor<U>& a, const U b) {
    tensor<U> c(a.dim_m);
    for (size_t i = 0; i < a.size(); i++)
        c.data_m[i] = a.data_m[i] + b;
    return c;
}

template <class U>
tensor<U> operator+(const U b, const tensor<U>& a) {
    tensor<U> c(a.dim_m);
    for (size_t i = 0; i < a.size(); i++)
        c.data_m[i] = b + a.data_m[i];
    return c;
}

template <typename U>
tensor<U> operator-(const tensor<U>& a, const U b) {
    tensor<U> c(a.dim_m);
    for (size_t i = 0; i < a.size(); i++)
        c.data_m[i] = a.data_m[i] - b;
    return c;
}

template <typename U>
tensor<U> operator-(const U b, const tensor<U>& a) {
    tensor<U> c(a.dim_m);
    for (size_t i = 0; i < a.size(); i++)
        c.data_m[i] = b - a.data_m[i];
    return c;
}

template <typename U>
tensor<U> operator*(const tensor<U>& a, const U b) {
    tensor<U> c(a.dim_m);
    for (size_t i = 0; i < a.size(); i++)
        c.data_m[i] = a.data_m[i] * b;
    return c;
}

template <typename U>
tensor<U> operator*(const U b, const tensor<U>& a) {
    tensor<U> c(a.dim_m);
    for (size_t i = 0; i < a.size(); i++)
        c.data_m[i] = b * a.data_m[i];
    return c;
}

template <typename U>
tensor<U> operator/(const tensor<U>& a, const U b) {
    tensor<U> c(a.dim_m);
    for (size_t i = 0; i < a.size(); i++)
        c.data_m[i] = a.data_m[i] / b;
    return c;
}

template <typename U>
tensor<U> operator/(const U b, const tensor<U>& a) {
    tensor<U> c(a.dim_m);
    for (size_t i = 0; i < a.size(); i++)
        c.data_m[i] = b / a.data_m[i];
    return c;
}

template <typename T>
tensor<T> operator+(const tensor<T>& a, const tensor<T>& b) {
    const auto n = a.size();
    const auto m = b.size();
    const auto k = n / m;
    if (n < m || k * m != n)
        throw std::runtime_error("Can't broadcast. Size is not a multiple.");
    tensor<T> c(a.dim_m);
    for (size_t l = 0; l < k; l++)
        for (size_t i = 0, j = l * m; i < m; i++, j++)
            c.data_m[j] = a.data_m[j] + b.data_m[i];
    return c;
}

template <typename T>
tensor<T> operator-(const tensor<T>& a, const tensor<T>& b) {
    const auto n = a.size();
    const auto m = b.size();
    const auto k = n / m;
    if (n < m || k * m != n)
        throw std::runtime_error("Can't broadcast. Size is not a multiple.");
    tensor<T> c(a.dim_m);
    for (size_t l = 0; l < k; l++)
        for (size_t i = 0, j = l * m; i < m; i++, j++)
            c.data_m[j] = a.data_m[j] - b.data_m[i];
    return c;
}

template <typename T>
tensor<T> operator*(const tensor<T>& a, const tensor<T>& b) {
    const auto n = a.size();
    const auto m = b.size();
    const auto k = n / m;
    if (n < m || k * m != n)
        throw std::runtime_error("Can't broadcast. Size is not a multiple.");
    tensor<T> c(a.dim_m);
    for (size_t l = 0; l < k; l++)
        for (size_t i = 0, j = l * m; i < m; i++, j++)
            c.data_m[j] = a.data_m[j] * b.data_m[i];
    return c;
}

template <typename T>
tensor<T> operator/(const tensor<T>& a, const tensor<T>& b) {
    const auto n = a.size();
    const auto m = b.size();
    const auto k = n / m;
    if (n < m || k * m != n)
        throw std::runtime_error("Can't broadcast. Size is not a multiple.");
    tensor<T> c(a.dim_m);
    for (size_t l = 0; l < k; l++)
        for (size_t i = 0, j = l * m; i < m; i++, j++)
            c.data_m[j] = a.data_m[j] / b.data_m[i];
    return c;
}

// -------------------------------------------------------------------------------

template <typename T>
typename std::vector<T>::iterator tensor<T>::begin() noexcept {
    return data_m.begin();
}

template <typename T>
typename std::vector<T>::iterator tensor<T>::end() noexcept {
    return data_m.end();
}

template <typename T>
typename std::vector<T>::const_iterator tensor<T>::begin() const noexcept {
    return data_m.cbegin();
}

template <typename T>
typename std::vector<T>::const_iterator tensor<T>::end() const noexcept {
    return data_m.cend();
}

template <typename T>
typename std::vector<T>::const_iterator tensor<T>::cbegin() const noexcept {
    return data_m.cbegin();
}

template <typename T>
typename std::vector<T>::const_iterator tensor<T>::cend() const noexcept {
    return data_m.cend();
}

template <typename T>
typename std::vector<T>::reverse_iterator tensor<T>::rbegin() noexcept {
    return data_m.rbegin();
}

template <typename T>
typename std::vector<T>::reverse_iterator tensor<T>::rend() noexcept {
    return data_m.rend();
}

template <typename T>
typename std::vector<T>::const_reverse_iterator tensor<T>::rbegin() const noexcept {
    return data_m.crbegin();
}

template <typename T>
typename std::vector<T>::const_reverse_iterator tensor<T>::rend() const noexcept {
    return data_m.crend();
}

template <typename T>
typename std::vector<T>::const_reverse_iterator tensor<T>::crbegin() const noexcept {
    return data_m.crbegin();
}

template <typename T>
typename std::vector<T>::const_reverse_iterator tensor<T>::crend() const noexcept {
    return data_m.crend();
}

// -------------------------------------------------------------------------------

template <class T>
void tensor<T>::to_stream(std::ostream& os) const {
    vector_to_stream(os, dim_m);
    os.write(reinterpret_cast<const char*>(data_m.data()), sizeof(T) * data_m.size());
}

template <class T>
tensor<T> tensor<T>::from_stream(std::istream& is) {
    tensor<T> t(vector_from_stream<shapeType>(is));
    is.read(reinterpret_cast<char*>(t.data_m.data()), sizeof(T) * t.data_m.size());
    return t;
}

template <typename T>
template <class Ch, class Tr>
void tensor<T>::print(std::basic_ostream<Ch, Tr>& os, const shapeType dim, const size_t idx) const {
    const auto n = (size_t)dim_m[dim - 1];
    os << PRINT_BEGIN;
    if (dim == (shapeType)dim_m.size()) {
        if (n <= LOW_DIM_PRINT_LIMIT) {
            for (size_t i = idx; i < idx + n; i++)
                os << (i == idx ? "" : " ") << data_m[i];
        } else {
            for (size_t i = idx; i < idx + LOW_DIM_PRINT_ELLIPSES; i++)
                os << (i == idx ? "" : " ") << data_m[i];
            os << " ...";
            for (size_t i = idx + n - LOW_DIM_PRINT_ELLIPSES; i < idx + n; i++)
                os << ' ' << data_m[i];
        }
    } else {
        std::string spaces(dim, ' ');
        size_t inc = 1;
        for (size_t i = dim; i < dim_m.size(); i++)
            inc *= dim_m[i];
        if (n <= HIGH_DIM_PRINT_LIMIT) {
            for (size_t i = idx, j = 0; j < n; i += inc, j++) {
                if (j)
                    os << spaces;
                print(os, dim + 1, i);
                if (j != n - 1)
                    os << '\n';
            }
        } else {
            for (size_t i = idx, j = 0; j < HIGH_DIM_PRINT_ELLIPSES; j++, i += inc) {
                if (j)
                    os << spaces;
                print(os, dim + 1, i);
                os << '\n';
            }
            os << spaces << "..." << '\n';
            for (size_t i = idx + (n - HIGH_DIM_PRINT_ELLIPSES) * inc, j = 0;
                 j < HIGH_DIM_PRINT_ELLIPSES; i += inc, j++) {
                os << spaces;
                print(os, dim + 1, i);
                if (j != HIGH_DIM_PRINT_ELLIPSES - 1)
                    os << '\n';
            }
        }
    }
    os << PRINT_END;
}

template <class Ch, class Tr, class U>
std::basic_ostream<Ch, Tr>& operator<<(std::basic_ostream<Ch, Tr>& os, const tensor<U>& t) {
    if (!t.dim_m.size())
        os << PRINT_BEGIN << PRINT_END;
    else
        t.print(os, 1, 0);
    return os;
}

template <class Ch, class Tr>
std::basic_ostream<Ch, Tr>& operator<<(std::basic_ostream<Ch, Tr>& os, const shape& s) {
    os << '(';
    for (size_t i = 0; i < s.size(); i++)
        os << (i ? " " : "") << s[i];
    return os << ')';
}

} // namespace snn
