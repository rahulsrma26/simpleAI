#pragma once

#include <string>
#include <unordered_map>
#include <cstdint>
#include <typeinfo>
#include <tuple>

namespace snn {

typedef uint32_t uint;
typedef float real;

constexpr size_t OPENMP_MINI_THRESHOLD = 256;
constexpr size_t OPENMP_SMALL_THRESHOLD = 64;
constexpr size_t OPENMP_MEDIUM_THRESHOLD = 16;
constexpr size_t OPENMP_LARGE_THRESHOLD = 4;

} // namespace snn