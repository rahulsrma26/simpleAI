#pragma once

#include <string>
#include <unordered_map>
#include <cstdint>
#include <typeinfo>
#include <tuple>

namespace snn {

typedef uint32_t uint;
typedef float real;

#ifdef _MSC_VER
typedef int ompint;
#else
typedef int ompint;
#endif

constexpr ompint OPENMP_MINI_THRESHOLD = 256;
constexpr ompint OPENMP_SMALL_THRESHOLD = 64;
constexpr ompint OPENMP_MEDIUM_THRESHOLD = 16;
constexpr ompint OPENMP_LARGE_THRESHOLD = 4;

} // namespace snn