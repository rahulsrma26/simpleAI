#pragma once

#include <tuple>
#include <cmath>
#include "snn/nntypes.hpp"
#include "snn/math/tensor.hpp"

namespace snn {
namespace dataset {

typedef std::pair<tensor<real>, tensor<real>> dataset2;
typedef std::tuple<tensor<real>, tensor<real>, tensor<real>, tensor<real>> dataset4;

} // namespace dataset
} // namespace snn
