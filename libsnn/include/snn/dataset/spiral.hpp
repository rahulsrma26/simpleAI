#pragma once

#include <tuple>
#include <cmath>
#include "snn/nntypes.hpp"
#include "snn/math/tensor.hpp"

namespace snn {
namespace dataset {
namespace spiral {

std::tuple<tensor<real>, tensor<real>, tensor<real>, tensor<real>>
generate(shapeType samples = 1000, bool sin_cos = false);

} // namespace spiral
} // namespace dataset
} // namespace snn
