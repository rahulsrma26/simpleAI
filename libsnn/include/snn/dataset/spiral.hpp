#pragma once

#include <tuple>
#include <cmath>
#include "snn/nntypes.hpp"
#include "snn/dataset/dataset.hpp"
#include "snn/math/tensor.hpp"

namespace snn {
namespace dataset {
namespace spiral {

dataset4 generate(shapeType samples = 1000, bool sin_cos = false);

tensor<real> generate_grid(int radius, bool sin_cos = false);

} // namespace spiral
} // namespace dataset
} // namespace snn
