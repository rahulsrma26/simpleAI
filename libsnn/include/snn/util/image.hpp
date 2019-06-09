#pragma once

#include "snn/nntypes.hpp"
#include "snn/math/tensor.hpp"

namespace snn {

void save_image(const std::string& filename, const tensor<real>& matrix, int levels = 100);

} // namespace snn