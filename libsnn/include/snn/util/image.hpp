#pragma once

#include <fstream>
#include "snn/nntypes.hpp"
#include "snn/math/tensor.hpp"

namespace snn {

void save_pgm(const std::string& filename, const tensor<real>& matrix, int levels = 100);

} // namespace snn