#pragma once

#include <regex>
#include "snn/nntypes.hpp"
#include "snn/math/tensor.hpp"

namespace snn {

namespace CONST {

const tensor<real> zero({1});
const tensor<real> one({1}, {1});

} // namespace constant

namespace REGEX{
const std::regex int_pattern("[-]{0,1}[\\d]*");
const std::regex double_pattern("[-]{0,1}[\\d]*[\\.]{0,1}[\\d]+([e]{0,1}[-]{0,1}[\\d]+){0,1}");
}

} // namespace snn