#pragma once

#include <random>
#include "snn/nntypes.hpp"

namespace snn {

extern std::mt19937 variable_random_engine;

void seed(unsigned long);

} // namespace snn