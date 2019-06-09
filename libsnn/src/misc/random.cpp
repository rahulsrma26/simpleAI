#include "snn/misc/random.hpp"

namespace snn {

std::mt19937 variable_random_engine;

void seed(unsigned long s) { variable_random_engine.seed(s); }

} // namespace snn