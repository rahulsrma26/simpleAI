#include "snn/initializer/normal.hpp"

namespace snn {
namespace initializers {

normal::normal(const kwargs& args) {
    distribution_m =
        std::normal_distribution<real>(args.get(TEXT::MEAN, 0.0), args.get(TEXT::STDDEV, 0.05));
}

std::string normal::type() { return TEXT::NORMAL; }

std::string normal::name() const { return this->type(); }

void normal::init(tensor<real>& t) {
    for (auto& x : t)
        x = distribution_m(variable_random_engine);
}

} // namespace initializers
} // namespace snn