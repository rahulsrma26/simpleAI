#include "snn/initializer/xavier.hpp"

namespace snn {
namespace initializers {

xavier::xavier(const kwargs& args) { uniform = args.get(TEXT::UNIFORM, true); }

std::string xavier::type() { return TEXT::XAVIER; }

std::string xavier::name() const { return this->type(); }

void xavier::init(tensor<real>& t) {
    const auto sh = t.get_shape();
    if (uniform) {
        double x = sqrt(6.0 / ((double)sh.back() + sh.front()));
        std::uniform_real_distribution<real> distribution(-x, x);
        for (auto& x : t)
            x = distribution(variable_random_engine);
    } else {
        double x = sqrt(2.0 / ((double)sh.back() + sh.front()));
        std::normal_distribution<real> distribution(0.0, x);
        for (auto& x : t)
            x = distribution(variable_random_engine);
    }
}

} // namespace initializers
} // namespace snn