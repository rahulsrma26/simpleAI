#include "snn/activator/tanh.hpp"

namespace snn {
namespace activators {

tanh::tanh(const kwargs& args) { std::ignore = args; }

std::string tanh::type() { return TEXT::TANH; }

std::string tanh::name() const { return this->type(); }

tensor<real> tanh::f(const tensor<real>& t) const { return math::tanh(t); }

tensor<real> tanh::df(const tensor<real>& t) const {
    auto r = math::tanh(t);
    for (size_t i = 0; i < t.size(); i++)
        r[i] = 1 - r[i] * r[i];
    return r;
}

void tanh::save(std::ostream& os) const { std::ignore = os; }

tanh::tanh(std::istream& is) { std::ignore = is; }

} // namespace activators
} // namespace snn
