#include "snn/activator/relu.hpp"

namespace snn {
namespace activators {

relu::relu(const kwargs& args) { std::ignore = args; }

std::string relu::type() { return TEXT::RELU; }

std::string relu::name() const { return this->type(); }

tensor<real> relu::f(const tensor<real>& t) const {
    tensor<real> r(t.get_shape());
    const auto n = t.size();
    for (size_t i = 0; i < n; i++)
        r[i] = t[i] > 0 ? t[i] : 0;
    return r;
}

tensor<real> relu::df(const tensor<real>& t) const {
    tensor<real> r(t.get_shape());
    const auto n = t.size();
    for (size_t i = 0; i < n; i++)
        r[i] = t[i] > 0 ? 1 : 0;
    return r;
}

void relu::save(std::ostream& os) const { std::ignore = os; };

relu::relu(std::istream& is) { std::ignore = is; };

} // namespace activators
} // namespace snn
