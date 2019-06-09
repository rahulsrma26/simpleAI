#include "snn/loss/quadratic.hpp"

namespace snn {
namespace losses {

quadratic::quadratic(const kwargs& args) { std::ignore = args; }

std::string quadratic::type() { return TEXT::QUADRATIC; }

std::string quadratic::name() const { return this->type(); }

real quadratic::f(const tensor<real>& o, const tensor<real>& l) const {
    if (o.get_shape() != l.get_shape())
        throw std::runtime_error("Shapes not matching. Can not calculate loss.");

    real r = 0;
    for (size_t i = 0; i < o.size(); i++)
        r += (o[i] - l[i]) * (o[i] - l[i]);
    return r / (real(2.0) * o.get_shape().front());
}

tensor<real> quadratic::df(const tensor<real>& o, const tensor<real>& l) const {
    if (o.get_shape() != l.get_shape())
        throw std::runtime_error("Shapes not matching. Can not calculate loss derivative.");

    tensor<real> r(o.get_shape());
    const auto n = o.get_shape().front();
    for (uint i = 0; i < o.size(); i++)
        r[i] = (o[i] - l[i]) / n;
    return r;
}

void quadratic::save(std::ostream& os) const { std::ignore = os; };

quadratic::quadratic(std::istream& is) { std::ignore = is; };

} // namespace losses
} // namespace snn
