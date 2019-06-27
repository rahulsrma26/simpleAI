#include "snn/loss/hillinger.hpp"

namespace snn {
namespace losses {

constexpr real h_eps = 1e-5f;

hillinger::hillinger(const kwargs& args) { std::ignore = args; }

std::string hillinger::type() { return TEXT::HILLINGER; }

std::string hillinger::name() const { return this->type(); }

real hillinger::f(const tensor<real>& o, const tensor<real>& l) const {
    real r = 0;
    for (size_t i = 0; i < o.size(); i++) {
        const real d = std::sqrt(o[i]) - std::sqrt(l[i]);
        r += d * d;
    }
    return r / (std::sqrt(real(2.0)) * o.get_shape().front());
}

tensor<real> hillinger::df(const tensor<real>& o, const tensor<real>& l) const {
    tensor<real> r(o.get_shape());
    const auto n = o.get_shape().front();
    for (uint i = 0; i < o.size(); i++) {
        const real denom = std::sqrt(2.0f * o[i]);
        r[i] = (std::sqrt(o[i]) - std::sqrt(l[i])) / ((h_eps + denom) * n);
    }
    return r;
}

void hillinger::save(std::ostream& os) const { std::ignore = os; };

hillinger::hillinger(std::istream& is) { std::ignore = is; };

} // namespace losses
} // namespace snn
