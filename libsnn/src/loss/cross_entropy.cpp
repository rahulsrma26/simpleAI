#include "snn/loss/cross_entropy.hpp"

namespace snn {
namespace losses {

constexpr real ce_eps = 1e-5f;

cross_entropy::cross_entropy(const kwargs& args) { std::ignore = args; }

std::string cross_entropy::type() { return TEXT::CROSS_ENTROPY; }

std::string cross_entropy::name() const { return this->type(); }

real cross_entropy::f(const tensor<real>& o, const tensor<real>& l) const {
    real r = 0;
    for (size_t i = 0; i < o.size(); i++)
        r += o[i] * std::log(ce_eps + l[i]) + (1 - o[i]) * std::log(1 + ce_eps - l[i]);
    return -r / o.get_shape().front();
}

tensor<real> cross_entropy::df(const tensor<real>& o, const tensor<real>& l) const {
    tensor<real> r(o.get_shape());
    const auto n = o.get_shape().front();
    for (uint i = 0; i < o.size(); i++) {
        const real denom = o[i] * (1 - o[i]);
        r[i] = (o[i] - l[i]) / ((ce_eps + denom) * n);
    }
    return r;
}

void cross_entropy::save(std::ostream& os) const { std::ignore = os; };

cross_entropy::cross_entropy(std::istream& is) { std::ignore = is; };

} // namespace losses
} // namespace snn
