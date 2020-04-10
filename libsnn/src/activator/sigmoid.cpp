#include "snn/activator/sigmoid.hpp"

namespace snn {
namespace activators {

sigmoid::sigmoid(const kwargs& args) { std::ignore = args; }

std::string sigmoid::type() { return TEXT::SIGMOID; }

std::string sigmoid::name() const { return this->type(); }

tensor<real> sigmoid::f(const tensor<real>& t) const {
    tensor<real> r(t.get_shape());
    const ompint n = t.size();
#pragma omp parallel for if (n >= OPENMP_SMALL_THRESHOLD)
    for (ompint i = 0; i < n; i++)
        r[i] = 1 / (1 + std::exp(-t[i]));
    return r;
}

tensor<real> sigmoid::df(const tensor<real>& t) const {
    tensor<real> r(t.get_shape());
    const ompint n = t.size();
#pragma omp parallel for if (n >= OPENMP_SMALL_THRESHOLD)
    for (ompint i = 0; i < n; i++) {
        const real v = 1 / (1 + std::exp(-t[i]));
        r[i] = v * (1 - v);
    }
    return r;
}

void sigmoid::save(std::ostream& os) const { std::ignore = os; }

sigmoid::sigmoid(std::istream& is) { std::ignore = is; }

} // namespace activators
} // namespace snn
