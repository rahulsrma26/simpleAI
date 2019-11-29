#include "snn/activator/leaky_relu.hpp"

namespace snn {
namespace activators {

leaky_relu::leaky_relu(const kwargs& args) { leak_m = args.get_double(TEXT::LEAK); }

std::string leaky_relu::type() { return TEXT::LEAKY_RELU; }

std::string leaky_relu::name() const { return this->type(); }

tensor<real> leaky_relu::f(const tensor<real>& t) const {
    tensor<real> r(t.get_shape());
    const auto n = t.size();
#pragma omp parallel for if (n >= OPENMP_SMALL_THRESHOLD)
    for (ompint i = 0; i < n; i++)
        r[i] = t[i] > 0 ? t[i] : leak_m * t[i];
    return r;
}

tensor<real> leaky_relu::df(const tensor<real>& t) const {
    tensor<real> r(t.get_shape());
    const auto n = t.size();
#pragma omp parallel for if (n >= OPENMP_SMALL_THRESHOLD)
    for (ompint i = 0; i < n; i++)
        r[i] = t[i] > 0 ? 1 : leak_m;
    return r;
}

void leaky_relu::save(std::ostream& os) const {
    os.write(reinterpret_cast<const char*>(&leak_m), sizeof(leak_m));
};

leaky_relu::leaky_relu(std::istream& is) {
    is.read(reinterpret_cast<char*>(&leak_m), sizeof(leak_m));
};

} // namespace activators
} // namespace snn
