#include "snn/optimizer/adagrad.hpp"

namespace snn {
namespace optimizers {

adagrad::adagrad(const tensor<real>& t, const kwargs& args)
    : cache_m(t.get_shape()) {
    learning_rate_m = args.get(TEXT::LEARNING_RATE, 0.001);
    decay_m = args.get(TEXT::DECAY, 0);
    eps_m = args.get(TEXT::EPSILON, 1e-8);
}

std::string adagrad::type() { return TEXT::ADAGRAD; }

std::string adagrad::name() const { return this->type(); }

void adagrad::update(tensor<real>& t, const tensor<real>& g) {
    const size_t n = t.size();
    const real lr = learning_rate_m;
#pragma omp parallel if (n >= OPENMP_SMALL_THRESHOLD)
    {
#pragma omp for simd
        for (size_t i = 0; i < n; i++) {
            cache_m[i] += g[i] * g[i];
            t[i] -= learning_rate_m * g[i] / (std::sqrt(cache_m[i]) + eps_m);
        }
    }
    learning_rate_m *= (1.0 - decay_m);
}

void adagrad::save(std::ostream& os, bool save_gradient) const {
    std::ignore = save_gradient;
    os.write(reinterpret_cast<const char*>(&learning_rate_m), sizeof(learning_rate_m));
    os.write(reinterpret_cast<const char*>(&decay_m), sizeof(decay_m));
    os.write(reinterpret_cast<const char*>(&eps_m), sizeof(eps_m));
    cache_m.to_stream(os);
};

adagrad::adagrad(std::istream& is) : cache_m({1}) {
    is.read(reinterpret_cast<char*>(&learning_rate_m), sizeof(learning_rate_m));
    is.read(reinterpret_cast<char*>(&decay_m), sizeof(decay_m));
    is.read(reinterpret_cast<char*>(&eps_m), sizeof(eps_m));
    cache_m = tensor<real>::from_stream(is);
};

} // namespace optimizers
} // namespace snn