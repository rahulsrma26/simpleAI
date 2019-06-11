#include "snn/optimizer/rmsprop.hpp"

namespace snn {
namespace optimizers {

rmsprop::rmsprop(const tensor<real>& t, const kwargs& args)
    : cache_m(t.get_shape()) {
    learning_rate_m = args.get(TEXT::LEARNING_RATE, 0.001);
    decay_m = args.get(TEXT::DECAY, 0);
    decay_rate = args.get(TEXT::DECAY_RATE, 0.9);
    eps_m = args.get(TEXT::EPSILON, 1e-8);
}

std::string rmsprop::type() { return TEXT::RMSPROP; }

std::string rmsprop::name() const { return this->type(); }

void rmsprop::update(tensor<real>& t, const tensor<real>& g) {
    const size_t n = t.size();
    const real lr = learning_rate_m;
#pragma omp parallel if (n >= OPENMP_SMALL_THRESHOLD)
    {
#pragma omp for simd
        for (size_t i = 0; i < n; i++) {
            cache_m[i] = decay_rate * cache_m[i] + (1 - decay_rate) * g[i] * g[i];
            t[i] -= learning_rate_m * g[i] / (std::sqrt(cache_m[i]) + eps_m);
        }
    }
    learning_rate_m *= (1.0 - decay_m);
}

void rmsprop::save(std::ostream& os, bool save_gradient) const {
    std::ignore = save_gradient;
    os.write(reinterpret_cast<const char*>(&learning_rate_m), sizeof(learning_rate_m));
    os.write(reinterpret_cast<const char*>(&decay_m), sizeof(decay_m));
    os.write(reinterpret_cast<const char*>(&decay_rate), sizeof(decay_rate));
    os.write(reinterpret_cast<const char*>(&eps_m), sizeof(eps_m));
    cache_m.to_stream(os);
};

rmsprop::rmsprop(std::istream& is) : cache_m({1}) {
    is.read(reinterpret_cast<char*>(&learning_rate_m), sizeof(learning_rate_m));
    is.read(reinterpret_cast<char*>(&decay_m), sizeof(decay_m));
    is.read(reinterpret_cast<char*>(&decay_rate), sizeof(decay_rate));
    is.read(reinterpret_cast<char*>(&eps_m), sizeof(eps_m));
    cache_m = tensor<real>::from_stream(is);
};

} // namespace optimizers
} // namespace snn