#include "snn/optimizer/sgd.hpp"

namespace snn {
namespace optimizers {

sgd::sgd(const tensor<real>& t, const kwargs& args) {
    learning_rate_m = args.get(TEXT::LEARNING_RATE, 0.001);
    decay_m = args.get(TEXT::DECAY, 0);
    std::ignore = t;
}

std::string sgd::type() { return TEXT::SGD; }

std::string sgd::name() const { return this->type(); }

void sgd::update(tensor<real>& t, const tensor<real>& g) {
    const size_t n = t.size();
    const real lr = learning_rate_m;
#pragma omp parallel if (n >= OPENMP_SMALL_THRESHOLD)
    {
#pragma omp for simd
        for (size_t i = 0; i < n; i++)
            t[i] -= lr * g[i];
        learning_rate_m *= (1.0 - decay_m);
    }
}

void sgd::save(std::ostream& os, bool save_gradient) const {
    std::ignore = save_gradient;
    os.write(reinterpret_cast<const char*>(&learning_rate_m), sizeof(learning_rate_m));
    os.write(reinterpret_cast<const char*>(&decay_m), sizeof(decay_m));
};

sgd::sgd(std::istream& is) {
    is.read(reinterpret_cast<char*>(&learning_rate_m), sizeof(learning_rate_m));
    is.read(reinterpret_cast<char*>(&decay_m), sizeof(decay_m));
};

} // namespace optimizers
} // namespace snn