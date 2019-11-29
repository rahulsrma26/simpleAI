#include "snn/optimizer/adam.hpp"

namespace snn {
namespace optimizers {

adam::adam(const tensor<real>& t, const kwargs& args)
    : momentum_m(t.get_shape()), velocity_m(t.get_shape()) {
    learning_rate_m = args.get(TEXT::LEARNING_RATE, 0.001);
    decay_m = args.get(TEXT::DECAY, 0);
    beta1_m = args.get(TEXT::BETA1, 0.9);
    beta2_m = args.get(TEXT::BETA2, 0.999);
    eps_m = args.get(TEXT::EPSILON, 1e-8);
}

std::string adam::type() { return TEXT::ADAM; }

std::string adam::name() const { return this->type(); }

void adam::update(tensor<real>& t, const tensor<real>& g) {
    const ompint n = t.size();
#pragma omp parallel for if (n >= OPENMP_SMALL_THRESHOLD)
    for (ompint i = 0; i < n; i++) {
        momentum_m[i] = beta1_m * momentum_m[i] + (1 - beta1_m) * g[i];
        velocity_m[i] = beta2_m * velocity_m[i] + (1 - beta2_m) * g[i] * g[i];
        t[i] -= learning_rate_m * momentum_m[i] / (std::sqrt(velocity_m[i]) + eps_m);
    }
    learning_rate_m *= (1.0 - decay_m);
}

void adam::save(std::ostream& os, bool save_gradient) const {
    std::ignore = save_gradient;
    os.write(reinterpret_cast<const char*>(&learning_rate_m), sizeof(learning_rate_m));
    os.write(reinterpret_cast<const char*>(&decay_m), sizeof(decay_m));
    os.write(reinterpret_cast<const char*>(&beta1_m), sizeof(beta1_m));
    os.write(reinterpret_cast<const char*>(&beta2_m), sizeof(beta2_m));
    os.write(reinterpret_cast<const char*>(&eps_m), sizeof(eps_m));
    momentum_m.to_stream(os);
    velocity_m.to_stream(os);
};

adam::adam(std::istream& is) : momentum_m({1}), velocity_m({1}) {
    is.read(reinterpret_cast<char*>(&learning_rate_m), sizeof(learning_rate_m));
    is.read(reinterpret_cast<char*>(&decay_m), sizeof(decay_m));
    is.read(reinterpret_cast<char*>(&beta1_m), sizeof(beta1_m));
    is.read(reinterpret_cast<char*>(&beta2_m), sizeof(beta2_m));
    is.read(reinterpret_cast<char*>(&eps_m), sizeof(eps_m));
    momentum_m = tensor<real>::from_stream(is);
    velocity_m = tensor<real>::from_stream(is);
};

} // namespace optimizers
} // namespace snn