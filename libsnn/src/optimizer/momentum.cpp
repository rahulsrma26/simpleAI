#include "snn/optimizer/momentum.hpp"

namespace snn {
namespace optimizers {

momentum::momentum(const tensor<real>& t, const kwargs& args) : velocity_m(t.get_shape()) {
    learning_rate_m = args.get(TEXT::LEARNING_RATE, 0.001);
    decay_m = args.get(TEXT::DECAY, 0);
    moment_m = args.get(TEXT::MOMENT, 0.5);
}

std::string momentum::type() { return TEXT::MOMENTUM; }

std::string momentum::name() const { return this->type(); }

void momentum::update(tensor<real>& t, const tensor<real>& g) {
    const size_t n = t.size();
    const real lr = learning_rate_m;
#pragma omp parallel if (n >= OPENMP_SMALL_THRESHOLD)
    {
#pragma omp for simd
        for (size_t i = 0; i < n; i++)
            velocity_m[i] = moment_m * velocity_m[i] - lr * g[i];
    }
#pragma omp parallel if (n >= OPENMP_SMALL_THRESHOLD)
    {
#pragma omp for simd
        for (size_t i = 0; i < n; i++)
            t[i] += velocity_m[i];
    }
    learning_rate_m *= (1.0 - decay_m);
}

void momentum::save(std::ostream& os, bool save_gradient) const {
    std::ignore = save_gradient;
    os.write(reinterpret_cast<const char*>(&learning_rate_m), sizeof(learning_rate_m));
    os.write(reinterpret_cast<const char*>(&decay_m), sizeof(decay_m));
    os.write(reinterpret_cast<const char*>(&moment_m), sizeof(moment_m));
    velocity_m.to_stream(os);
};

momentum::momentum(std::istream& is) : velocity_m({1}) {
    is.read(reinterpret_cast<char*>(&learning_rate_m), sizeof(learning_rate_m));
    is.read(reinterpret_cast<char*>(&decay_m), sizeof(decay_m));
    is.read(reinterpret_cast<char*>(&moment_m), sizeof(moment_m));
    velocity_m = tensor<real>::from_stream(is);
};

} // namespace optimizers
} // namespace snn