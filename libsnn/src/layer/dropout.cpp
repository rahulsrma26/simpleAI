#include "snn/layer/dropout.hpp"

namespace snn {
namespace layers {

dropout::dropout(const kwargs& args) : weight_m({1}) {
    inputs_m = args.get_int_vector(TEXT::INPUT);
    rate_m = (size_t)args.get_double(TEXT::RATE);
    generator_m = std::bernoulli_distribution(1 - rate_m);
}

std::string dropout::type() { return TEXT::DROPOUT; }

std::string dropout::name() const { return this->type(); }

shape dropout::output() const { return inputs_m; }

size_t dropout::params() const { return 0; }

void dropout::set_optimizer(const kwargs& args) { std::ignore = args; }

tensor<real> dropout::forward(tensor<real>& prev_activation) {
    // NxI x IxO = NxO
    const size_t n = prev_activation.size();
    if (weight_m.size() != n)
        weight_m = tensor<real>(prev_activation.get_shape());

#pragma omp parallel for if (n >= OPENMP_MEDIUM_THRESHOLD)
    for (size_t i = 0; i < n; i++)
        weight_m[i] = generator_m(variable_random_engine);

#pragma omp parallel for if (n >= OPENMP_SMALL_THRESHOLD)
    for (size_t i = 0; i < n; i++)
        prev_activation[i] *= weight_m[i];
    return prev_activation;
}

tensor<real> dropout::predict(tensor<real>& input) {
    const size_t n = input.size();
    const real multiplier = 1 - rate_m;
#pragma omp parallel for if (n >= OPENMP_SMALL_THRESHOLD)
    for (size_t i = 0; i < n; i++)
        input[i] *= multiplier;
    return input;
}

tensor<real> dropout::backward(tensor<real>& pre_gradients) {
    const size_t n = pre_gradients.size();
#pragma omp parallel for if (n >= OPENMP_SMALL_THRESHOLD)
    for (size_t i = 0; i < n; i++)
        pre_gradients[i] *= weight_m[i];
    return pre_gradients;
}

void dropout::save(std::ostream& os, bool save_gradient) const {
    std::ignore = save_gradient;
    vector_to_stream(os, inputs_m);
    os.write(reinterpret_cast<const char*>(&rate_m), sizeof(rate_m));
};

dropout::dropout(std::istream& is) : weight_m({1}) {
    inputs_m = vector_from_stream<shapeType>(is);
    is.read(reinterpret_cast<char*>(&rate_m), sizeof(rate_m));
    generator_m = std::bernoulli_distribution(1 - rate_m);
};

} // namespace layers
} // namespace snn
