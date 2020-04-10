#include "snn/layer/dense.hpp"

namespace snn {
namespace layers {

dense::dense(const kwargs& args) : input_m({1}), weighted_input_m({1}), use_bias_m(true) {
    inputs_m = args.get_int_vector(TEXT::INPUT).front();
    outputs_m = (size_t)args.get_int(TEXT::UNITS);

    activator_m.create(args.get(TEXT::ACTIVATION, TEXT::SIGMOID + "()"));

    auto kernel_initializer = args.get(TEXT::KERNEL_INITIALIZER, TEXT::XAVIER + "()");
    kwargs weight_args("");
    weight_args.set_int_vector(TEXT::SHAPE, {(int)outputs_m, (int)inputs_m});
    weight_args.set(TEXT::INITIALIZER, kernel_initializer);
    weight_m.create(weight_args);

    auto use_bias_m = args.get(TEXT::USE_BIAS, true);
    if (use_bias_m) {
        auto bias_initializer = args.get(TEXT::BIAS_INITIALIZER, TEXT::ZEROS + "()");
        kwargs bias_args("");
        bias_args.set_int_vector(TEXT::SHAPE, {(int)outputs_m});
        bias_args.set(TEXT::INITIALIZER, bias_initializer);
        bias_m.create(bias_args);
    }
}

std::string dense::type() { return TEXT::DENSE; }

std::string dense::name() const { return this->type(); }

std::string dense::info() const { return name() + "(" + activator_m.name() + ")"; }

shape dense::output() const { return {(shapeType)outputs_m}; }

size_t dense::params() const { return weight_m.var.size() + (use_bias_m ? bias_m.var.size() : 0); }

void dense::set_optimizer(const kwargs& args) {
    weight_m.set_optimizer(args);
    if (use_bias_m)
        bias_m.set_optimizer(args);
}

tensor<real> dense::forward(tensor<real>& prev_activation) {
    // NxI x IxO = NxO
    input_m = std::move(prev_activation);
    const ompint batch_size = input_m.get_shape().front();
    weighted_input_m = tensor<real>({(shapeType)batch_size, (shapeType)outputs_m});

#pragma omp parallel for
    for (ompint i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < outputs_m; j++) {
            real sum = use_bias_m ? bias_m.var[j] : 0.0;
            for (size_t k = 0; k < inputs_m; k++)
                sum += input_m[i * inputs_m + k] * weight_m.var[j * inputs_m + k];
            weighted_input_m[i * outputs_m + j] = sum;
        }
    }
    return activator_m.f(weighted_input_m);
}

tensor<real> dense::predict(tensor<real>& input) {
    const ompint batch_size = input.get_shape().front();
    tensor<real> output({(shapeType)batch_size, (shapeType)outputs_m});

#pragma omp parallel for
    for (ompint i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < outputs_m; j++) {
            real sum = use_bias_m ? bias_m.var[j] : 0.0;
            for (size_t k = 0; k < inputs_m; k++)
                sum += input[i * inputs_m + k] * weight_m.var[j * inputs_m + k];
            output[i * outputs_m + j] = sum;
        }
    }
    return activator_m.f(output);
}

tensor<real> dense::backward(tensor<real>& pre_gradients) {
    auto delta = activator_m.df(weighted_input_m);
    delta *= pre_gradients;
    const ompint batch_size = input_m.get_shape().front();

    if (use_bias_m) {
        tensor<real> bias_grad({(shapeType)outputs_m});
        for (int i = 0; i < batch_size; i++)
            for (size_t j = 0; j < outputs_m; j++)
                bias_grad[j] += delta[i * outputs_m + j];
        bias_m.optimize(bias_grad);
    }

    tensor<real> next_grad({(shapeType)batch_size, (shapeType)inputs_m});
#pragma omp parallel if (batch_size >= OPENMP_LARGE_THRESHOLD)
    {
        for (ompint i = 0; i < batch_size; i++)
#pragma omp for
            for (ompint j = 0; j < (int)inputs_m; j++) {
                real sum = 0.0;
                for (size_t k = 0; k < outputs_m; k++)
                    sum += weight_m.var[k * inputs_m + j] * delta[i * outputs_m + k];
                next_grad[i * inputs_m + j] = sum;
            }
    }

    tensor<real> weight_grad({(shapeType)outputs_m, (shapeType)inputs_m});
#pragma omp parallel for if (outputs_m >= OPENMP_LARGE_THRESHOLD)
    for (ompint j = 0; j < (int)outputs_m; j++)
        for (size_t k = 0; k < inputs_m; k++) {
            real sum = 0;
            for (int i = 0; i < batch_size; i++)
                sum += delta[i * outputs_m + j] * input_m[i * inputs_m + k];
            weight_grad[j * inputs_m + k] = sum;
        }

    weight_m.optimize(weight_grad);

    return next_grad;
}

void dense::save(std::ostream& os, bool save_gradient) const {
    weight_m.save(os, save_gradient);
    os.write(reinterpret_cast<const char*>(&use_bias_m), sizeof(bool));
    if (use_bias_m)
        bias_m.save(os, save_gradient);
    activator_m.save(os);
    os.write(reinterpret_cast<const char*>(&inputs_m), sizeof(size_t));
    os.write(reinterpret_cast<const char*>(&outputs_m), sizeof(size_t));
}

dense::dense(std::istream& is) : input_m({1}), weighted_input_m({1}) {
    weight_m.load(is);
    is.read(reinterpret_cast<char*>(&use_bias_m), sizeof(bool));
    if (use_bias_m)
        bias_m.load(is);
    activator_m.load(is);
    is.read(reinterpret_cast<char*>(&inputs_m), sizeof(size_t));
    is.read(reinterpret_cast<char*>(&outputs_m), sizeof(size_t));
}

} // namespace layers
} // namespace snn
