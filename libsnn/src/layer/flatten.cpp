#include "snn/layer/flatten.hpp"

namespace snn {
namespace layers {

flatten::flatten(const kwargs& args) {
    input_shape_m = vector_get_size(args.get_int_vector(TEXT::INPUT));
}

std::string flatten::type() { return TEXT::FLATTEN; }

std::string flatten::name() const { return this->type(); }

shape flatten::output() const { return {input_shape_m}; }

size_t flatten::params() const { return 0; }

void flatten::set_optimizer(const kwargs& args) { std::ignore = args; }

tensor<real> flatten::forward(tensor<real>& prev_activation) {
    saved_shape = prev_activation.get_shape();
    prev_activation.reshape({0, input_shape_m});
    return std::move(prev_activation);
}

tensor<real> flatten::predict(tensor<real>& input) {
    input.reshape({0, input_shape_m});
    return std::move(input);
}

tensor<real> flatten::backward(tensor<real>& pre_gradients) {
    pre_gradients.reshape(saved_shape);
    return std::move(pre_gradients);
}

void flatten::save(std::ostream& os, bool save_gradient) const {
    std::ignore = save_gradient;
    os.write(reinterpret_cast<const char*>(&input_shape_m), sizeof(input_shape_m));
};

flatten::flatten(std::istream& is) {
    is.read(reinterpret_cast<char*>(&input_shape_m), sizeof(input_shape_m));
};

} // namespace layers
} // namespace snn
