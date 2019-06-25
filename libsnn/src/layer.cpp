#include "snn/layer.hpp"

namespace snn {

template <class T>
std::unique_ptr<layers::base_layer> get_layer_from_type(const std::string& type, T args) {
    if (type == layers::dense::type())
        return std::make_unique<layers::dense>(args);
    else if (type == layers::dropout::type())
        return std::make_unique<layers::dropout>(args);
    else if (type == layers::flatten::type())
        return std::make_unique<layers::flatten>(args);
    throw std::runtime_error("Invalid layer type: " + type);
}

void layer::create(const kwargs& args) {
    auto [type, sub_args] = args.get_function("");
    layer_m = get_layer_from_type<const kwargs&>(type, sub_args);
}

void layer::load(std::istream& is) {
    uint32_t type_length;
    is.read(reinterpret_cast<char*>(&type_length), sizeof(uint32_t));
    std::string type(type_length, ' ');
    is.read(reinterpret_cast<char*>(&type[0]), type_length);
    layer_m = get_layer_from_type<std::istream&>(type, is);
}

void layer::save(std::ostream& os, bool save_gradient) const {
    std::string type = layer_m->name();
    uint32_t type_length = type.length();
    os.write(reinterpret_cast<const char*>(&type_length), sizeof(uint32_t));
    os.write(reinterpret_cast<const char*>(&type[0]), type_length);
    layer_m->save(os, save_gradient);
}

std::string layer::name() const { return layer_m->name(); }

std::string layer::info() const { return layer_m->info(); }

shape layer::output() const { return layer_m->output(); }

size_t layer::params() const { return layer_m->params(); }

void layer::set_optimizer(const kwargs& args) { return layer_m->set_optimizer(args); }

tensor<real> layer::forward(tensor<real>& prev_activation) {
    return layer_m->forward(prev_activation);
}

tensor<real> layer::predict(tensor<real>& input) { return layer_m->predict(input); }

tensor<real> layer::backward(tensor<real>& pre_gradients) {
    return layer_m->backward(pre_gradients);
}

} // namespace snn