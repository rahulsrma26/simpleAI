#include "snn/optimizer.hpp"

namespace snn {

template <typename... Ts>
std::unique_ptr<optimizers::base_optimizer> get_optimizer_from_type(const std::string& type, Ts... args) {
    if (type == optimizers::sgd::type())
        return std::make_unique<optimizers::sgd>(args...);
    else if (type == optimizers::momentum::type())
        return std::make_unique<optimizers::momentum>(args...);
    else if (type == optimizers::adagrad::type())
        return std::make_unique<optimizers::adagrad>(args...);
    else if (type == optimizers::rmsprop::type())
        return std::make_unique<optimizers::rmsprop>(args...);
    else if (type == optimizers::adam::type())
        return std::make_unique<optimizers::adam>(args...);
    throw std::runtime_error("Invalid optimizer type: " + type);
}

void optimizer::create(const tensor<real>& t, const kwargs& args) {
    auto [type, sub_args] = args.get_function("");
    optimizer_m = get_optimizer_from_type<const tensor<real>&, const kwargs&>(type, t, sub_args);
}

void optimizer::load(std::istream& is) {
    uint32_t type_length;
    is.read(reinterpret_cast<char*>(&type_length), sizeof(uint32_t));
    std::string type(type_length, ' ');
    is.read(reinterpret_cast<char*>(&type[0]), type_length);
    optimizer_m = get_optimizer_from_type<std::istream&>(type, is);
}

void optimizer::save(std::ostream& os, bool save_gradient) const {
    std::string type = optimizer_m->name();
    uint32_t type_length = type.length();
    os.write(reinterpret_cast<const char*>(&type_length), sizeof(uint32_t));
    os.write(reinterpret_cast<const char*>(&type[0]), type_length);
    optimizer_m->save(os, save_gradient);
}

std::string optimizer::name() const { return optimizer_m->name(); }

void optimizer::update(tensor<real>& t, const tensor<real>& g) { return optimizer_m->update(t, g); }

} // namespace snn