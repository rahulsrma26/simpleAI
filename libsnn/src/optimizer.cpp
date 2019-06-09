#include "snn/optimizer.hpp"

namespace snn {

void optimizer::create(const tensor<real>& t, const kwargs& args) {
    auto [type, sub_args] = args.get_function("");
    if (type == optimizers::sgd::type())
        optimizer_m = std::make_unique<optimizers::sgd>(t, sub_args);
    else
        throw std::runtime_error("Invalid optimizer type: " + type);
}

void optimizer::load(std::istream& is) {
    uint32_t type_length;
    is.read(reinterpret_cast<char*>(&type_length), sizeof(uint32_t));
    std::string type(type_length, ' ');
    is.read(reinterpret_cast<char*>(&type[0]), type_length);
    if (type == optimizers::sgd::type())
        optimizer_m = std::make_unique<optimizers::sgd>(is);
    else
        throw std::runtime_error("Invalid optimizer type: " + type);
};

void optimizer::save(std::ostream& os, bool save_gradient) const {
    std::string type = optimizer_m->name();
    uint32_t type_length = type.length();
    os.write(reinterpret_cast<const char*>(&type_length), sizeof(uint32_t));
    os.write(reinterpret_cast<const char*>(&type[0]), type_length);
    optimizer_m->save(os, save_gradient);
};

std::string optimizer::name() const { return optimizer_m->name(); }

void optimizer::update(tensor<real>& t, const tensor<real>& g) { return optimizer_m->update(t, g); }

} // namespace snn