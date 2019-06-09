#include "snn/activator.hpp"

namespace snn {

// Todo: template<class T>

void activator::create(const kwargs& args) {
    auto [type, sub_args] = args.get_function("");
    if (type == activators::sigmoid::type())
        activator_m = std::make_unique<activators::sigmoid>(sub_args);
    else if (type == activators::relu::type())
        activator_m = std::make_unique<activators::relu>(sub_args);
    else if (type == activators::tanh::type())
        activator_m = std::make_unique<activators::tanh>(sub_args);
    else
        throw std::runtime_error("Invalid activator type: " + type);
}

void activator::load(std::istream& is) {
    uint32_t type_length;
    is.read(reinterpret_cast<char*>(&type_length), sizeof(uint32_t));
    std::string type(type_length, ' ');
    is.read(reinterpret_cast<char*>(&type[0]), type_length);
    if (type == activators::sigmoid::type())
        activator_m = std::make_unique<activators::sigmoid>(is);
    else if (type == activators::relu::type())
        activator_m = std::make_unique<activators::relu>(is);
    else if (type == activators::tanh::type())
        activator_m = std::make_unique<activators::tanh>(is);
    else
        throw std::runtime_error("Invalid loss type: " + type);
};

void activator::save(std::ostream& os) const {
    std::string type = activator_m->name();
    uint32_t type_length = type.length();
    os.write(reinterpret_cast<const char*>(&type_length), sizeof(uint32_t));
    os.write(reinterpret_cast<const char*>(&type[0]), type_length);
    activator_m->save(os);
};

std::string activator::name() const { return activator_m->name(); }

tensor<real> activator::f(const tensor<real>& t) const { return activator_m->f(t); }

tensor<real> activator::df(const tensor<real>& t) const { return activator_m->df(t); }

} // namespace snn