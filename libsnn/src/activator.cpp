#include "snn/activator.hpp"

namespace snn {

template <class T>
std::unique_ptr<activators::base_activator> get_activator_from_type(const std::string& type,
                                                                    T args) {
    if (type == activators::sigmoid::type())
        return std::make_unique<activators::sigmoid>(args);
    else if (type == activators::relu::type())
        return std::make_unique<activators::relu>(args);
    else if (type == activators::tanh::type())
        return std::make_unique<activators::tanh>(args);
    throw std::runtime_error("Invalid activator type: " + type);
}

void activator::create(const kwargs& args) {
    auto [type, sub_args] = args.get_function("");
    activator_m = get_activator_from_type<const kwargs&>(type, sub_args);
}

void activator::load(std::istream& is) {
    uint32_t type_length;
    is.read(reinterpret_cast<char*>(&type_length), sizeof(uint32_t));
    std::string type(type_length, ' ');
    is.read(reinterpret_cast<char*>(&type[0]), type_length);
    activator_m = get_activator_from_type<std::istream&>(type, is);
}

void activator::save(std::ostream& os) const {
    std::string type = activator_m->name();
    uint32_t type_length = type.length();
    os.write(reinterpret_cast<const char*>(&type_length), sizeof(uint32_t));
    os.write(reinterpret_cast<const char*>(&type[0]), type_length);
    activator_m->save(os);
}

std::string activator::name() const { return activator_m->name(); }

tensor<real> activator::f(const tensor<real>& t) const { return activator_m->f(t); }

tensor<real> activator::df(const tensor<real>& t) const { return activator_m->df(t); }

} // namespace snn