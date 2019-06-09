#include "snn/initializer.hpp"

namespace snn {

void initializer::create(const kwargs& args) {
    auto [type, sub_args] = args.get_function("");
    if (type == initializers::zeros::type())
        initializer_m = std::make_unique<initializers::zeros>(sub_args);
    else if (type == initializers::normal::type())
        initializer_m = std::make_unique<initializers::normal>(sub_args);
    else if (type == initializers::xavier::type())
        initializer_m = std::make_unique<initializers::xavier>(sub_args);
    else
        throw std::runtime_error("Invalid initializer type: " + type);
}

std::string initializer::name() const { return initializer_m->name(); }

void initializer::init(tensor<real>& t) { initializer_m->init(t); }

} // namespace snn