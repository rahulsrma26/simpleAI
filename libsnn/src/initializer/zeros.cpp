#include "snn/initializer/zeros.hpp"

namespace snn {
namespace initializers {

zeros::zeros(const kwargs& args) { std::ignore = args; }

std::string zeros::type() { return TEXT::ZEROS; }

std::string zeros::name() const { return this->type(); }

void zeros::init(tensor<real>& t) {
    for (auto& x : t)
        x = 0;
}

} // namespace initializers
} // namespace snn