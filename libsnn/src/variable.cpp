#include "snn/variable.hpp"

namespace snn {

variable::variable() : var({1}) {}

void variable::create(const kwargs& args) {
    auto shape_vector = args.get_int_vector("shape");
    shape sh(shape_vector.begin(), shape_vector.end());
    var = tensor<real>(sh);
    if (args.has_key("initializer")) {
		init_m.create(args.get_string("initializer").c_str());
		init_m.init(var);
    }
}

void variable::set_optimizer(const kwargs& args) {
    opt_m.create(var, args);
}

void variable::optimize(const tensor<real>& g) { opt_m.update(var, g); }

void variable::save(std::ostream& os, bool save_gradient) const {
    var.to_stream(os);
    opt_m.save(os, save_gradient);
};

void variable::load(std::istream& is) {
    var = tensor<real>::from_stream(is);
    opt_m.load(is);
};

} // namespace snn