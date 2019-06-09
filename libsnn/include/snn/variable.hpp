#pragma once

#include "snn/initializer.hpp"
#include "snn/optimizer.hpp"

namespace snn {

class variable {
    initializer init_m;
    optimizer opt_m;

public:
    tensor<real> var;

    variable();

    void create(const kwargs&);

    void set_optimizer(const kwargs&);

    void optimize(const tensor<real>&);

    void save(std::ostream& os, bool save_gradient) const;

    void load(std::istream& is);
};

} // namespace snn