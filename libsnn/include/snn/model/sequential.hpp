#pragma once

#include <sstream>
#include "snn/model/base_model.hpp"

namespace snn {
namespace models {

class sequential : public base_model {
    std::vector<layer> layers_m;
    loss loss_m;

public:
    void add(const kwargs&);
    void compile(const kwargs&);
    void summary() override;
    void run(const tensor<real>&, const tensor<real>&, const kwargs& args = "") override;
    tensor<real> predict(const tensor<real>&, const kwargs& args = "") override;
    void save(std::ostream& os, bool save_gradient = true) const override;
    void load(std::istream& is) override;
};

} // namespace models
} // namespace snn
