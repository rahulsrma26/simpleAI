#pragma once

#include "snn/optimizer/base_optimizer.hpp"

namespace snn {
namespace optimizers {

class adam : public base_optimizer{
    double learning_rate_m, decay_m;
    double beta1_m, beta2_m, eps_m;
    tensor<real> momentum_m, velocity_m;

public:
    adam(const tensor<real>&, const kwargs&);
    static std::string type();
    virtual std::string name() const override;
    virtual void update(tensor<real>&, const tensor<real>&) override;
    virtual void save(std::ostream& os, bool save_gradient) const override;
    adam(std::istream& is);
};

} // namespace optimizers
} // namespace snn