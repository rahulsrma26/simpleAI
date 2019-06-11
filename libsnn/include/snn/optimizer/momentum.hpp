#pragma once

#include "snn/optimizer/base_optimizer.hpp"

namespace snn {
namespace optimizers {

class momentum : public base_optimizer{
    double learning_rate_m;
    double decay_m;
    double moment_m;
    tensor<real> velocity_m;

public:
    momentum(const tensor<real>&, const kwargs&);
    static std::string type();
    virtual std::string name() const override;
    virtual void update(tensor<real>&, const tensor<real>&) override;
    virtual void save(std::ostream& os, bool save_gradient) const override;
    momentum(std::istream& is);
};

} // namespace optimizers
} // namespace snn