#pragma once

#include "snn/optimizer/base_optimizer.hpp"

namespace snn {
namespace optimizers {

class adagrad : public base_optimizer{
    double learning_rate_m, decay_m;
    double eps_m;
    tensor<real> cache_m;

public:
    adagrad(const tensor<real>&, const kwargs&);
    static std::string type();
    virtual std::string name() const override;
    virtual void update(tensor<real>&, const tensor<real>&) override;
    virtual void save(std::ostream& os, bool save_gradient) const override;
    adagrad(std::istream& is);
};

} // namespace optimizers
} // namespace snn