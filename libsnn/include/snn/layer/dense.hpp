#pragma once

#include "snn/layer/base_layer.hpp"

namespace snn {
namespace layers {

class dense : public base_layer{
    variable weight_m, bias_m;
    tensor<real> input_m, weighted_input_m;
    activator activator_m;
    bool use_bias_m;
    size_t inputs_m, outputs_m;

public:
    dense(const kwargs&);
    static std::string type();
    virtual std::string name() const override;
    virtual size_t output() const override;
    virtual size_t params() const override;
    virtual void set_optimizer(const kwargs&) override;
    virtual tensor<real> forward(tensor<real>&) override;
    virtual tensor<real> backward(tensor<real>&) override;
    virtual void save(std::ostream& os, bool save_gradient) const override;
    dense(std::istream& is);
};

} // namespace layers
} // namespace snn