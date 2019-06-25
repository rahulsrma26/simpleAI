#pragma once

#include "snn/layer/base_layer.hpp"

namespace snn {
namespace layers {

class dropout : public base_layer{
    tensor<real> weight_m;
    double rate_m;
    shape inputs_m;
    std::bernoulli_distribution generator_m;

public:
    dropout(const kwargs&);
    static std::string type();
    virtual std::string name() const override;
    virtual std::string info() const override;
    virtual shape output() const override;
    virtual size_t params() const override;
    virtual void set_optimizer(const kwargs&) override;
    virtual tensor<real> forward(tensor<real>&) override;
    virtual tensor<real> backward(tensor<real>&) override;
    virtual tensor<real> predict(tensor<real>&) override;
    virtual void save(std::ostream& os, bool save_gradient) const override;
    dropout(std::istream& is);
};

} // namespace layers
} // namespace snn