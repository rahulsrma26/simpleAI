#pragma once

#include "snn/layer/base_layer.hpp"

namespace snn {
namespace layers {

class flatten : public base_layer{
    shapeType input_shape_m;
    shape saved_shape;

public:
    flatten(const kwargs&);
    static std::string type();
    virtual std::string name() const override;
    virtual shape output() const override;
    virtual size_t params() const override;
    virtual void set_optimizer(const kwargs&) override;
    virtual tensor<real> forward(tensor<real>&) override;
    virtual tensor<real> backward(tensor<real>&) override;
    virtual tensor<real> predict(tensor<real>&) override;
    virtual void save(std::ostream& os, bool save_gradient) const override;
    flatten(std::istream& is);
};

} // namespace layers
} // namespace snn