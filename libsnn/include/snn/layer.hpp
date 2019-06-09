#pragma once

#include <memory>
#include "snn/layer/base_layer.hpp"
#include "snn/layer/dense.hpp"

namespace snn {

class layer : public layers::base_layer {
    std::unique_ptr<layers::base_layer> layer_m;

public:
    void create(const kwargs&);
    
	size_t output() const override;
    std::string name() const override;
    size_t params() const override;
    void set_optimizer(const kwargs&) override;
    tensor<real> forward(tensor<real>&) override;
    tensor<real> backward(tensor<real>&) override;
    void save(std::ostream& os, bool save_gradient) const override;
    void load(std::istream& is);
};

} // namespace snn