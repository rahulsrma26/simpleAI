#pragma once

#include "snn/nntypes.hpp"
#include "snn/misc/string_constant.hpp"
#include "snn/math/tmath.hpp"
#include "snn/misc/kwargs.hpp"
#include "snn/activator.hpp"
#include "snn/variable.hpp"

namespace snn {
namespace layers {

class base_layer {

public:
    virtual ~base_layer() = default;
    virtual std::string name() const = 0;
    virtual std::string info() const = 0;
    virtual shape output() const = 0;
    virtual size_t params() const = 0;
    virtual void set_optimizer(const kwargs&) = 0;
    virtual tensor<real> forward(tensor<real>&) = 0;
    virtual tensor<real> backward(tensor<real>&) = 0;
    virtual tensor<real> predict(tensor<real>&) = 0;
    virtual void save(std::ostream& os, bool save_gradient) const = 0;
};

} // namespace layers
} // namespace snn
