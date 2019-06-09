#pragma once

#include "snn/nntypes.hpp"
#include "snn/misc/kwargs.hpp"
#include "snn/math/tensor.hpp"

namespace snn {
namespace optimizers {

class base_optimizer {

public:
    virtual std::string name() const = 0;
    virtual void update(tensor<real>&, const tensor<real>&) = 0;
    virtual void save(std::ostream& os, bool save_gradient) const = 0;
};

} // namespace optimizers
} // namespace snn
