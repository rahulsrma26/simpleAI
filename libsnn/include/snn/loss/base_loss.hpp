#pragma once

#include "snn/nntypes.hpp"
#include "snn/misc/kwargs.hpp"
#include "snn/math/tensor.hpp"

namespace snn {
namespace losses {

class base_loss {

public:
    virtual ~base_loss() = default;
    virtual std::string name() const = 0;
    virtual real f(const tensor<real>&, const tensor<real>&) const = 0;
    virtual tensor<real> df(const tensor<real>&, const tensor<real>&) const = 0;
    virtual void save(std::ostream& os) const = 0;
};

} // namespace losses
} // namespace snn
