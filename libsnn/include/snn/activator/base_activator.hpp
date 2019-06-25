#pragma once

#include "snn/nntypes.hpp"
#include "snn/misc/kwargs.hpp"
#include "snn/math/tensor.hpp"

namespace snn {
namespace activators {

class base_activator {

public:
    virtual ~base_activator() = default;
    virtual tensor<real> f(const tensor<real>&) const = 0;
    virtual tensor<real> df(const tensor<real>&) const = 0;
    virtual std::string name() const = 0;
    virtual void save(std::ostream& os) const = 0;
};

} // namespace activators
} // namespace snn
