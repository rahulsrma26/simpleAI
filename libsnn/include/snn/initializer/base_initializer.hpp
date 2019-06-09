#pragma once

#include <iostream>
#include "snn/nntypes.hpp"
#include "snn/misc/string_constant.hpp"
#include "snn/misc/kwargs.hpp"
#include "snn/misc/random.hpp"
#include "snn/math/tensor.hpp"

namespace snn {
namespace initializers {

class base_initializer {

public:
    virtual void init(tensor<real>&) = 0;
    virtual std::string name() const = 0;
};

} // namespace initializers
} // namespace snn
