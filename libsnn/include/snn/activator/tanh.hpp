#pragma once

#include "snn/activator/base_activator.hpp"
#include "snn/math/tmath.hpp"

namespace snn {
namespace activators {

class tanh : public base_activator {

public:
    tanh(const kwargs&);
    static std::string type();
    virtual std::string name() const override;
    virtual tensor<real> f(const tensor<real>&) const override;
    virtual tensor<real> df(const tensor<real>&) const override;
    virtual void save(std::ostream& os) const override;
    tanh(std::istream& is);
};

} // namespace activators 
} // namespace snn
