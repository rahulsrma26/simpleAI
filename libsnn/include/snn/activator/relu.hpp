#pragma once

#include "snn/activator/base_activator.hpp"

namespace snn {
namespace activators {

class relu : public base_activator {

public:
    relu(const kwargs&);
    static std::string type();
    virtual std::string name() const override;
    virtual tensor<real> f(const tensor<real>&) const override;
    virtual tensor<real> df(const tensor<real>&) const override;
    virtual void save(std::ostream& os) const override;
    relu(std::istream& is);
};

} // namespace activators
} // namespace snn