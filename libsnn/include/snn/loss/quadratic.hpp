#pragma once

#include "snn/loss/base_loss.hpp"
#include "snn/math/tmath.hpp"

namespace snn {
namespace losses {

class quadratic : public base_loss {

public:
    quadratic(const kwargs&);
    static std::string type();
    virtual std::string name() const override;
    virtual real f(const tensor<real>&, const tensor<real>&) const override;
    virtual tensor<real> df(const tensor<real>&, const tensor<real>&) const override;
    virtual void save(std::ostream& os) const override;
    quadratic(std::istream& is);
};

} // namespace losses
} // namespace snn
