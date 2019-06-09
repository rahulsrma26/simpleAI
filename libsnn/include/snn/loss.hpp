#pragma once

#include <memory>
#include "snn/loss/base_loss.hpp"
#include "snn/loss/quadratic.hpp"
#include "snn/loss/hillinger.hpp"
#include "snn/loss/cross_entropy.hpp"

namespace snn {

class loss : public losses::base_loss {
    std::unique_ptr<losses::base_loss> loss_m;

public:
    void create(const kwargs&);
    std::string name() const override;

    real f(const tensor<real>& output, const tensor<real>& label) const override;
    tensor<real> df(const tensor<real>& output, const tensor<real>& label) const override;

    void save(std::ostream& os) const override;
    void load(std::istream& is);
};

} // namespace snn