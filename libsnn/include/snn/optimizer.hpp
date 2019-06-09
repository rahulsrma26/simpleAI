#pragma once

#include <memory>
#include "snn/optimizer/base_optimizer.hpp"
#include "snn/optimizer/sgd.hpp"

namespace snn {

class optimizer : public optimizers::base_optimizer {
    std::unique_ptr<optimizers::base_optimizer> optimizer_m;

public:
    void create(const tensor<real>&, const kwargs&);
    std::string name() const override;
    void update(tensor<real>&, const tensor<real>&) override;
    void save(std::ostream& os, bool save_gradient) const override;
    void load(std::istream& is);
};

} // namespace snn