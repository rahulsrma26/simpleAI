#pragma once

#include <memory>
#include "snn/activator/base_activator.hpp"
#include "snn/activator/sigmoid.hpp"
#include "snn/activator/relu.hpp"
#include "snn/activator/tanh.hpp"

namespace snn {

class activator : public activators::base_activator {
    std::unique_ptr<activators::base_activator> activator_m;

public:
    void create(const kwargs&);
    std::string name() const override;

    tensor<real> f(const tensor<real>&) const override;
    tensor<real> df(const tensor<real>&) const override;
    void save(std::ostream& os) const override;
    void load(std::istream& is);
};

} // namespace snn