#pragma once

#include "snn/initializer/base_initializer.hpp"

namespace snn {
namespace initializers {

class xavier : public base_initializer {
    bool uniform;

public:
    xavier(const kwargs&);
    static std::string type();
    virtual std::string name() const override;

    virtual void init(tensor<real>&) override;
};

} // namespace initializers
} // namespace snn