#pragma once

#include <memory>
#include "snn/initializer/base_initializer.hpp"
#include "snn/initializer/zeros.hpp"
#include "snn/initializer/normal.hpp"
#include "snn/initializer/xavier.hpp"

namespace snn {

class initializer : public initializers::base_initializer {
    std::unique_ptr<initializers::base_initializer> initializer_m;

public:
    void create(const kwargs&);
    std::string name() const override;

    void init(tensor<real>&) override;
};

} // namespace snn