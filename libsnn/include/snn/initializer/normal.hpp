#pragma once

#include "snn/initializer/base_initializer.hpp"

namespace snn {
namespace initializers {

class normal : public base_initializer {
    std::normal_distribution<real> distribution_m;

public:
    /*
	Default args:
		  mean=0.0, stddev=0.05
	*/
    normal(const kwargs&);
    static std::string type();
    virtual std::string name() const override;

    virtual void init(tensor<real>&) override;
};

} // namespace initializers
} // namespace snn