#pragma once

#include <sstream>
#include "snn/model/sequential.hpp"

namespace snn {
namespace models {

class autoencoder : public base_model {
    loss loss_m;

public:
    sequential encoder;
    sequential decoder;
    void compile(const kwargs&);
    void summary() override;
    void run(const tensor<real>&, const tensor<real>&, const kwargs& args = "");
    tensor<real> encode(const tensor<real>&, const kwargs& args = "");
    tensor<real> decode(const tensor<real>&, const kwargs& args = "");
    void save(std::ostream& os, bool save_gradient = true) const override;
    void load(std::istream& is) override;
};

} // namespace models
} // namespace snn
