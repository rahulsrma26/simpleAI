#pragma once

#include "snn/nntypes.hpp"
#include "snn/misc/string_constant.hpp"
#include "snn/loss.hpp"
#include "snn/layer.hpp"
#include "snn/util/progress_bar.hpp"

namespace snn {
namespace models {

class base_model {
protected:
    uint64_t batches_m;
    double avg_loss_m;

public:
    virtual ~base_model() = default;
    virtual void summary() = 0;
    virtual void run(const tensor<real>&, const tensor<real>&, const kwargs&) = 0;
    virtual tensor<real> predict(const tensor<real>&, const kwargs&) = 0;
    virtual void save(std::ostream& os, bool save_gradient = true) const = 0;
    virtual void load(std::istream& is) = 0;
};

} // namespace models
} // namespace snn
