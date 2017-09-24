#ifndef __NEURAL_NETWORK_LOSS_FUNCTIONS_H__
#define __NEURAL_NETWORK_LOSS_FUNCTIONS_H__

#include "nntypes.h"
#include <cmath>

namespace simpleNN {

    using LossCalculate = std::function<real(const realVector&, const realVector&)>;
    using LossDerivative = std::function<realVector(const realVector&, const realVector&)>;
    using LossFunction = std::pair<LossCalculate, LossDerivative>;

    namespace loss {
        const static LossFunction quadratic = { {
                [](const realVector& output, const realVector& label) {
                    real loss = 0.0;
                    for (size_t i = 0; i < output.size(); ++i) {
                        real difference = output[i] - label[i];
                        loss += difference*difference;
                    }
                    return loss / 2.0;
                }
            },{
                [](const realVector& output, const realVector& label) {
                    realVector difference = output;
                    for (size_t i = 0; i < difference.size(); ++i)
                        difference[i] -= label[i];
                    return difference;
                }
            }
        };
    }
}

#endif // !__NEURAL_NETWORK_LOSS_FUNCTIONS_H__
