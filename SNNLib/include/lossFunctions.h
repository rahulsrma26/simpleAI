#ifndef __NEURAL_NETWORK_LOSS_FUNCTIONS_H__
#define __NEURAL_NETWORK_LOSS_FUNCTIONS_H__

#include "nntypes.h"
#include "NNLayer.h"
#include <cmath>

namespace simpleNN {

    using LossCalculate = std::function<real(const realVector&, const realVector&)>;
    using LossDerivative = std::function<realVector(const realVector&, const realVector&)>;
    using LossFunction = std::pair<LossCalculate, LossDerivative>;

    namespace loss {
        const static LossFunction quadratic = { {
                [](const realVector& output, const realVector& label) {
                    real loss = 0.0f;
                    for (uint i = 0; i < output.size(); ++i) {
                        real difference = output[i] - label[i];
                        loss += difference*difference;
                    }
                    return loss / 2.0f;
                }
            },{
                [](const realVector& output, const realVector& label) {
                    realVector difference = output;
                    for (uint i = 0; i < difference.size(); ++i)
                        difference[i] -= label[i];
                    return difference;
                }
            }
        };

        constexpr real lossEpsilon = 1e-5f;

        const static LossFunction crossEntropy = { {
                [](const realVector& output, const realVector& label) {
                    real loss = 0.0f;
                    for (uint i = 0; i < output.size(); ++i)
                        loss += output[i] * log(lossEpsilon + label[i])
                            + (1.0f - output[i]) * log(1.0f + lossEpsilon - label[i]);
                    return -loss;
                }
            },{
                [](const realVector& output, const realVector& label) {
                    realVector gradient(output.size());
                    for (uint i = 0; i < gradient.size(); ++i) {
                        const real denom = output[i] * (1.0f - output[i]);
                        gradient[i] = (output[i] - label[i]) / (lossEpsilon + denom);
                    }
                    return gradient;
                }
            }
        };

        // only positive values ideally between 0 and 1
        const static LossFunction hillinger = { {
                [](const realVector& output, const realVector& label) {
                    real loss = 0.0f;
                    for (uint i = 0; i < output.size(); ++i) {
                        real difference = sqrt(output[i]) - sqrt(label[i]);
                        loss += difference*difference;
                    }
                    return loss / sqrt(2.0f);
                }
            },{
                [](const realVector& output, const realVector& label) {
                    realVector gradient(output.size());
                    for (uint i = 0; i < gradient.size(); ++i) {
                        const real denom = sqrt(2.0f * output[i]);
                        gradient[i] = (sqrt(output[i]) - sqrt(label[i])) / (lossEpsilon + denom);
                    }
                    return gradient;
                }
            }
        };
    }
}

#endif // !__NEURAL_NETWORK_LOSS_FUNCTIONS_H__
