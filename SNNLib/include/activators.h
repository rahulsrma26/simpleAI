#ifndef __NEURAL_NETWORK_ACTIVATORS_H__
#define __NEURAL_NETWORK_ACTIVATORS_H__

#include "nntypes.h"
#include <cmath>
#include <functional>

namespace simpleNN {
    using VectorProcessor = std::function<realVector(const realVector&)>;

    using Activator = std::pair<VectorProcessor, VectorProcessor>;

    namespace activators {
        const static Activator sigmoid = { {
                [](const realVector& in) {
                    realVector out = in;
                    for (auto& x : out)
                        x = 1.0 / (1.0 + exp(-x));
                    return out;
                }
            },{
                [](const realVector& in) {
                    realVector out = in;
                    for (auto& x : out)
                        x = -exp(-x);
                    return out;
                }
            }
        };
    }
}

#endif // !__NEURAL_NETWORK_ACTIVATORS_H__
