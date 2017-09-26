#ifndef __NEURAL_NETWORK_ACTIVATORS_H__
#define __NEURAL_NETWORK_ACTIVATORS_H__

#include "nntypes.h"
#include <cmath>
#include <numeric>
#include <functional>

namespace simpleNN {

    using Activator = std::pair<VectorProcessor, VectorProcessor>;

    namespace activators {
        const static Activator sigmoid = { {
                [](const realVector& in) {
                    realVector out = in;
                    for (auto& x : out)
                        x = 1 / (1 + std::exp(-x));
                    return out;
                }
            },{
                [](const realVector& in) {
                    realVector out = in;
                    for (auto& x : out) {
                        const real v = 1 / (1 + std::exp(-x));
                        x = v*(1 - v);
                    }
                    return out;
                }
            }
        };

        const static Activator tanh = { {
                [](const realVector& in) {
                    realVector out = in;
                    for (auto& x : out)
                        x = std::tanh(x);
                    return out;
                }
            },{
                [](const realVector& in) {
                    realVector out = in;
                    for (auto& x : out) {
                        const real v = std::tanh(x);
                        x = 1 - v*v;
                    }
                    return out;
                }
            }
        };

        const static Activator relu = { {
                [](const realVector& in) {
                    realVector out = in;
                    for (auto& x : out)
                        x = x < 0 ? 0 : x;
                    return out;
                }
            },{
                [](const realVector& in) {
                    realVector out = in;
                    for (auto& x : out)
                        x = x < 0 ? 0.0f : 1.0f;
                    return out;
                }
            }
        };

        const static Activator lrelu = { {
                [](const realVector& in) {
                    realVector out = in;
                    for (auto& x : out)
                        x = x < 0 ? 0.01f*x : x;
                    return out;
                }
            },{
                [](const realVector& in) {
                    realVector out = in;
                    for (auto& x : out)
                        x = x < 0 ? 0.01f : 1.0f;
                    return out;
                }
            }
        };

        const static Activator softmax = { {
                [](const realVector& in) {
                    realVector out = in;
                    for (auto& x : out)
                        x = std::exp(x);
                    real sum = std::accumulate(out.begin(), out.end(), real(0.0));
                    for (auto& x : out)
                        x /= sum;
                    return out;
                }
            },{
                [](const realVector& in) {
                    realVector out = in;
                    for (auto& x : out)
                        x = std::exp(x);
                    real sum = std::accumulate(out.begin(), out.end(), real(0.0));
                    for (auto& x : out) {
                        const real v = x / sum;
                        x = v*(1 - v);
                    }
                    return out;
                }
            }
        };
    }
}

#endif // !__NEURAL_NETWORK_ACTIVATORS_H__
