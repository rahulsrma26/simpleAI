#ifndef __NEURAL_NETWORK_H_
#define __NEURAL_NETWORK_H_

#include <memory>
#include "nntypes.h"
#include "NNLayer.h"
#include "activators.h"

namespace simpleNN {

    class NeuralNetwork {
        size_t inputNeurons;
        std::vector<std::unique_ptr<NNLayer>> layers;

    public:
        NeuralNetwork(size_t);

        size_t inputSize() const;
        size_t numLayers() const;
        size_t outputSize() const;

        void add(std::unique_ptr<NNLayer>);
        realVector feedForward(const realVector&);
    };
}

#endif // !__NEURAL_NETWORK_H_



