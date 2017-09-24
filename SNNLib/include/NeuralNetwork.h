#ifndef __NEURAL_NETWORK_H_
#define __NEURAL_NETWORK_H_

#include <memory>
#include "nntypes.h"
#include "NNLayer.h"
#include "lossFunctions.h"

namespace simpleNN {

    struct nnStats {
        real loss;
        real accuracy;
    };

    class NeuralNetwork {
        size_t inputNeurons;
        std::vector<std::unique_ptr<NNLayer>> layers;
        LossFunction cost;

    public:
        NeuralNetwork(size_t, LossFunction cost=loss::quadratic);

        size_t inputSize() const;
        size_t numLayers() const;
        size_t outputSize() const;

        void add(std::unique_ptr<NNLayer> layer);
        realVector feedForward(const realVector& activation);
        void backPropogate(const realVector& output, const realVector& label);
        void update(real eta);
        void train(const realMatrix& input, const realMatrix& label, 
            size_t batchSize=1, real learningRate=0.3);
        nnStats getStats(const realMatrix& input, const realMatrix& label);
    };
}

#endif // !__NEURAL_NETWORK_H_



