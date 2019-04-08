#ifndef __NEURAL_NETWORK_H_
#define __NEURAL_NETWORK_H_

#include <memory>
#include <algorithm>
#include "nntypes.h"
#include "NNLayer.h"
#include "lossFunctions.h"

namespace simpleNN {

    struct nnStats {
        real loss;
        real accuracy;
    };

    class NeuralNetwork {
        uint inputNeurons;
        std::vector<std::unique_ptr<NNLayer>> layers;
        LossFunction cost;

    public:
        NeuralNetwork(uint, LossFunction cost=loss::quadratic);

        uint inputSize() const;
        size_t numLayers() const;
        uint outputSize() const;

        void add(std::unique_ptr<NNLayer> layer);
        realVector predict(const realVector& input);
        realMatrix predict(const realMatrix& inputs);
        realVector feedForward(const realVector& activation);
        void backPropogate(const realVector& output, const realVector& label);
        void update(real eta);
        void train(const realMatrix& input, const realMatrix& label, 
            uint batchSize=1, real learningRate=0.3);
        nnStats getStats(const realMatrix& input, const realMatrix& label);
    };
}

#endif // !__NEURAL_NETWORK_H_



