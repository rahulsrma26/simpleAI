#include "stdafx.h"
#include "DenseLayer.h"

namespace simpleNN {

    DenseLayer::DenseLayer(size_t _numNeurons, Activator _activator)
        : NNLayer(_numNeurons), activator(_activator)
    {
    }

    void DenseLayer::initialize(size_t inputs) {
        bias = realVector(numNeurons);
        for (auto& x : bias)
            x = Gaussian(nnRandomEngine);

        weights = realMatrix(numNeurons, realVector(inputs));
        for (auto& row : weights)
            for (auto& x : row)
                x = Gaussian(nnRandomEngine);
    }

    realVector DenseLayer::forward(const realVector& input) {
        auto out = bias;
        for (size_t i = 0; i < numNeurons; i++) {
            real dot = 0.0;
            for (size_t j = 0; j < input.size(); j++)
                dot += input[j] * weights[i][j];
            out[i] += dot;
        }
        return activator.first(out);
    }

}