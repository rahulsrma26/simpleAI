#include "stdafx.h"
#include "Dropout.h"

namespace simpleNN {

    Dropout::Dropout(real dropoutProbability)
        : NNLayer(0), probability(dropoutProbability)
    { }

    void Dropout::initialize(int numInputConnections) {
        numNeurons = inputConnections = numInputConnections;
        uint activeNeurons = static_cast<uint>(inputConnections * (1 - probability));
        active = realVector(inputConnections, 0);
        std::fill(active.begin(), active.begin() + activeNeurons, 1.0f);
        update(0);
    }

    realVector Dropout::getInput() const {
        throw std::invalid_argument("dropout can't be used as last layer");
        return {};
    }

    realVector Dropout::predict(const realVector& previousActivation) {
        auto activation = previousActivation;
        const real factor = 1 - probability;
        for (int i = 0; i < numNeurons; ++i)
            activation[i] *= factor;
        return activation;
    }

    realVector Dropout::forward(const realVector& previousActivation) {
        auto activation = previousActivation;
        for (int i = 0; i < numNeurons; ++i)
            activation[i] *= active[i];
        return activation;
    }

    realVector Dropout::backward(const realVector& activationDelta) {
        auto activationDeltaPrev = activationDelta;
        for (int i = 0; i < inputConnections; ++i)
            activationDeltaPrev[i] *= active[i];
        return activationDeltaPrev;
    }

    void Dropout::update(real eta) {
        std::shuffle(active.begin(), active.end(), nnRandomEngine);
    }

}
