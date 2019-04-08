#include "DenseLayer.h"

namespace simpleNN {

DenseLayer::DenseLayer(int _numNeurons, Activator _activator)
    : NNLayer(_numNeurons), activator(_activator) {}

void DenseLayer::initialize(int numInputConnections) {
    inputConnections = numInputConnections;

    input = realVector(inputConnections);
    weightedInput = realVector(numNeurons);

    bias = realVector(numNeurons);
    for (auto& x : bias)
        x = Gaussian(nnRandomEngine);

    weights = realMatrix(numNeurons, realVector(inputConnections));
    for (auto& row : weights)
        for (auto& x : row)
            x = Gaussian(nnRandomEngine);

    deltaB = realVector(numNeurons, 0.0f);
    deltaW = realMatrix(numNeurons, realVector(inputConnections, 0.0f));
}

realVector DenseLayer::getInput() const { return input; }

realVector DenseLayer::predict(const realVector& previousActivation) {
    return forward(previousActivation);
}

realVector DenseLayer::forward(const realVector& previousActivation) {
    std::copy(previousActivation.begin(), previousActivation.end(), input.begin());
#pragma omp parallel for
    for (int i = 0; i < numNeurons; ++i) {
        real sum = 0.0;
        for (int j = 0; j < inputConnections; ++j)
            sum += input[j] * weights[i][j];
        weightedInput[i] = sum + bias[i];
    }
    return activator.first(weightedInput);
}

realVector DenseLayer::backward(const realVector& activationDelta) {
    realVector delta = activator.second(weightedInput);
    for (int i = 0; i < numNeurons; ++i)
        deltaB[i] += (delta[i] *= activationDelta[i]);

#pragma omp parallel for
    for (int i = 0; i < numNeurons; ++i)
        for (int j = 0; j < inputConnections; ++j)
            deltaW[i][j] += delta[i] * input[j];

    ++deltaN;

    realVector activationDeltaPrev(inputConnections);
#pragma omp parallel for
    for (int j = 0; j < inputConnections; ++j) {
        real sum = 0.0;
        for (int i = 0; i < numNeurons; ++i)
            sum += weights[i][j] * delta[i];
        activationDeltaPrev[j] = sum;
    }
    return activationDeltaPrev;
}

void DenseLayer::update(real eta) {
    for (int i = 0; i < numNeurons; ++i) {
        bias[i] -= eta * deltaB[i] / deltaN;
        deltaB[i] = 0.0f;
    }

#pragma omp parallel for
    for (int i = 0; i < numNeurons; ++i)
        for (int j = 0; j < inputConnections; ++j) {
            weights[i][j] -= eta * deltaW[i][j] / deltaN;
            deltaW[i][j] = 0.0f;
        }

    deltaN = 0;
}

} // namespace simpleNN